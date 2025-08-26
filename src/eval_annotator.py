# evaluation_annotator.py
import pandas as pd
import argparse
import webbrowser
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash

''' 
NOTE: This was made with the HELP of ChatGPT - it is nothing more than a simple evaluation tool.

Hosts a webpage where a human can view and evaluate the pipeline verdicts one by one.
'''

def create_app(input_filepath: Path):
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    app.secret_key = 'super-secret-key-for-evaluation'

    # --- Configuration ---
    app.config['INPUT_FILEPATH'] = input_filepath
    p = input_filepath
    app.config['OUTPUT_FILEPATH'] = p.parent / f"{p.stem}_annotated.csv"

    # --- Data Loading and Preparation ---
    if app.config['OUTPUT_FILEPATH'].exists():
        print(f"Resuming from existing annotated file: {app.config['OUTPUT_FILEPATH']}")
        df = pd.read_csv(app.config['OUTPUT_FILEPATH'])
    else:
        print(f"Starting new annotation for: {app.config['INPUT_FILEPATH']}")
        df = pd.read_csv(app.config['INPUT_FILEPATH'])

    # CHANGED: Use a single, more descriptive set of evaluation columns
    evaluation_cols = ['human_verdict', 'human_response_quality', 'human_notes']
    for col in evaluation_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # REMOVED: Drop old, confusing columns if they exist
    old_cols = ['human_correct_block', 'human_false_positive']
    df.drop(columns=[c for c in old_cols if c in df.columns], inplace=True)

    app.config['DATASET'] = df

    @app.route("/")
    def index():
        """Show the next item to evaluate or the completion screen."""
        df = app.config['DATASET']

        # CHANGED: Find the next unevaluated item based on the new 'human_verdict' column
        unevaluated = df[pd.isna(df['human_verdict'])]

        if unevaluated.empty:
            return render_template("complete.html", output_path=app.config['OUTPUT_FILEPATH'])

        # Get the next item and its index
        item_index = int(unevaluated.index[0])
        item = unevaluated.iloc[0].to_dict()

        # Calculate progress
        total = len(df)
        completed = total - len(unevaluated)
        progress = int((completed / total) * 100) if total > 0 else 0

        return render_template(
            "evaluate.html",
            item=item,
            item_index=item_index,
            progress=progress,
            completed=completed,
            total=total
        )

    @app.route("/save", methods=["POST"])
    def save():
        """Save the evaluation and move to the next item."""
        df = app.config['DATASET']
        try:
            item_index = int(request.form['item_index'])

            # CHANGED: Save the new single verdict and other fields
            df.loc[item_index, 'human_verdict'] = request.form.get('human_verdict')
            df.loc[item_index, 'human_response_quality'] = request.form.get('quality_rating', '')
            df.loc[item_index, 'human_notes'] = request.form.get('notes', '').strip()

            # Save to the file immediately after each annotation
            df.to_csv(app.config['OUTPUT_FILEPATH'], index=False)

            flash("Annotation saved!", "success")
        except KeyError:
            flash("Invalid form submission. Please try again.", "danger")
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")

        return redirect(url_for('index'))

    return app


def main():
    parser = argparse.ArgumentParser(description="Web-based annotation tool for AI pipeline results")
    parser.add_argument('-i', '--input-file', required=True, help="Input CSV file from the n8n tester")
    parser.add_argument('-p', '--port', type=int, default=5001, help="Port to run the web server on")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    app = create_app(input_filepath=input_path)

    print("\n" + "=" * 50)
    print("         ANNOTATION TOOL STARTED")
    print("=" * 50)
    print(f"Input file: {input_path.name}")
    print(f"Annotated results will be saved to: {app.config['OUTPUT_FILEPATH'].name}")
    print(f"\nOpen this URL in your browser: http://127.0.0.1:{args.port}")
    print("\nPress CTRL+C to stop the server")
    print("=" * 50)

    webbrowser.open_new(f"http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()