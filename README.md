# Using Transformer Pipelines to Detect Malicious Prompt Injection in LLMs

This repository contains the complete source code, datasets, and workflow configurations for the research paper, "Using Transformer Pipelines to Detect Malicious Prompt Injection in LLMs".

This study evaluates a spectrum of defensive pipelines against prompt injection, from traditional SVM classifiers to multi-stage LLM-based reasoning and a final, engineered hybrid guardrail. Our results demonstrate the fundamental trade-off between security effectiveness and user experience, arguing that robust LLM security requires a layered, tunable, and risk-aware approach.

---

## Repository Structure

-   **`/data/`**: Contains all datasets used for testing and the final, human-annotated results.
    -   **/testing/**: Holds the clean, un-annotated test sets.
        -   `balanced_test_dataset_100.json`: The 100-sample balanced test set used for evaluating all pipelines.
    -   **/annotated_results/**: Contains the final, human-annotated CSV files for all five pipelines, which serve as the ground truth for the analysis.
-   **`/papers/`**: The research proposal, final paper draft, and related literature.
-   **`/reports/`**: The output directory for all generated reports and charts.
-   **`/src/`**: Contains all Python source code, models, and configurations required to run the project.
    -   **/models/**: Contains the serialized, pre-trained machine learning models (SVM, vectorizer, scaler).
    -   **/routes/**: Contains the blueprints that define the REST API endpoints (`/predict`, `/classify`) for the Flask application.
    -   **/templates/**: Contains the HTML templates for the web-based human annotation tool.
    -   `advanced_svm_classifier.py`: Script for training and saving the SVM classifier.
    -   `app.py`: The main entry point for the Flask REST API server that serves the SVM and BERT models.
    -   `detector.py`: A class containing the core logic for loading and running the SVM models.
    -   `eval_annotator.py`: Launches the web UI for human annotation of test results.
    -   `eval_tools.py`: The main analysis script for generating a detailed report from a single annotated file.
    -   `generate_comparison_plots.py`: The final script that aggregates all individual reports and creates the comprehensive comparison charts for the paper.
    -   `n8n_tester.py`: The primary orchestration script that runs the evaluation by calling n8n webhooks.
    -   `requirements.txt`: Python libraries required to run the scripts in this directory.
-   **`/workflows/`**: The exported JSON files for the n8n-based pipelines.

---

## Project Architecture

This project uses a distributed architecture where Python scripts, run from the `/src/` directory, interact with workflows hosted on an **n8n** server. The n8n workflows, in turn, make calls to two different services:
1.  **OpenRouter:** For accessing large language models (e.g., Llama 3.1 8B).
2.  **Local Flask API Server:** For accessing the locally-hosted SVM and BERT models, which runs from the `/src/` directory.

Correctly configuring the networking between these components is essential for the tests to run.

---

## Setup and Installation

Follow these steps carefully to configure the complete testing environment. The majority of commands should be run from **inside the `/src/` directory**.

### 1. Set Up the Python Environment (from within `/src/`)

**1. Navigate into the source directory:**
```bash
cd src
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

**3. Install dependencies:**
The `requirements.txt` file is located in this directory.
```bash
pip install -r requirements.txt
```

### 2. Run the Local Model API Server (from within `/src/`)

The SVM and BERT pipelines are exposed via a local Flask REST API. This server must be running for n8n to access them.

**1. Open a NEW terminal window** and navigate to the `/src/` directory.

**2. Activate the virtual environment in the new terminal:**
```bash
# Make sure you are in the /src/ directory
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

**3. Start the Flask API server:**
The server runs on port **5000** by default.
```bash
python app.py
```
- **Keep this terminal window open.** It is now serving the SVM and BERT models at `http://127.0.0.1:5000`.

### 3. Set Up and Configure n8n

You will need a running n8n instance to orchestrate the tests.

**1. Set up n8n:**
If you do not have n8n installed, this [YouTube tutorial](https://youtube.com/watch?v=pBy0HZ2ohOA) provides a good guide for getting started. Ensure your n8n instance is running.

**2. Configure Credentials in n8n:**
- **OpenRouter:**
    - In n8n, go to **Credentials > Add credential**.
    - Search for and select **"Header Auth"**.
    - **Credential Name:** `OpenRouter Key`
    - **Header Name:** `Authorization`
    - **Header Value:** `Bearer YOUR_OPENROUTER_API_KEY` (replace with your key)
    - Click **Save**.

**3. Import and Configure the Workflows:**
- Import all workflow JSON files from the `../workflows/` directory into your n8n instance.
- **IMPORTANT:** Open each workflow and perform the following configurations:
    - **Update LLM Nodes:** For theworkflows, click on each "OpenRouter Chat Model" node and select the **"OpenRouter Key"** credential if not already selected.
    - **Update SVM/BERT Nodes:** For any workflow that calls the SVM or BERT models, find the "HTTP Request" node. The URL must point to your local Flask API server:
        - **URL:** `http://127.0.0.1:5000/predict` (for SVM) or `http://127.0.0.1:5000/classify` (for BERT).
    - **Update Callback URL:** Find the final "HTTP Request" node that sends the verdict back to the Python tester script. This must point to your local machine:
        - **URL:** `http://127.0.0.1:8888/callback` (or your specified callback port).

**4. Activate the Workflows:**
- Activate all workflows you intend to test by toggling the switch at the top right of the n8n editor.

### 4. Configure the Pipeline Tester Script

The `src/pipelines.json` file tells the tester script which pipelines to run.
- Open each workflow in n8n, click the initial **"Webhook"** node, copy the **Test URL**, and paste it into the corresponding `webhook_url` field in the file.

---

## How to Run the Full Evaluation

You should now have **three terminal windows open**:
1.  Your main terminal, inside the `/src/` directory with the venv activated.
2.  A terminal running the Flask API server (`app.py`).
3.  Your running n8n instance.

All the following commands are run from your **main terminal inside the `/src/` directory**.

### Step 1: Testing the Pipelines

The `n8n_tester.py` script runs the balanced test set against the pipelines defined in `pipelines.json`.

**Example Command (running all 100 samples):**
```bash
python n8n_tester.py --dataset ../data/testing/balanced_test_dataset_100.json --config pipelines.json --max-concurrent 20 --sample-size 100
```
- The script starts its own callback server on port 8888.
- Raw (un-annotated) CSV files will be generated inside the `src/test_results/` directory.

### Step 2: Human Annotation

Use the `eval_annotator.py` script on the generated CSV files.

**Example Usage:**
```bash
python eval_annotator.py -i test_results/[evaluation].csv
```
- This launches a local web server for annotation.
- A new `..._annotated.csv` file will be created in `src/test_results/` after annotation.

### Step 3: Analyzing the Results

The `eval_tools.py` script generates a detailed report for a single annotated file.

**Example Usage:**
```bash
python eval_tools.py --evaluation-file ../data/annotated_results/[evaluation]_annotated.csv
```
- This will create a new subdirectory inside `src/analysis_reports/`.

### Step 4: Generating Final Comparison Charts

After running the analyzer for all five pipelines, use `generate_comparison_plots.py` to create the final charts.

**Example Usage (uses defaults):**
```bash
python generate_comparison_plots.py
```
- This script scans `src/analysis_reports/` for all the individual JSON reports.
- It saves the final `comparison_plot_tpr.png` and `comparison_plot_fpr.png` inside `src/reports/comparison_charts/`. These are the primary figures for the paper.
