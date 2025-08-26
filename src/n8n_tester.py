import json
import asyncio
import random

import aiohttp
from aiohttp import web
import time
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
import argparse
import uuid
from asyncio import Event

@dataclass
class PipelineConfig:
    name: str
    webhook_url: str
    description: str
    timeout: int = 60


@dataclass
class TestResult:
    test_id: str
    prompt_id: str
    original_label: str
    predicted_label: Optional[str]
    webhook_response: Optional[str]
    ai_response: Optional[str]
    webhook_time: Optional[float]
    total_time: Optional[float]
    error: Optional[str]
    category: Optional[str]
    technique: Optional[str]
    difficulty: Optional[str] # Some of these, including this metric, are remnants of past experiments.
    callback_received: bool = False
    prompt_text: str = ""

class CallbackServer:
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.responses: Dict[str, Dict] = {}
        self.response_events: Dict[str, Event] = {}
        self.runner = None
        self.site = None
        self.logger = logging.getLogger("CallbackServer")

        self.app.middlewares.append(self._cors_middleware)
        self.app.router.add_post('/callback', self.handle_callback)
        self.app.router.add_get('/health', self.handle_health)

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    async def handle_health(self, request: web.Request):
        return web.json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    async def handle_callback(self, request: web.Request):
        try:
            data = await request.json()
            test_id = data.get('test_id')
            if not test_id:
                self.logger.error("Missing test_id in callback data")
                return web.json_response({'error': 'Missing test_id'}, status=400)

            self.logger.info(f"Callback received for test_id: {test_id}")
            self.responses[test_id] = {'ai_response': data.get('response', ''), 'full_payload': data}

            if test_id in self.response_events:
                self.response_events[test_id].set()
            else:
                self.logger.warning(f"No waiting event for test_id: {test_id} (might have timed out)")
            return web.json_response({'status': 'received'}, status=200)
        except Exception as e:
            self.logger.error(f"Callback processing error: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def wait_for_response(self, test_id: str, timeout: float) -> Optional[Dict]:
        if test_id in self.responses:
            return self.responses.pop(test_id, None)

        event = Event()
        self.response_events[test_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self.responses.pop(test_id, None)
        except asyncio.TimeoutError:
            self.logger.error(f"Wait for response timed out for test_id {test_id}")
            return None
        finally:
            self.response_events.pop(test_id, None)

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        self.logger.info(f"Callback server started at http://{self.host}:{self.port}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("Callback server stopped")

class N8NPipelineTester:
    def __init__(self, dataset_path: str, results_dir: str = "test_results",
                 callback_host: str = "localhost", callback_port: int = 8888):
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.callback_server = CallbackServer(callback_host, callback_port)
        self.callback_url = f"http://{callback_host}:{callback_port}/callback"

        log_file = self.results_dir / f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()], force=True)
        self.logger = logging.getLogger("Tester")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.logger.info(f"Loaded {len(self.dataset)} test samples from {self.dataset_path.name}")

    '''
    NOTE: Thought this would be a lot more helpful, but barely works.
    This feature can be utilized properly with larger LLMs, as it seems Llama 3.1 8B really doesn't follow directions too closely and likes adding flair at the end. (e.g. **BENIGN**)
    '''
    def _parse_ai_response(self, response_text: Optional[str]) -> Optional[str]:
        if not response_text or not isinstance(response_text, str):
            return "NO_RESPONSE"

        last_line = response_text.strip().split('\n')[-1].lower()
        if "benign" in last_line:
            return "BENIGN"
        if "suspicious" in last_line:
            return "SUSPICIOUS" # Some models use 'SUSPICIOUS'
        if "malicious" in last_line:
            return "MALICIOUS"

        return "UNPARSABLE"

    async def _test_single_prompt(self, session: aiohttp.ClientSession, pipeline: PipelineConfig,
                                  prompt_data: Dict, prompt_id: str) -> TestResult:
        test_id = str(uuid.uuid4())
        start_time = time.time()
        payload = {"test_id": test_id, "text": prompt_data["text"], "callback_url": self.callback_url}
        self.logger.info(
            f"Testing prompt {prompt_id} (Category: {prompt_data.get('category', 'N/A')}) -> test_id {test_id}")

        webhook_response, webhook_time, ai_response, total_time, error, callback_received, predicted_label = (None,) * 7
        callback_received = False

        try:
            async with session.post(pipeline.webhook_url, json=payload, timeout=30) as response:
                webhook_time = time.time() - start_time
                webhook_response = await response.text()
                response.raise_for_status()

            callback_data = await self.callback_server.wait_for_response(test_id, timeout=pipeline.timeout)

            if callback_data:
                callback_received = True
                ai_response = callback_data.get("ai_response")
                predicted_label = self._parse_ai_response(ai_response)
            else:
                error = "No callback received within timeout"
                predicted_label = "TIMEOUT"

        except Exception as e:
            self.logger.error(f"Test failed for {test_id}: {e}")
            error = str(e)
            predicted_label = "CLIENT_ERROR"

        total_time = time.time() - start_time
        return TestResult(
            test_id=test_id, prompt_id=prompt_id, original_label=prompt_data["label"],
            predicted_label=predicted_label, webhook_response=webhook_response,
            ai_response=ai_response, webhook_time=webhook_time, total_time=total_time,
            error=error, category=prompt_data.get("category"), technique=prompt_data.get("technique"),
            difficulty=prompt_data.get("difficulty"), callback_received=callback_received,
            prompt_text=prompt_data["text"]
        )

    async def test_pipeline(self, pipeline: PipelineConfig, max_concurrent: int,
                            delay_between_requests: float, sample_size: Optional[int]) -> List[TestResult]:
        self.logger.info(f"Testing pipeline: {pipeline.name}")
        dataset_to_test = random.sample(self.dataset, sample_size) if sample_size and sample_size < len(
            self.dataset) else self.dataset

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, data in enumerate(dataset_to_test):
                async def task_with_semaphore(prompt_data=data, prompt_id=f"prompt_{i:03d}"):
                    async with semaphore:
                        result = await self._test_single_prompt(session, pipeline, prompt_data, prompt_id)
                        await asyncio.sleep(delay_between_requests)
                        return result

                tasks.append(asyncio.create_task(task_with_semaphore()))

            for i, f in enumerate(asyncio.as_completed(tasks)):
                result = await f
                results.append(result)
                self.logger.info(
                    f"Progress: {i + 1}/{len(tasks)} | Prompt {result.prompt_id} -> {result.predicted_label}")
        return sorted(results, key=lambda r: r.prompt_id)

    def save_results(self, pipeline_name: str, results: List[TestResult]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{pipeline_name.replace(' ', '_')}_{timestamp}_evaluation.csv"

        results_df = pd.DataFrame([asdict(r) for r in results])
        results_df.rename(columns={'total_time': 'response_time'}, inplace=True)

        human_cols = ["human_verdict_assessment", "human_response_quality", "human_notes"]
        for col in human_cols:
            if col not in results_df.columns:
                results_df[col] = ""

        ordered_cols = [
                           "test_id", "prompt_id", "original_label", "predicted_label", "category",
                           "technique", "difficulty", "response_time", "callback_received", "error",
                           "prompt_text", "ai_response"
                       ] + human_cols
        results_df = results_df[ordered_cols]

        results_df.to_csv(filename, index=False)
        self.logger.info(f"Evaluation file saved: {filename}")
        return filename

    def print_summary(self, pipeline_name: str, results: List[TestResult]):
        print(f"\n{'=' * 60}\nDETAILED SUMMARY: {pipeline_name}\n{'=' * 60}")

        if not results:
            print("No results to summarize.")
            return

        df = pd.DataFrame([asdict(r) for r in results])
        total_tests = len(df)
        successful_callbacks = df['callback_received'].sum()

        print(f"Total Prompts Tested: {total_tests}")
        print(f"Successful Callbacks: {successful_callbacks} ({successful_callbacks / total_tests:.1%})")
        if successful_callbacks > 0:
            avg_time = df[df['callback_received']]['total_time'].mean()
            print(f"Average Response Time: {avg_time:.2f}s")

        print("\n--- Overall Performance ---")
        contingency_table = pd.crosstab(
            df['original_label'],
            df['predicted_label'],
            margins=True
        )
        print(contingency_table.to_string())

        print("\n--- Performance by Category ---")
        category_groups = df.groupby('category')
        for category, group in category_groups:
            total_in_category = len(group)
            print(f"\nCategory: {category} ({total_in_category} total)")

            predictions = group['predicted_label'].value_counts().sort_index()
            for verdict, count in predictions.items():
                print(f"  - Predicted {verdict:<12}: {count:<3} ({count / total_in_category:.1%})")

        print(f"{'=' * 60}")


async def main():
    parser = argparse.ArgumentParser(description='n8n Pipeline Evaluation Tester')
    parser.add_argument('--dataset', required=True, help='Path to the test dataset JSON file.')
    parser.add_argument('--config', required=True, help='Path to the pipeline config JSON file.')
    parser.add_argument('--sample-size', type=int, help='Optional: Number of samples to test from the dataset.')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent requests to n8n.')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay between requests in seconds.')
    parser.add_argument('--callback-port', type=int, default=8888, help='Port for the local callback server.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = json.load(f)
    pipelines = [PipelineConfig(**p) for p in config_data['pipelines']]

    tester = N8NPipelineTester(args.dataset, "test_results", "localhost", args.callback_port)

    try:
        await tester.callback_server.start()
        for pipeline in pipelines:
            results = await tester.test_pipeline(
                pipeline, args.max_concurrent, args.delay, args.sample_size
            )
            if results:
                tester.save_results(pipeline.name, results)
                tester.print_summary(pipeline.name, results)
            else:
                tester.logger.warning(f"No results generated for pipeline: {pipeline.name}")
    except Exception as e:
        tester.logger.error(f"An unexpected error occurred during the main execution: {e}", exc_info=True)
    finally:
        await tester.callback_server.stop()


if __name__ == "__main__":
    asyncio.run(main())