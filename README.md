# LLM Prompt Injection Detection Pipeline

This independent study project explores various methods to detect and classify malicious prompts aimed at manipulating large language models (LLMs). These include prompt injection attacks, jailbreak attempts, and other adversarial inputs that may result in sensitive information disclosure or insecure outputs.

## Motivation

Detecting prompt injection attacks is critical for building robust AI guardrail systems. This project focuses on:

- Building a scalable pipeline (guardrail) to detect adversarial prompts
- Comparing multiple techniques: classical ML, BERT, and LLM-assisted pipelines
- Studying the tradeoffs in complexity, accuracy, and performance

## Project Structure

- `data/`: Contains raw and processed datasets
- `src/`: All source code for models and preprocessing
- `papers/`: Collection of academic and industry resources
- `results/`: Outputs from experiments (charts, logs, metrics)
- `reports/`: Proposal and final writeups