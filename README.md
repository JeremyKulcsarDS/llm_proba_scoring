# LLM Performance Measurement Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This Python module provides a framework for measuring the performance of Large Language Models (LLMs) following different dimensions using a probabilistic evaluation technique. It aims at reducing the evaluation uncertainty cause by the probabilistic nature of the LLMs, by relying notably on checker LLMs that return a yes/no answer upon verifying a condition, and approaching the set of answers statistically.

The module offers a set of functions and utilities that enable you to:

- Measure the performance of LLMs based on probabilistic evaluation techniques.
- Compare the performance of different LLMs.
- Generate reports and visualizations to analyze and interpret the results.

## Features

General usages:
- **Correctness**: Check if the answer replies the initial question
- **Bias**: Check for presence of racial, gender, ethnic discrimination
- **Toxicity**: Check for presence of toxic or harmful content
- **In-chat memory**: Check if the model can remember relevant information between two queries of different context (On the roadmap)
- **Expectancy**: Check if the answer comes as expected (On the roadmap)

RAG-specific
- **Context Relevance (user query VS context)**: Is the retrieval process able to retrieve the relevant context to the question? (On the roadmap)
- **Context Adherence**: Is the model's response consistent with the information in the context? (On the roadmap)
- **Completeness**: Is the relevant information in the context fully reflected in the model's response? (On the roadmap)
- **Chunk utilization**: For each retrieved chunk, how much does one contribute to the final response? (On the roadmap)

## Installation

To install the project, clone this repository and copy to your python site-packages directory:

```bash
git clone https://github.com/JeremyKulcsarDS/llm_proba_scoring.git
cd llm_proba_scoring
cp -r llm_proba_scoring /usr/local/lib/python3.8/site-packages/ # replace with your own site-packages directory
```

## Contribution

If you want to contribute, feel free to fork the repository and submit pull requests. If you found any bugs or have suggestions, please create an issue in the [issues](https://github.com/JeremyKulcsarDS/llm_proba_scoring/issues) section.

## Contact

For questions or feedback, you can reach out to jeremy.kulcsar@diamondhill.io.

---

Cheers, 
Jeremy