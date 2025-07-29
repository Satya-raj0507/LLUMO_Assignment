# LLM Log Evaluation via Direct Scoring

This script provides a fast, direct method for evaluating Large Language Model (LLM) performance based on a log file. It bypasses formal evaluation libraries in favor of using an LLM to score itself.

---
## Approach

The core approach is to use a powerful LLM (like Google's Gemini or OpenAI's GPT) as an "expert evaluator" to score the outputs found in a `logs.json` file.

The process for each log entry is as follows:

1.  **Log Parsing**: The script reads the `logs.json` file and, for each item, extracts three key pieces of information:
    * **Context**: The entire raw content from the `system` role's prompt.
    * **Question**: The content from the `user` role's prompt.
    * **Answer**: The generated output from the `assistant` role.

2.  **Dynamic LLM Prompting**: A detailed prompt is constructed for each item. This prompt instructs the evaluator LLM to analyze the `ANSWER` based on the provided `CONTEXT` and `QUESTION` and score it against three specific metrics: **faithfulness**, **answer_relevancy**, and **context_precision**.

3.  **Direct Scoring & JSON Output**: The LLM is explicitly instructed to provide a floating-point score between 0.0 and 1.0 for each metric and to return its response *only* as a JSON object. This removes the need for a complex evaluation framework and relies on the LLM's reasoning to generate scores.

4.  **Robust Parsing**: The script anticipates that the LLM might occasionally wrap its response in Markdown (e.g., ```json ... ```). A cleaning step using regular expressions (`re.sub`) is applied to the LLM's output string to remove this formatting before it's parsed, preventing errors.

5.  **Configurability**: The script can easily switch between **Gemini** and **OpenAI** as the evaluator LLM by setting the `LLM_PROVIDER` environment variable.

---
## Libraries Used

* **`langchain`**: The primary framework for interacting with the LLMs.
* **`langchain_openai`**: Provides the specific connector for OpenAI models (like GPT-3.5-Turbo).
* **`langchain_google_genai`**: Provides the specific connector for Google's Gemini models.
* **`os`**: Used to read environment variables for configuring the API keys and LLM provider.
* **`json`**: Used for parsing the input `logs.json` file and the JSON-formatted string response from the LLM.
* **`re`**: Used for the crucial step of cleaning the LLM's response string to ensure it's valid JSON.
* **`pandas`**: Imported in the script, though not actively used in this version's logic.

---
## Assumptions and Simplifications

* **No RAGAs Library**: This script **deliberately does not use the `ragas` library**. The primary simplification is to trade the robust, multi-step evaluation of `ragas` for a faster, single-call evaluation from a general-purpose LLM. The generated scores are therefore qualitative estimates and are not directly comparable to formal RAGAs metrics.
* **LLM as a Reliable Judge**: The approach assumes that the chosen LLM (e.g., `gemini-2.5-flash`) is capable of accurately understanding the nuances of faithfulness, relevancy, and precision and can provide consistent, well-calibrated numeric scores based on a single prompt.
* **Raw System Prompt as Context**: The script uses the *entire* `system` prompt as the context for evaluation, without any summarization or extraction. This was a specific simplification that can result in very large prompts, potentially impacting speed and cost.

## Setup
Before running ragas_integration.py, set the environment variables `OPENAI_API_KEY`, `GEMINI_API_KEY` and `LLM_PROVIDER` as `GEMINI` or `OPENAI`. The output json file given here is generated using GEMINI.
