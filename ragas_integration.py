import json
import os
import pandas as pd
import langchain
import re

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def parse_log_data(log_file_path: str):
    """
    Loads and parses the log file into a list of dictionaries, using the raw system prompt as context.
    """
    all_items = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading log file: {e}")
        return None

    for log_entry in logs:
        for item in log_entry.get("items", []):
            item_id = item.get("id")
            inputs = item.get("input", [])
            expected_output = item.get("expectedOutput", [])
            question, context, answer = None, None, None

            for i in inputs:
                if i.get("role") == "user":
                    question = str(i.get("context", ""))
                elif i.get("role") == "system":
                    context = str(i.get("context", ""))
            
            if expected_output and isinstance(expected_output, list) and len(expected_output) > 0:
                answer = expected_output[0].get("content")

            if all((item_id, question, context, answer)):
                all_items.append({
                    "id": item_id,
                    "question": question,
                    "context": context,
                    "answer": answer
                })
            else:
                print(f"Skipping item '{item_id or 'Unknown ID'}' due to missing data.")

    if not all_items:
        print("No valid items found to evaluate.")
        return None
        
    return all_items


def main():
    """
    Main function to orchestrate the loading, direct LLM scoring, and saving process.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "GEMINI").upper()
    llm = None

    if llm_provider == "OPENAI":
        print("Using OpenAI as the LLM provider.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER is OpenAI, but OPENAI_API_KEY is not set.")
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)
    
    elif llm_provider == "GEMINI":
        print("Using Gemini as the LLM provider.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("LLM_PROVIDER is Gemini, but GOOGLE_API_KEY is not set.")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}. Please use 'OPENAI' or 'GEMINI'.")

    log_file = 'logs.json'
    output_file = 'direct_score_output.json'

    parsed_data = parse_log_data(log_file)
    if not parsed_data:
        return

    final_output = []
    print(f"Starting direct scoring with {llm.__class__.__name__}...")

    for item in parsed_data:
        prompt = f"""
        You are an expert evaluator. Your task is to evaluate a generated ANSWER based on a CONTEXT and a QUESTION.
        Provide a score from 0.0 to 1.0 for each of the following metrics (Note that it has to be a float from 0 to 1 not 0 or 1):
        1. faithfulness: Does the answer stay true to the facts in the context?
        2. answer_relevancy: Is the answer relevant to the question?
        3. context_precision: Is all the information in the context useful for answering the question?

        Output your response ONLY in JSON format like this:
        {{"faithfulness": score, "answer_relevancy": score, "context_precision": score}}

        CONTEXT:
        {item['context']}

        QUESTION:
        {item['question']}

        ANSWER:
        {item['answer']}
        """
        
        scores = {"faithfulness": None, "answer_relevancy": None, "context_precision": None}
        try:
            response = llm.invoke(prompt).content
            response = re.sub(r"^```json\s*|\s*```$", "", response.strip(), flags=re.DOTALL)
            print(response)
            scores = json.loads(response)
        except Exception as e:
            print(f"Error processing item {item['id']}: {str(e)}")
        
        print(f"Scored item: {item['id']}")
        final_output.append({
            "id": item['id'],
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision")
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"\nSuccessfully completed direct scoring. Results saved to '{output_file}'.")


if __name__ == "__main__":
    main()