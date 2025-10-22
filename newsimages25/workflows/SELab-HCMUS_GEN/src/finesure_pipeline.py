import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import argparse
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

# Import configuration
from config import OUTPUT_DIR, MODEL_NAME_FINESURE, GPU_ID, MAX_NEW_TOKENS, TEMPERATURE, TOP_P

# ==============================================================================
# 1. CONSTANTS AND PROMPTS
# ==============================================================================

ERROR_TYPES = [
    'out-of-context error', 'entity error', 'predicate error', 
    'circumstantial error', 'grammatical error', 'coreference error', 
    'linking error', 'other error'
]

KEY_FACT_PROMPT = """
Your task is to read a given text and decompose it into a list of key facts. A "key fact" is a single, atomic piece of core information, written as a concise and clear sentence. Do not include minor details or your own interpretations. Provide the output as a JSON object with a single key "key_facts" containing a list of strings.

Here are some examples:

---
**Example 1:**
**Text:** "Yesterday, Silicon Valley-based tech giant AlphaTech announced record-breaking Q3 profits of $5 billion, far exceeding analyst expectations. CEO Jane Doe attributed the success primarily to strong growth in its cloud computing division and the successful launch of its 'Nova X' smartphone in August."

**JSON Output:**
{
  "key_facts": [
    "AlphaTech announced record-breaking Q3 profits.",
    "AlphaTech's profit was $5 billion.",
    "The profit exceeded analyst expectations.",
    "The success was attributed to growth in the cloud computing division.",
    "The success was also attributed to the launch of the 'Nova X' smartphone.",
    "The 'Nova X' smartphone was launched in August."
  ]
}
---
**Example 2:**
**Text:** "A massive fire broke out at a chemical plant in Houston, Texas, on Tuesday morning, sending thick plumes of black smoke into the sky. Local authorities issued a shelter-in-place order for residents within a 1-mile radius and closed several major highways. No injuries were reported, but the cause of the fire is still under investigation."

**JSON Output:**
{
  "key_facts": [
    "A large fire occurred at a chemical plant in Houston, Texas.",
    "The fire broke out on Tuesday morning.",
    "Local authorities issued a shelter-in-place order.",
    "The order applies to residents within a 1-mile radius.",
    "Several major highways were closed.",
    "No injuries were reported.",
    "The cause of the fire is under investigation."
  ]
}
---
Ready? Please read the following text and extract the key facts in the same format as above.
"""

SUMMARY_PROMPT = """
Your task is to generate a concise, neutral, and factually accurate summary of the following news article. You are also given some key facts from the article:

**Key Facts:**
<key_facts>

Follow these instructions carefully:
1.  **Read the entire article** to understand the main events and key information.
2.  **Summarize the most important points.** Focus on who, what, when, where, and why. The summary should be long and detailed.
3.  **Do NOT add any information** that is not explicitly mentioned in the text.
4.  **Do NOT include your own opinions** or interpretations.
5.  **Write in an illustrative and explanable way, always define the subject after the name (i.e: a basketball player/team, a famous singer...).**
6.  Provide the output as a single JSON object with the key "summary".
7   **Create 7 different summaries**
Here is an example:

---
**Example 1:**
**Title:** AlphaTech Announces Record-Breaking Profits
**Text:** "Yesterday, Silicon Valley-based tech giant AlphaTech announced record-breaking Q3 profits of $5 billion, far exceeding analyst expectations. CEO Jane Doe attributed the success primarily to strong growth in its cloud computing division and the successful launch of its 'Nova X' smartphone in August."
**Key Facts:**
- AlphaTech announced record-breaking Q3 profits.
- The profit was $5 billion.
- The success was attributed to the cloud division and the new 'Nova X' phone.

**JSON Output:**
{
  "summary_1": "AlphaTech reported a record $5 billion Q3 profit, driven by its cloud division and the new 'Nova X' smartphone."
  "summary_2": "AlphaTech's Q3 profit of $5 billion exceeded expectations, attributed to cloud growth and the 'Nova X' launch."  
  ...
    "summary_7": "AlphaTech's record Q3 profit of $5 billion was driven by cloud growth and the 'Nova X' smartphone launch."
}
---
Ready? Please read the following article and generate a summary.
"""

# ==============================================================================
# 2. MODEL AND DATA HANDLING FUNCTIONS
# ==============================================================================

def initialize_model(model_id: str):
    """Initializes and returns the language model pipeline."""
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        top_k=50,
        temperature=TEMPERATURE,
    )
    return pipe


def load_data(filepath: str) -> list:
    """Loads data from a JSON file and uses article_id as id."""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use article_id as id for consistency with existing pipeline
    if data and 'article_id' in data[0]:
        print("Using article_id as id field...")
        for item in data:
            item['id'] = item['article_id']
    return data


def save_data(data: list, output_dir: str, specific_ids=None):
    """Saves the processed data to a JSON file."""
    if specific_ids:
        output_filepath = os.path.join(output_dir, f"finesure_{'_'.join(map(str, specific_ids))}.json")
    else:
        output_filepath = os.path.join(output_dir, "finesure_all.json")
        
    print(f"Saving processed data to {output_filepath}...")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# ==============================================================================
# 3. PROMPT AND PARSING FUNCTIONS
# ==============================================================================

def get_fact_checking_prompt(input_doc: str, sentences: list) -> str:
    """Creates the prompt for the fact-checking task."""
    num_sentences = str(len(sentences))
    sentences_str = '\n'.join(sentences)
    return """
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}, {"sentence": "third sentence", "reason": "your reason", "category": "entity error"},]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input_doc, num_sentences, sentences_str)


def parse_llm_fact_checking_output(output: str) -> tuple[list, list]:
    """Parses the LLM output for the fact-checking task."""
    try:
        start_idx = output.find('[')
        if start_idx != -1:
            end_idx = output.rfind(']')
            json_str = output[start_idx:end_idx+1].replace('\n', '')
            parsed_output = ast.literal_eval(json_str)
            
            pred_labels, pred_types = [], []
            for out in parsed_output:
                category = out.get("category", "").lower().strip()
                pred_labels.append(0 if category == "no error" else 1)
                pred_types.append(category or "parsing error")
            return pred_labels, pred_types
        
        else:
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            json_str = output[start_idx:end_idx+1].replace('\n', '')
            parsed_output = ast.literal_eval(json_str)

            category = parsed_output.get("category", "").lower().strip()
            pred_labels = [0 if category == "no error" else 1]
            pred_types = [category or "parsing error"]
            return pred_labels, pred_types
            
    except Exception:
        try:
            pred_labels, pred_types = [], []
            for subseq in output.split("category"):
                detected_type = "no error"
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        detected_type = error_type
                        break
                pred_labels.append(0 if detected_type == "no error" else 1)
                pred_types.append(detected_type)
            return pred_labels[1:], pred_types[1:] if pred_labels else ([], [])
        
        except Exception as e:
            print(f'Critical parsing error: {e}')
            return [], []


def parse_llm_output_for_keyfacts(llm_output: str) -> list:
    """Parses the raw LLM output to extract a list of key facts."""
    try:
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}')
        if start_index == -1 or end_index == -1:
            print("Error: Could not find JSON object brackets in LLM output.")
            return []
        json_str = llm_output[start_index : end_index + 1]
        data = json.loads(json_str)
        key_facts_list = data.get("key_facts", [])
        
        if not isinstance(key_facts_list, list):
            print(f"Error: The value for 'key_facts' is not a list, but {type(key_facts_list)}.")
            return []
            
        return key_facts_list if isinstance(key_facts_list, list) else []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for key facts: {e}")
        print(f"Problematic JSON string: {llm_output[:200]}...")
        return []
    except Exception as e:
        print(f"Unexpected error in parse_llm_output_for_keyfacts: {e}")
        return []


def parse_llm_multiple_summaries(llm_output: str) -> list:
    """Parses raw LLM output containing a JSON object with multiple summary keys."""
    try:
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}')
        if start_index == -1 or end_index == -1:
            print("Error: Could not find JSON object brackets in summary output.")
            return []
        json_str = llm_output[start_index : end_index + 1]
        data = json.loads(json_str)
        if not isinstance(data, dict):
            print("Error: Parsed JSON is not a dictionary.")
            return []
        
        summaries = [
            text for key, text in sorted(data.items()) 
            if key.startswith("summary_") and isinstance(text, str)
        ]
        
        if not summaries:
            print("Warning: No keys starting with 'summary_' found in the JSON object.")
            
        return summaries
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for summaries: {e}")
        print(f"Problematic JSON string: {llm_output[:200]}...")
        return []
    except Exception as e:
        print(f"Unexpected error in parse_llm_multiple_summaries: {e}")
        return []


def get_keyfact_alignment_prompt(keyfacts: list, sentences: list) -> str:
    """Creates the prompt for the keyfact alignment task."""
    summary = ['[' + str(line_num + 1) + '] ' + sentence for line_num, sentence in enumerate(sentences)]
    summary = '\n'.join(summary)
    num_key_facts = str(len(keyfacts))
    key_facts_str = '\n'.join(keyfacts)
    
    return """
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact. 

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}, {"key fact": "third key fact", "response": "Yes", "line number": [1, 2, 3]}]

Summary:
%s

%s key facts:
%s
""" % (summary, num_key_facts, key_facts_str)


def parse_llm_keyfact_alignment_output(output: str) -> tuple[list, list]:
    """Parses the LLM output for the keyfact alignment task."""
    try:
        output = output.replace('```', '')
        start_idx = output.find('[')
        # Check if a JSON list is found
        if start_idx == -1:
            return [], []
        
        end_idx = output.rfind(']')
        if end_idx == -1:
             return [], []

        json_str = output[start_idx:end_idx+1]
        parsed_output = ast.literal_eval(json_str)

        matched_lines = set()
        pred_labels = []

        for out in parsed_output:
            response = out.get("response", "no").lower()
            pred_labels.append(1 if response == "yes" else 0)
            
            if response == "yes" and 'line number' in out:
                line_nums = out["line number"]
                if isinstance(line_nums, list):
                    for line_num in line_nums:
                        try:
                            # Handle if line_num is string like '[1]'
                            if isinstance(line_num, str):
                                line_num = line_num.replace('[', '').replace(']', '')
                            matched_lines.add(int(line_num))
                        except (ValueError, TypeError):
                            continue # Ignore invalid line numbers
        
        return pred_labels, list(matched_lines)
    
    except Exception as e:
        print(f"Error parsing keyfact alignment output: {e}")
        return [], []


def compute_completeness_percentage_score(pred_alignment_labels: list) -> float:
    """Calculates the completeness score."""
    if not pred_alignment_labels:
        return 0.0
    return sum(pred_alignment_labels) / len(pred_alignment_labels)


def compute_conciseness_percentage_score(pred_sentence_line_numbers: list, num_sentences: int) -> float:
    """Calculates the conciseness score."""
    if num_sentences == 0:
        return 0.0
    # Ensure line numbers are unique before calculating
    return len(set(pred_sentence_line_numbers)) / num_sentences


def compute_faithfulness_percentage_score(pred_faithfulness_labels: list) -> float:
    """Calculates the faithfulness score."""
    if not pred_faithfulness_labels:
        return 0.0
    # Faithfulness is the ratio of non-erroneous sentences (label 0).
    # sum(labels) counts the number of errors (label 1).
    return 1.0 - (sum(pred_faithfulness_labels) / len(pred_faithfulness_labels))


# ==============================================================================
# 4. CORE GENERATION LOGIC
# ==============================================================================

def _call_llm(pipe, messages: list) -> str:
    """A helper function to call the LLM and extract the response content."""
    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    print(f"[Seed used: {seed}]")
    
    try:
        response = pipe(messages, max_new_tokens=1024)[0]['generated_text']
        assistant_contents = [msg["content"] for msg in response if msg["role"] == "assistant"]
        
        if assistant_contents:
            print("LLM call successful")
            return assistant_contents[0].strip()
        
        print("LLM call failed to produce assistant content")
        return ""
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return ""


def generate_key_facts(pipe, article: dict) -> list:
    """Generates and parses key facts for a given article."""
    title = article.get("title", "")
    main_text = article.get("main_text", "")
    input_text = f"{KEY_FACT_PROMPT}\n\n**Title:** \n{title}\n**Content**:\n{main_text}"
    messages = [{"role": "user", "content": input_text}]
    
    try:
        raw_output = _call_llm(pipe, messages)
        if not raw_output:
            print("No output from LLM for key facts generation")
            return []
        
        key_facts = parse_llm_output_for_keyfacts(raw_output)
        if not key_facts:
            print("Failed to parse key facts from LLM output")
            print(f"Raw output (first 300 chars): {raw_output[:300]}...")
        return key_facts
    except Exception as e:
        print(f"Error in generate_key_facts: {e}")
        return []


def generate_summaries(pipe, article: dict, key_facts: list) -> list:
    """Generates and parses multiple summaries for a given article and its key facts."""
    title = article.get("title", "")
    main_text = article.get("main_text", "")
    key_facts_str = "\n".join([f"- {fact}" for fact in key_facts])
    
    prompt_with_facts = SUMMARY_PROMPT.replace("<key_facts>", key_facts_str)
    input_text = f"{prompt_with_facts}\n\n**Title:** \n{title}\nContent:\n{main_text}"
    messages = [{"role": "user", "content": input_text}]

    try:
        raw_output = _call_llm(pipe, messages)
        if not raw_output:
            print("No output from LLM for summary generation")
            return []
        
        summaries = parse_llm_multiple_summaries(raw_output)
        if not summaries:
            print("Failed to parse summaries from LLM output")
            print(f"Raw output (first 300 chars): {raw_output[:300]}...")
        return summaries
    except Exception as e:
        print(f"Error in generate_summaries: {e}")
        return []


def perform_fact_checking(pipe, article: dict, summaries: list) -> tuple[list, list]:
    """Performs fact-checking on generated summaries against the original article."""
    main_text = article.get("main_text", "")
    input_text = get_fact_checking_prompt(main_text, summaries)
    messages = [{"role": "user", "content": input_text}]

    try:
        raw_output = _call_llm(pipe, messages)
        if not raw_output:
            print("No output from LLM for fact checking")
            return [], []
        
        pred_labels, pred_types = parse_llm_fact_checking_output(raw_output)
        if not pred_labels and not pred_types:
            print("Failed to parse fact checking output from LLM")
            print(f"Raw output (first 300 chars): {raw_output[:300]}...")
        return pred_labels, pred_types
    except Exception as e:
        print(f"Error in perform_fact_checking: {e}")
        return [], []


def perform_keyfact_alignment(pipe, key_facts: list, summaries: list) -> tuple[list, list]:
    """Performs keyfact alignment on generated summaries against the key facts."""
    # Thêm kiểm tra đầu vào để tránh gọi LLM vô ích
    if not key_facts or not summaries:
        print("Skipping keyfact alignment due to empty key facts or summaries.")
        return [], []
        
    input_text = get_keyfact_alignment_prompt(key_facts, summaries)
    messages = [{"role": "user", "content": input_text}]

    try:
        raw_output = _call_llm(pipe, messages)
        if not raw_output:
            print("No output from LLM for keyfact alignment")
            return [], []
        
        alignment_labels, matched_lines = parse_llm_keyfact_alignment_output(raw_output)
        if not alignment_labels and not matched_lines:
            print("Failed to parse keyfact alignment output from LLM")
            print(f"Raw output (first 300 chars): {raw_output[:300]}...")
        return alignment_labels, matched_lines
    except Exception as e:
        print(f"Error in perform_keyfact_alignment: {e}")
        return [], []


# ==============================================================================
# 5. MAIN WORKFLOW
# ==============================================================================

def process_items(pipe, data: list, specific_ids=None):
    """
    Iterates through specified items, generates content, and updates the data list.
    """
    if specific_ids:
        items_to_process = [item for item in data if str(item.get('id', -1)) in specific_ids]
        print(f"Processing specific IDs: {specific_ids}")
    else:
        items_to_process = data
        print("Processing all items")
    
    if not items_to_process:
        print(f"No items found to process.")
        return data

    print(f"Starting to process {len(items_to_process)} items.")
    
    fail_count = 0
    for item in tqdm(items_to_process, desc="Processing Articles"):
        print(f"\n--- Processing Item ID: {item['id']} ---")

        # 1. Generate Key Facts
        key_facts = generate_key_facts(pipe, item)
        if key_facts:
            item['key_facts'] = key_facts
        else:
            fail_count += 1
            print(f"Failed to generate key facts for item {item['id']}.")
            continue

        # 2. Generate Summaries
        summaries = generate_summaries(pipe, item, key_facts)
        if summaries:
            item['summaries'] = summaries
        else:
            fail_count += 1
            print(f"Failed to generate summaries for item {item['id']}.")
            continue

        # 3. Perform Fact Checking
        pred_labels, pred_types = perform_fact_checking(pipe, item, summaries)
        if pred_labels and pred_types:
            faithfulness_score = compute_faithfulness_percentage_score(pred_labels)
            item['fact_checking'] = {
                "pred_labels": pred_labels,
                "pred_types": pred_types,
                "faithfulness_score": faithfulness_score
            }
        else:
            fail_count += 1
            print(f"Failed to perform fact checking for item {item['id']}.")

        alignment_labels, matched_lines = perform_keyfact_alignment(pipe, item['key_facts'], item['summaries'])
        
        if alignment_labels:
            completeness_score = compute_completeness_percentage_score(alignment_labels)
            conciseness_score = compute_conciseness_percentage_score(matched_lines, len(item['summaries']))
            
            item['keyfact_alignment'] = {
                "alignment_labels": alignment_labels,
                "matched_summary_lines": matched_lines,
                "completeness_score": completeness_score,
                "conciseness_score": conciseness_score
            }
            print(f"Keyfact Alignment for item {item['id']}: Completeness={completeness_score:.2%}, Conciseness={conciseness_score:.2%}")
        else:
            # This is not a critical failure, so we don't 'continue' but log it.
            print(f"Could not perform keyfact alignment for item {item['id']}.")
            item['keyfact_alignment'] = {}

    print(f"\nProcessing finished. Total failures: {fail_count}")
    return data


def filter_matched_summaries(item):
    """Filter summaries to only keep those that appear in matched summary lines."""
    if 'keyfact_alignment' not in item or 'summaries' not in item:
        return []
    
    matched_lines = item['keyfact_alignment'].get('matched_summary_lines', [])
    summaries = item['summaries']
    
    # Convert 1-based indices to 0-based and filter summaries
    filtered_summaries = []
    for line_num in matched_lines:
        if 1 <= line_num <= len(summaries):  # line_num is 1-based
            filtered_summaries.append(summaries[line_num - 1])  # Convert to 0-based index
    
    return filtered_summaries


def update_crawled_file_with_summaries(crawled_file_path, processed_data, specific_id=None):
    """Update the original crawled file with filtered summaries."""
    # Load original crawled data
    with open(crawled_file_path, 'r', encoding='utf-8') as f:
        crawled_data = json.load(f)
    
    # Create a mapping of article_id to filtered summaries
    summaries_map = {}
    for item in processed_data:
        if 'id' in item:
            filtered_summaries = filter_matched_summaries(item)
            summaries_map[str(item['id'])] = filtered_summaries
    
    # Update crawled data with filtered summaries
    for crawled_item in crawled_data:
        article_id = str(crawled_item.get('article_id', ''))
        if article_id in summaries_map:
            crawled_item['filtered_summaries'] = summaries_map[article_id]
    
    # Save updated crawled file
    if specific_id:
        output_file = os.path.join(OUTPUT_DIR, f'crawled_with_summaries_{specific_id}.json')
    else:
        output_file = os.path.join(OUTPUT_DIR, 'crawled_with_summaries_all.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=4)
    
    print(f"Updated crawled file saved to {output_file}")
    return output_file


def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Process news articles to generate key facts and summaries")
    parser.add_argument('--input', type=str, required=True, help='Input JSON file from crawler output')
    parser.add_argument('--id', type=str, help='Specific article ID to process')
    
    args = parser.parse_args()

    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Execute workflow
    model_pipeline = initialize_model(MODEL_NAME_FINESURE)
    all_data = load_data(args.input)
    
    # Process specific ID or all data
    specific_ids = [args.id] if args.id else None
    updated_data = process_items(model_pipeline, all_data, specific_ids)
    
    # Save finesure results
    if args.id:
        output_file = os.path.join(OUTPUT_DIR, f'finesure_result_{args.id}.json')
    else:
        output_file = os.path.join(OUTPUT_DIR, 'finesure_result_all.json')
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

    print(f"Finesure results saved to {output_file}")
    
    # Update original crawled file with filtered summaries
    updated_crawled_file = update_crawled_file_with_summaries(args.input, updated_data, args.id)
    print(f"Updated crawled file with filtered summaries: {updated_crawled_file}")


if __name__ == "__main__":
    main()
