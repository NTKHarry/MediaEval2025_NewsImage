import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import configuration
from config import (
    OUTPUT_DIR, MODEL_NAME_JUDGE, GPU_ID, JUDGE_OUTPUT_DIR, 
    JUDGE_MAX_NEW_TOKENS, JUDGE_TEMPERATURE, JUDGE_TOP_P,
    SUPPORTED_STYLES, IMAGE_OUTPUT_DIR
)


class LLMJudgePipeline:
    """LLM Judge Pipeline using Qwen2.5-VL-7B-Instruct with transformers"""
    
    def __init__(self):
        self.device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # System prompts
        self.sys_prompt_candidate = """
You are a professional LLM judge tasked with evaluating how well 4 images match an article. 

You are given an article with the following fields:
- id: the unique identifier of the article
- title: the article title
- text: the main body content of the article

You are also given 4 images. Your task is to assign an integer number of points to each image according to how well it represents the article. You start with a total of 10 points and must distribute all points among the 4 images.

Follow these strict rules:
1. Capture Key Attributes: The image must reflect the main themes, key attributes, and overall message of the article.
2. Avoid Extraneous Information: The image must not depict anything not present in the article (people, events, objects, or context). Misleading imagery is unacceptable.
3. IMPORTANT: The image styling is not important, do not care about the styling should fit, focus on the relevance only

Your output must be strictly a single dictionary only, with no extra text. Use the following format:

{
"id": "<article id>",
"reason_1": "<detailed and comprehensive reasoning for the first image>",
"reason_2": "<detailed and comprehensive reasoning for the second image>",
"reason_3": "<detailed and comprehensive reasoning for the third image>",
"reason_4": "<detailed and comprehensive reasoning for the fourth image>",
"scores": [<score for image 1>, <score for image 2>, <score for image 3>, <score for image 4>]
}

Make sure that:
- The reasoning explains exactly why each image fits or does not fit the article content.
- The reasoning should be as detailed as possible, really long and informative. Your reasoning SHOULD BE AT LEAST 100 WORDS
- The sum of the 4 scores is exactly 10 points.
- Scores are integers and reflect your judgment based on the three rules above.
- The output format must follow strictly the specified dictionary format without any additional text
- The reasoning should mention the rules above explicitly where relevant.
- If the input text is empty, just leave the field blank.

OUTPUT EXAMPLE:
{
"id": "1",
"reason_1": "Image 1 depicts a city skyline covered in heavy smog with cars emitting exhaust fumes. This image is highly relevant to the article, which discusses urban air pollution and its effects on public health. The image captures the key attributes of the article: urban environment, visible air pollution, and sources of emissions, which aligns with rule 1 (Capture Key Attributes). There are no extraneous elements like unrelated people, animals, or events that could mislead the viewer, satisfying rule 2 (Avoid Extraneous Information). Although the image is realistic and photorealistic, it does not depict a specific fabricated event, ensuring rule 3 (Prevent Deception) is respected. The composition clearly conveys the severity of pollution and its impact on the city, which makes it a strong candidate for conveying the main message of the article.",        
...
"scores": [4, 0, 2, 4]
}

READY? HERE IS THE INPUT:
"""

        self.sys_prompt_teacher = """
You are a professional teacher LLM judge tasked with evaluating the outputs of two candidate judges, Candidate 1 and Candidate 2. Both candidates have evaluated how well 4 images match a given article.

You are given:
- The article with fields:
- id: unique identifier
- title: article title
- text: main body content
- The outputs of Candidate 1 and Candidate 2, each in the following format:

{
"id": "<article id>",
"reason_1": "<candidate reasoning for image 1>",
"reason_2": "<candidate reasoning for image 2>",
"reason_3": "<candidate reasoning for image 3>",
"reason_4": "<candidate reasoning for image 4>",
"scores": [<score1>, <score2>, <score3>, <score4>]
}

Follow these strict rules:
    1. Capture Key Attributes: The image must reflect the main themes, key attributes, and overall message of the article.
    2. Avoid Extraneous Information: The image must not depict anything not present in the article (people, events, objects, or context). Misleading imagery is unacceptable.
    3. IMPORTANT: The image styling is not important, do not care about the styling should fit, focus on the relevance only

Your task is to:
1. For each image, evaluate both candidates' outputs. 
2. Assign final scores for each image, distributing a total of 10 points across all 4 images. Scores must be integers.
3. Provide a detailed reasoning while judging both candidates and giving your own detailed evaluation for each image.
4. Provide your output strictly in the following dictionary format with no extra text:
5. Provide detailed own evaluation for each image, your reasoning and own evaluation should be extra long, about 100 words.

{
"id": "<article id>",
"reason_1": "<your reasoning judging both candidates and giving your own detailed evaluation for image 1>",
"reason_2": "<your reasoning judging both candidates and giving your own detailed evaluation for image 2>",
"reason_3": "<your reasoning judging both candidates and giving your own detailed evaluation for image 3>",
"reason_4": "<your reasoning judging both candidates and giving your own detailed evaluation for image 4>",
"scores": [<score for image 1>, <score for image 2>, <score for image 3>, <score for image 4>]
}

Make sure that:
- Your reasoning explicitly judges both candidates' outputs for each image.
- Your reasoning clearly states your own evaluation for each image.
- Mention the three rules above where relevant.
- The sum of scores is exactly 10 points.
- No extra text outside the dictionary.
- If the input text is empty, just leave the feild blank.
- Do not care about the styling should fit, focus on the relevance only 

OUTPUT EXAMPLE:
{
"id": "1",
"reason_1": "Candidate 1 for Image 1 highlights that the picture shows a city skyline covered in heavy smog and links this directly to the article's central theme of urban pollution and public health. This is a strong observation because it captures the main attributes mentioned in the article (Rule 1: capture key attributes) and does not bring in any irrelevant or misleading content (Rule 2). Candidate 2 also comments on the urban environment but places more emphasis on how the image looks aesthetically, describing it as dramatic or visually appealing. This is not in line with the judging rules, as Rule 3 clearly states that styling should not matter; relevance is the only focus. From my perspective as the teacher judge, Image 1 is highly relevant because it clearly visualizes the core issue discussed in the article—the presence of pollution in an urban environment and its potential health risks. It contains no extraneous elements and effectively reinforces the message. Therefore, I assign a high score to this image.",
.....
"scores": [4, 0, 2, 4]
}

READY? HERE IS THE INPUT:
"""

    def load_model(self):
        """Load the Qwen2.5-VL model and processor"""
        print(f"Loading Qwen2.5-VL-7B-Instruct model on {self.device}...")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME_JUDGE,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME_JUDGE)
        
        print("Model loaded successfully!")

    def parse_llm_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse LLM output for judge evaluation"""
        if not output or output.strip() == "":
            print("ERROR: Empty or None output received")
            return None
            
        try:
            # Try to find JSON in the output
            start_idx = output.find('{')
            if start_idx != -1:
                end_idx = output.rfind('}') + 1
                json_str = output[start_idx:end_idx]
                # Clean up the JSON string
                json_str = json_str.replace('\n', ' ').strip()
                
                # Try ast.literal_eval first (safer)
                try:
                    parsed_output = ast.literal_eval(json_str)
                    return parsed_output
                except (ValueError, SyntaxError) as e:
                    print(f"ERROR: ast.literal_eval failed: {e}")
                    # Fall back to json.loads
                    try:
                        parsed_output = json.loads(json_str)
                        return parsed_output
                    except json.JSONDecodeError as e:
                        print(f"ERROR: json.loads also failed: {e}")
                        print(f"   Extracted JSON string: {json_str[:200]}...")
                        return None
            else:
                print("ERROR: No JSON structure found in output")
                print(f"   Output preview: {output[:300]}...")
                return None
                
        except Exception as e:
            print(f'ERROR: Unexpected parsing error: {type(e).__name__}: {e}')
            print(f'   Raw output preview: {output[:500]}...')
            return None

    def generate_response(self, messages: List[Dict], images: List[str]) -> Optional[str]:
        """Generate response using Qwen2.5-VL model"""
        try:
            # Process the conversation and images
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Load and process images
            image_inputs = []
            for img_path in images:
                image = Image.open(img_path)
                image_inputs.append(image)
            
            inputs = self.processor(
                text=[text],
                images=[image_inputs],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                temperature=JUDGE_TEMPERATURE,
                top_p=JUDGE_TOP_P,
                do_sample=True
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else None
            
        except Exception as e:
            print(f"ERROR: Generation failed: {e}")
            return None

    def get_image_paths_for_id_and_info_type(self, article_id: str, information_type: str) -> Optional[List[str]]:
        """Get the 4 style image paths for a specific ID and information type"""
        image_paths = []
        
        for style in SUPPORTED_STYLES:
            # Expected filename format: {id}_{style}_{information_type}.png
            filename = f"{article_id}_{style}_{information_type}.png"
            image_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
            
            if not os.path.exists(image_path):
                print(f"WARNING: Missing image file: {image_path}")
                return None
            
            image_paths.append(image_path)
        
        return image_paths

    def copy_images_to_judge_folder(self, article_id: str, information_type: str, image_paths: List[str]):
        """Copy images to the judge output folder"""
        judge_folder = os.path.join(JUDGE_OUTPUT_DIR, f"llm_judge_{information_type}")
        os.makedirs(judge_folder, exist_ok=True)
        
        for i, source_path in enumerate(image_paths):
            filename = os.path.basename(source_path)
            dest_path = os.path.join(judge_folder, filename)
            
            # Copy the image if it doesn't exist in destination
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                print(f"Copied image: {filename}")

    def create_candidate_messages(self, article: Dict, image_paths: List[str]) -> List[Dict]:
        """Create messages for candidate evaluation"""
        # Build user prompt
        user_prompt = f"""
Article Information:
ID: {article['article_id']}
Title: {article['title']}
Text: {article.get('main_text', '')}

You are provided with 4 images in this specific order:
1. Image 1: Cartoon style representation
2. Image 2: Realistic style representation  
3. Image 3: Abstract style representation
4. Image 4: Modern style representation

Please evaluate each image based on how well it represents the article content (ignore styling).
"""
        
        # Create messages with images
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt_candidate
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt}
                ] + [{"type": "image", "image": img_path} for img_path in image_paths]
            }
        ]
        
        return messages

    def create_teacher_messages(self, article: Dict, image_paths: List[str], candidate1_result: Dict, candidate2_result: Dict) -> List[Dict]:
        """Create messages for teacher evaluation"""
        user_prompt = f"""
Article Information:
ID: {article['article_id']}
Title: {article['title']}
Text: {article.get('main_text', '')}

Candidate 1 Output:
{json.dumps(candidate1_result, indent=2)}

Candidate 2 Output:
{json.dumps(candidate2_result, indent=2)}

Please evaluate both candidates' outputs and provide your final judgment.
"""
        
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt_teacher
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ] + [{"type": "image", "image": img_path} for img_path in image_paths]
            }
        ]
        
        return messages

    def process_single_article(self, article: Dict, information_type: str) -> Optional[Dict]:
        """Process a single article for evaluation"""
        article_id = str(article['article_id'])
        
        print(f"\nProcessing article ID {article_id} with information type '{information_type}'")
        print(f"Title: {article['title'][:100]}...")
        
        # Get image paths for this ID and information type
        image_paths = self.get_image_paths_for_id_and_info_type(article_id, information_type)
        if not image_paths:
            print(f"ERROR: Cannot find all 4 style images for ID {article_id} and type {information_type}")
            return None
        
        print(f"Found all 4 required images for styles: {SUPPORTED_STYLES}")
        
        # Copy images to judge folder
        # self.copy_images_to_judge_folder(article_id, information_type, image_paths)
        
        # Create output directory for this ID
        id_output_dir = os.path.join(JUDGE_OUTPUT_DIR, f"llm_judge_{information_type}", article_id)
        os.makedirs(id_output_dir, exist_ok=True)
        
        results = {}
        
        # Process Candidate 1
        print("Evaluating with Candidate 1...")
        candidate1_messages = self.create_candidate_messages(article, image_paths)
        candidate1_response = self.generate_response(candidate1_messages, image_paths)
        
        if candidate1_response:
            candidate1_parsed = self.parse_llm_output(candidate1_response)
            if candidate1_parsed:
                # Save candidate 1 result
                candidate1_file = os.path.join(id_output_dir, "candidate1.json")
                with open(candidate1_file, 'w', encoding='utf-8') as f:
                    json.dump(candidate1_parsed, f, ensure_ascii=False, indent=4)
                print(f"Candidate 1 result saved to {candidate1_file}")
                results['candidate1'] = candidate1_parsed
            else:
                print("ERROR: Failed to parse Candidate 1 response")
                return None
        else:
            print("ERROR: Candidate 1 failed to generate response")
            return None
        
        # Process Candidate 2
        print("Evaluating with Candidate 2...")
        candidate2_messages = self.create_candidate_messages(article, image_paths)
        candidate2_response = self.generate_response(candidate2_messages, image_paths)
        
        if candidate2_response:
            candidate2_parsed = self.parse_llm_output(candidate2_response)
            if candidate2_parsed:
                # Save candidate 2 result
                candidate2_file = os.path.join(id_output_dir, "candidate2.json")
                with open(candidate2_file, 'w', encoding='utf-8') as f:
                    json.dump(candidate2_parsed, f, ensure_ascii=False, indent=4)
                print(f"Candidate 2 result saved to {candidate2_file}")
                results['candidate2'] = candidate2_parsed
            else:
                print("ERROR: Failed to parse Candidate 2 response")
                return None
        else:
            print("ERROR: Candidate 2 failed to generate response")
            return None
        
        # Process Teacher
        print("Evaluating with Teacher...")
        teacher_messages = self.create_teacher_messages(article, image_paths, candidate1_parsed, candidate2_parsed)
        teacher_response = self.generate_response(teacher_messages, image_paths)
        
        if teacher_response:
            teacher_parsed = self.parse_llm_output(teacher_response)
            if teacher_parsed:
                # Save teacher result
                teacher_file = os.path.join(id_output_dir, "teacher.json")
                with open(teacher_file, 'w', encoding='utf-8') as f:
                    json.dump(teacher_parsed, f, ensure_ascii=False, indent=4)
                print(f"Teacher result saved to {teacher_file}")
                results['teacher'] = teacher_parsed
            else:
                print("ERROR: Failed to parse Teacher response")
                return None
        else:
            print("ERROR: Teacher failed to generate response")
            return None
        
        print(f"Successfully processed article ID {article_id}")
        return results

    def run(self, input_json_path: str, information_type: str, article_id: Optional[str] = None):
        """Run the judge pipeline"""
        # Load model
        self.load_model()
        
        # Load input data
        print(f"Loading data from {input_json_path}...")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Loaded {len(articles)} articles")
        
        # Filter for specific ID if provided
        if article_id:
            articles = [art for art in articles if str(art.get('article_id', '')) == article_id]
            if not articles:
                print(f"ERROR: Article ID {article_id} not found in the data")
                return
            print(f"Processing single article ID: {article_id}")
        else:
            print(f"Processing all articles")
        
        # Process articles
        success_count = 0
        total_count = len(articles)
        
        for i, article in enumerate(articles, 1):
            print(f"\n{'='*60}")
            print(f"Processing article {i}/{total_count}")
            print(f"{'='*60}")
            
            try:
                result = self.process_single_article(article, information_type)
                if result:
                    success_count += 1
                    print(f"Article {article.get('article_id', 'unknown')} processed successfully")
                else:
                    print(f"Article {article.get('article_id', 'unknown')} failed to process")
            except Exception as e:
                print(f"ERROR: Exception while processing article {article.get('article_id', 'unknown')}: {e}")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {success_count}/{total_count} articles")
        print(f"Output directory: {os.path.join(JUDGE_OUTPUT_DIR, f'llm_judge_{information_type}')}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="LLM Judge Pipeline for Image-Article Matching Evaluation")
    
    parser.add_argument('--input', type=str, required=True, 
                       help='Input JSON file path (e.g., output/crawled_with_summaries_all.json)')
    parser.add_argument('--information_type', type=str, required=True, 
                       choices=['title', 'summaries'], 
                       help='Information type used for image generation')
    parser.add_argument('--id', type=str, 
                       help='Specific article ID to process (omit to process all)')
    
    args = parser.parse_args()
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    # Create and run pipeline
    pipeline = LLMJudgePipeline()
    pipeline.run(args.input, args.information_type, args.id)


if __name__ == "__main__":
    main()
