import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import re
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Import configuration
from config import (
    OUTPUT_DIR, MODEL_NAME_IMAGE_GEN, GPU_ID, IMAGE_OUTPUT_DIR, PROMPTS_DIR,
    SUPPORTED_STYLES, SUPPORTED_INFO_SOURCES, INFINITY_MODEL_PATH, INFINITY_VAE_PATH,
    INFINITY_TEXT_ENCODER_PATH, IMG_GEN_MAX_NEW_TOKENS, IMG_GEN_TEMPERATURE, IMG_GEN_TOP_P
)

# Import text-to-image functionality
from text2image import load_model, inference


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt by removing special characters and formatting."""
    if not isinstance(prompt, str) or not prompt:
        return ""
    normalized = prompt.lower()
    normalized = re.sub(r'\n+', ' ', normalized)
    normalized = re.sub(r'[^a-z0-9\s.,\-]', '', normalized)
    normalized = normalized.strip().strip('-')
    return normalized


def load_prompt_template(style: str) -> str:
    """Load prompt template for the specified style."""
    prompt_file = os.path.join(PROMPTS_DIR, f"{style}.txt")
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_file}")
        return ""


def extract_content_by_source(item: dict, information_source: str) -> str:
    """Extract content based on information source."""
    if information_source == "title":
        return item.get("title", "")
    elif information_source == "summaries":
        # Handle both filtered_summaries and summaries
        summaries = item.get("filtered_summaries", item.get("summaries", []))
        if isinstance(summaries, list):
            return " ".join(summaries)
        return str(summaries) if summaries else ""
    elif information_source == "description":
        return item.get("main_text", "")
    else:
        return ""


def generate_prompt(pipe, content_text: str, style_prompt: str) -> str:
    """Generate a creative prompt using the language model."""
    # Replace placeholder in prompt template with actual content
    input_text = style_prompt.replace("<information type (can be summaries, title, or description)>", content_text)
    messages = [{"role": "user", "content": input_text}]

    try:
        # Set random seed for reproducibility
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        print(f"[Seed used: {seed}]")

        response = pipe(messages, max_new_tokens=IMG_GEN_MAX_NEW_TOKENS)[0]['generated_text']
        assistant_contents = [
            message["content"] for message in response
            if message["role"] == "assistant"
        ]
        if assistant_contents:
            caption = assistant_contents[0].strip()
            caption = normalize_prompt(caption)
            return caption
    except Exception as e:
        print(f"Error during prompt generation: {e}")

    return ""


def generate_image(infinity, vae, text_tokenizer, text_encoder, prompt: str, 
                  article_id: str, style: str, information_source: str, 
                  output_dir: str, args) -> str:
    """Generate image from prompt using text-to-image model."""
    if not prompt:
        print(f"Empty prompt for article {article_id}, skipping image generation")
        return ""
    
    print(f"Generating image for article ID={article_id}, style={style}, source={information_source}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename: <id>_<style>_<information_source>.png
    filename = f"{article_id}_{style}_{information_source}.png"
    image_path = os.path.join(output_dir, filename)
    
    try:
        print(f"  -> Generating image with prompt: {prompt[:100]}...")
        image_array = inference(infinity, vae, text_tokenizer, text_encoder, prompt, args=args)
        image = Image.fromarray(image_array.astype(np.uint8))
        image.save(image_path)
        print(f"  -> Saved image: {image_path}")
        return image_path
    except Exception as e:
        print(f"  -> Error generating image for ID {article_id}: {e}")
        return ""


def load_llm_model(model_id: str = None):
    """Load the language model for prompt generation."""
    if model_id is None:
        model_id = MODEL_NAME_IMAGE_GEN
    
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
        max_new_tokens=IMG_GEN_MAX_NEW_TOKENS,
        do_sample=True,
        top_p=IMG_GEN_TOP_P,
        top_k=50,
        temperature=IMG_GEN_TEMPERATURE,
    )
    return pipe


def main():
    """Main function for image generation."""
    parser = argparse.ArgumentParser(description="Generate images from news articles")
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--style', type=str, required=True, choices=SUPPORTED_STYLES,
                       help=f'Image style: {", ".join(SUPPORTED_STYLES)}')
    parser.add_argument('--information_source', type=str, required=True, choices=SUPPORTED_INFO_SOURCES,
                       help=f'Information source: {", ".join(SUPPORTED_INFO_SOURCES)}')
    parser.add_argument('--id', type=str, help='Specific article ID to process')
    
    args = parser.parse_args()
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    # Load input data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for specific ID if provided
    if args.id:
        filtered_data = [item for item in data if str(item.get('article_id', '')) == args.id]
        if not filtered_data:
            print(f"Error: Article ID {args.id} not found in data")
            return
        data = filtered_data
        print(f"Processing article ID: {args.id}")
    else:
        print(f"Processing all {len(data)} articles")
    
    # Load prompt template
    style_prompt = load_prompt_template(args.style)
    if not style_prompt:
        print(f"Error: Could not load prompt template for style '{args.style}'")
        return
    
    # Load models
    print("Loading language model...")
    pipe = load_llm_model()
    
    print("Loading text-to-image model...")
    try:
        infinity, vae, text_tokenizer, text_encoder, model_args = load_model()
        print("Text-to-image model loaded successfully!")
    except Exception as e:
        print(f"Error loading text-to-image model: {e}")
        return
    
    # Create output directory
    output_dir = IMAGE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each article
    results = []
    for item in tqdm(data, desc="Generating images"):
        article_id = item.get('article_id', 'unknown')
        
        # Extract content based on information source
        content_text = extract_content_by_source(item, args.information_source)
        if not content_text:
            print(f"Warning: No {args.information_source} found for article {article_id}, skipping")
            continue
        
        print(f"\nProcessing article ID={article_id}")
        print(f"Content ({args.information_source}): {content_text[:150]}...")
        
        # Generate prompt
        prompt = generate_prompt(pipe, content_text, style_prompt)
        if not prompt:
            print(f"Failed to generate prompt for article {article_id}")
            continue
        
        # Generate image
        image_path = generate_image(
            infinity, vae, text_tokenizer, text_encoder, prompt,
            article_id, args.style, args.information_source, 
            output_dir, model_args
        )
        
        if image_path:
            results.append({
                "article_id": article_id,
                "style": args.style,
                "information_source": args.information_source,
                "content_text": content_text[:200] + "..." if len(content_text) > 200 else content_text,
                "generated_prompt": prompt,
                "image_path": image_path
            })
    
    # Save results
    if results:
        if args.id:
            results_file = os.path.join(OUTPUT_DIR, f'image_generation_results_{args.id}_{args.style}_{args.information_source}.json')
        else:
            results_file = os.path.join(OUTPUT_DIR, f'image_generation_results_{args.style}_{args.information_source}.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {results_file}")
    
    print(f"\nGeneration complete! Generated {len(results)} images in {output_dir}")


if __name__ == "__main__":
    main()
