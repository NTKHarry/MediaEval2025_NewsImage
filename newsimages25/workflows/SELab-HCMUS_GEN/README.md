# News Image Generation Pipeline

This pipeline processes news articles through four main steps:
1. **Crawling**: Extract article content from URLs in the CSV data
2. **Finesure Processing**: Generate key facts and summaries using LLM
3. **Image Generation**: Generate stylized images based on article content
4. **LLM Judge**: Evaluate how well generated images match the articles

## Setup

### Prerequisites
- Our configuration follows Infinity, please refer to the installation: https://github.com/FoundationVision/Infinity

- Other installation:
```bash
pip install -r requirements.txt
```
### Configuration
All settings are configured in `config.py`:
- Input CSV path and output directories
- Model settings (LLM model, GPU ID, generation parameters)
- Crawler settings (timeout, retries)
- Image generation settings (model paths, styles, sources)
- Judge model settings

## Data Format
The pipeline uses v1.1 newsarticles.csv format where:
- **Article IDs are 1-based** (row 1 = article ID 1, row 2 = article ID 2, etc.)
- Each row contains: article_title, article_url, image_id, etc.
- The data is stored in data/

## Usage

### Step 1: Crawl Articles

**Process all articles:**
```bash
python src/crawler.py
```
- **Input**: `data/newsarticles.csv`
- **Output**: `output/crawled_articles_all.json`

**Process specific article:**
```bash
python src/crawler.py --id 5
```
- **Input**: `data/newsarticles.csv`
- **Output**: `output/crawled_article_5.json`

### Step 2: Generate Summaries (Finesure Processing)

**Process all crawled articles:**
```bash
python src/finesure_pipeline.py --input output/crawled_articles_all.json
```
- **Input**: `output/crawled_articles_all.json`
- **Output**: 
  - `output/finesure_result_all.json` (detailed processing results)
  - `output/crawled_with_summaries_all.json` (enhanced crawled data with filtered summaries)

**Process specific article:**
```bash
python src/finesure_pipeline.py --input output/crawled_article_5.json --id 5
```
- **Input**: `output/crawled_article_5.json`
- **Output**: 
  - `output/finesure_result_5.json`
  - `output/crawled_with_summaries_5.json`

### Step 3: Generate Images

**Generate images for all articles:**
```bash
# Using filtered summaries in cartoon style
python src/gen.py --input output/crawled_with_summaries_all.json --style cartoon --information_source summaries

# Using titles in realistic style
python src/gen.py --input output/crawled_with_summaries_all.json --style realistic --information_source title

# Using descriptions in abstract style
python src/gen.py --input output/crawled_with_summaries_all.json --style abstract --information_source description
```
- **Input**: `output/crawled_with_summaries_all.json`
- **Output**: Images in `output/images/` with format `{id}_{style}_{information_source}.png`

**Generate images for specific article:**
```bash
python src/gen.py --input output/crawled_with_summaries_5.json --style modern --information_source summaries --id 5
```
- **Input**: `output/crawled_with_summaries_5.json`
- **Output**: Single image `output/images/5_modern_summaries.png`

**Available options:**
- **Styles**: `cartoon`, `realistic`, `abstract`, `modern`
- **Information sources**: `title`, `summaries`, `description`

### Step 4: LLM Judge Evaluation (optional)

**Evaluate all articles:**
```bash
# Judge images generated from summaries
python src/judge.py --input output/crawled_with_summaries_all.json --information_type summaries

# Judge images generated from titles
python src/judge.py --input output/crawled_with_summaries_all.json --information_type title
```
- **Input**: `output/crawled_with_summaries_all.json` + images from `output/images/`
- **Output**: 
  - Results in `output/llm_judge_summaries/` or `output/llm_judge_title/`
  - For each ID: folder `{id}/` containing `candidate1.json`, `candidate2.json`, `teacher.json`
  - Copied images in the judge output directory

**Evaluate specific article:**
```bash
python src/judge.py --input output/crawled_with_summaries_5.json --information_type summaries --id 5
```
- **Input**: `output/crawled_with_summaries_5.json` + 4 style images for ID 5
- **Output**: Results in `output/llm_judge_summaries/5/` containing 3 JSON files

**Note**: For each ID and information type, exactly 4 style images must exist (cartoon, realistic, abstract, modern) for the judge to work.

## Complete Workflow Example

```bash
# Step 1: Crawl article ID 5
python src/crawler.py --id 5

# Step 2: Generate summaries for article ID 5
python src/finesure_pipeline.py --input output/crawled_article_5.json --id 5

# Step 3: Generate images using summaries in all 4 styles
python src/gen.py --input output/crawled_with_summaries_5.json --style cartoon --information_source summaries --id 5
python src/gen.py --input output/crawled_with_summaries_5.json --style realistic --information_source summaries --id 5
python src/gen.py --input output/crawled_with_summaries_5.json --style abstract --information_source summaries --id 5
python src/gen.py --input output/crawled_with_summaries_5.json --style modern --information_source summaries --id 5

# Step 4: Judge the generated images
python src/judge.py --input output/crawled_with_summaries_5.json --information_type summaries --id 5
```

## Output Structure

```
pipeline/
├── data/
│   └── newsarticles.csv                    # Input CSV data
├── output/
│   ├── crawled_articles_all.json           # Step 1 output (all)
│   ├── crawled_article_5.json              # Step 1 output (single)
│   ├── finesure_result_all.json            # Step 2 detailed results
│   ├── crawled_with_summaries_all.json     # Step 2 enhanced data
│   ├── images/                             # Step 3 output
│   │   ├── 5_cartoon_summaries.png
│   │   ├── 5_realistic_summaries.png
│   │   ├── 5_abstract_summaries.png
│   │   └── 5_modern_summaries.png
│   └── llm_judge_summaries/                # Step 4 output
│       └── 5/                             # Evaluation results
│           ├── candidate1.json
│           ├── candidate2.json
│           └── teacher.json
├── src/                                    # Source code
└── config.py                              # Configuration
```

