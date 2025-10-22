# Configuration file for the news image generation pipeline
# You can modify these default values to suit your setup

# Input data
INPUT_CSV = "data/newsarticles.csv"

# Output directory
OUTPUT_DIR = "output"

# Model configuration
MODEL_NAME_FINESURE = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME_IMAGE_GEN = "meta-llama/Meta-Llama-3-8B-Instruct"
GPU_ID = "0"

# Processing configuration
DEFAULT_MODE = "all"  # "all" or "specific"
DEFAULT_STEP = "both"  # "crawler", "finesure", or "both"

# Crawler settings
CRAWLER_TIMEOUT = 50  # seconds
MAX_RETRIES = 3

# LLM settings
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50

# Image generation settings
IMAGE_OUTPUT_DIR = "output/images"
PROMPTS_DIR = "prompts"
SUPPORTED_STYLES = ["cartoon", "realistic", "abstract", "modern"]
SUPPORTED_INFO_SOURCES = ["title", "summaries", "description"]

# Text-to-image model paths (adjust these paths as needed)
INFINITY_MODEL_PATH = '../Infinity/weights/infinity_8b_weights'
INFINITY_VAE_PATH = '../Infinity/weights/infinity_vae_d56_f8_14_patchify.pth'
INFINITY_TEXT_ENCODER_PATH = '../Infinity/weights/flan_t5_xl'

# Image generation parameters
IMG_GEN_MAX_NEW_TOKENS = 1024
IMG_GEN_TEMPERATURE = 0.7
IMG_GEN_TOP_P = 0.95

# LLM Judge settings
MODEL_NAME_JUDGE = "Qwen/Qwen2.5-VL-7B-Instruct"
JUDGE_OUTPUT_DIR = "output"  # Base directory for judge outputs
JUDGE_MAX_NEW_TOKENS = 1024
JUDGE_TEMPERATURE = 0.1  # Lower temperature for more consistent judgments
JUDGE_TOP_P = 0.9
