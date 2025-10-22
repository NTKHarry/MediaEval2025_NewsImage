import sys
import os
import torch
import numpy as np
from PIL import Image
import time
import random
# Add Infinity directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level from src to pipeline, then into Infinity
pipeline_dir = os.path.dirname(current_dir)
infinity_dir = os.path.join(pipeline_dir, 'Infinity')
sys.path.append(infinity_dir)

# Now we can import from the Infinity module
from tools.run_infinity import *
import argparse
def load_model():   
    """
    Returns:
        tuple: (infinity, vae, text_tokenizer, text_encoder) - loaded models
    """
    #use cuda 
    
    model_path = None
    vae_path = None
    text_encoder_ckpt = None
    args = argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=14,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_8b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=1,
        h_div_w_template=1.750,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch_shard',
        seed=0,
        bf16=1,
        save_file='tmp.jpg'
    )
    
    # Load text encoder and tokenizer
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    
    # Load VAE
    vae = load_visual_tokenizer(args)
    
    # Load infinity model
    infinity = load_transformer(vae, args)
    
    return infinity, vae, text_tokenizer, text_encoder, args

def inference(infinity, vae, text_tokenizer, text_encoder, prompt, 
              cfg=3, tau=1.0, h_div_w=16/28, seed=None, args=None, enable_positive_prompt=0, negative_prompt = ''):
    """
    Generate an image from text prompt
    
    Args:
        infinity: The loaded infinity model
        vae: The loaded VAE model
        text_tokenizer: The loaded text tokenizer
        text_encoder: The loaded text encoder
        prompt (str): Text prompt for image generation
        cfg (float): Classifier-free guidance scale
        tau (float): Temperature parameter
        h_div_w (float): Aspect ratio (height/width)
        seed (int): Random seed for generation, random if None
        args: Model arguments
        enable_positive_prompt (int): Whether to enable positive prompt
        
    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    import time
    import random
    if seed is None:
        seed = random.randint(0, 10000)
    
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
        negative_prompt = negative_prompt,
    )
    
    # Convert to numpy array and fix color channel order
    generated_image = generated_image.cpu().numpy()
    generated_image = generated_image[:, :, ::-1]  # BGR to RGB
    #generated_image = cv2.resize(generated_image, (460, 260), interpolation=cv2.INTER_AREA)
    return generated_image

# Example usage:
# infinity, vae, text_tokenizer, text_encoder, args = load_model()
# prompt = "Your text prompt here"
# image = inference(infinity, vae, text_tokenizer, text_encoder, prompt, args=args)