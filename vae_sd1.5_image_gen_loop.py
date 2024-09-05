import os
import sys
import requests
import random
import cv2
import torch
import datetime
import time
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import numpy as np

# ###########################
# ####  GLOBAL VARIABLES  ####
# ###########################

# Prompt configuration
PROMPT = "human"
NEGATIVE_PROMPT = "(deformed iris, deformed pupils, semi-realistic), text, (worst quality:2), (low quality:2), (normal quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, flash, text"

# Image generation parameters
NUM_SAMPLES = 2  # Number of samples to generate per image
IMAGE_WIDTH = 1024  # Width of output images
IMAGE_HEIGHT = 1024  # Height of output images
NUM_INFERENCE_STEPS = 20  # Number of inference steps for image generation

# Loop configuration
NUM_LOOPS = 3  # Number of loops to repeat the process

# ###########################

# Define the shared directory absolute path
shared_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared")

# Add the shared directory to the system path to import shared modules
sys.path.append(shared_dir)

# Import the ensure_setup function from get_ipadapter.py
from get_ipadapter import ensure_setup

# Import check_free_space and free_up_space functions from check_hub_size.py
from check_hub_size import check_free_space, free_up_space, CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB

# Ensure the IP-Adapter setup is complete
ensure_setup()

# Add the IP-Adapter directory to sys.path
sys.path.append(os.path.join(shared_dir, "IP-Adapter"))

# Now import the IP-Adapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

# Check and ensure sufficient free space before downloading models
free_space_gb = check_free_space(CACHE_DIRECTORY)
print(f"Currently available free space: {free_space_gb:.2f} GB")
if free_space_gb < REQUIRED_FREE_SPACE_GB:
    print("Not enough free space, initiating cleanup...")
    free_up_space(CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB)
else:
    print("Enough free space is available, no cleanup needed.")

# Configuration constants

# List of Hugging Face model paths
model_paths = [
    "digiplay/Photon_v1",
    "darkstorm2150/Protogen_x5.8_Official_Release",
    "DucHaiten/DucHaiten-StyleLikeMe",
    "Lykon/dreamshaper-7",
    "SG161222/Paragon_V1.0",
]

# VAE model path
vae_model_name = "stabilityai/sd-vae-ft-mse"  # VAE model
custom_cache_dir = "d:/cache2"  # Custom cache directory

# Define paths and ensure directories exist
input_dir = os.path.join(shared_dir, "incoming_images")
output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
debug_output_dir = os.path.join(os.path.dirname(__file__), "debug_images")
ignore_dir_name = "ignore_these2"
ip_ckpt_path = os.path.join(shared_dir, "IP-Adapter", "models", "ip_adapter_faceid_sd15.bin")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_output_dir, exist_ok=True)

# URLs for the required files
ip_ckpt_url = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true"

# Helper function to download a file
def download_file(url, filename):
    if not os.path.isfile(filename):
        print(f"****** Downloading {filename}... ******")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"****** Downloaded {filename} ******")
    else:
        print(f"****** {filename} already exists ******")

# Download the IP-Adapter checkpoint if it doesn't exist
download_file(ip_ckpt_url, ip_ckpt_path)

# Load face analysis model
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Define parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

# Function to load model, checking standard cache, custom cache, then downloading if needed
def load_model(model_path, model_class, **kwargs):
    # Standard Hugging Face Transformers cache directory
    standard_cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
    
    # Check standard cache first
    try:
        model = model_class.from_pretrained(model_path, **kwargs)
        print(f"****** Loaded {model_path} from standard cache ******")
        return model
    except Exception as e:
        pass
    
    # Try to load from custom cache directory
    custom_cache_model_path = os.path.join(custom_cache_dir, model_path.replace("/", "_"))
    if os.path.exists(custom_cache_model_path):
        try:
            model = model_class.from_pretrained(custom_cache_model_path, **kwargs)
            print(f"****** Loaded {model_path} from custom cache directory ******")
            return model
        except Exception as e:
            pass

    # If still not found, download from Hugging Face and save in the custom cache directory
    print(f"****** {model_path} not found in cache, downloading from Hugging Face... ******")
    model = model_class.from_pretrained(model_path, cache_dir=custom_cache_dir, **kwargs)
    print(f"****** Downloaded {model_path} and cached to {custom_cache_dir} ******")
    return model

# Function to generate images with a given model and VAE
def generate_images_with_vae(base_model_path, vae, input_image_path):
    try:
        pipe = load_model(base_model_path, StableDiffusionPipeline, torch_dtype=torch.float16, scheduler=noise_scheduler, vae=vae, feature_extractor=None, safety_checker=None).to(device)

        # Load IP-Adapter
        ip_model = IPAdapterFaceID(pipe, ip_ckpt_path, device)

        print(f"****** Processing file: {input_image_path} with model: {base_model_path} ******")

        # Load and process input image
        image = cv2.imread(input_image_path)
        faces = app.get(image)
        if len(faces) == 0:
            print(f"****** No faces detected in {input_image_path} ******")
            return []
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

        # Generate image with higher fidelity
        images = ip_model.generate(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            faceid_embeds=faceid_embeds,
            num_samples=NUM_SAMPLES,  # Use global variable for number of samples
            width=IMAGE_WIDTH,  # Use global variable for image width
            height=IMAGE_HEIGHT,  # Use global variable for image height
            num_inference_steps=NUM_INFERENCE_STEPS  # Use global variable for inference steps
        )

        # Save generated images
        base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
        saved_filepaths = []
        for idx, img in enumerate(images):
            timestamp = datetime.datetime.now().strftime("%H%M%S%f")
            output_filename = f"{base_filename}_{timestamp}_{os.path.basename(base_model_path)}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            img.save(output_filepath)
            saved_filepaths.append(output_filepath)

        print(f"****** Images generated and saved: {saved_filepaths} ******")
        return saved_filepaths
    except Exception as e:
        print(f"****** Skipping {input_image_path} due to error: {e} ******")
        return []

# Construct model-image pairs
model_image_pairs = []
for model_path in model_paths:
    for root, dirs, files in os.walk(input_dir):
        # Skip any directory named ignore_these2
        dirs[:] = [d for d in dirs if d != ignore_dir_name]
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_image_path = os.path.join(root, file)
                model_image_pairs.append((model_path, input_image_path))

# Multiply the list of model-image pairs by the number of loops
model_image_pairs *= NUM_LOOPS

# Shuffle the list of model-image pairs
random.shuffle(model_image_pairs)

# Calculate the total number of output images
total_output_images = len(model_image_pairs)

# Print the total number of images to be generated
print(f"****** Total number of images to be generated: {total_output_images} ******")

# Track the number of processed images
processed_images_count = 0

# Process each model-image pair
vae = load_model(vae_model_name, AutoencoderKL).to(dtype=torch.float16).to(device)
for model_path, input_image_path in model_image_pairs:
    
    # Start the timer
    start_time = time.time()
    
    # Generate images with the current model and image
    generate_images_with_vae(model_path, vae, input_image_path)

    # Stop the timer
    elapsed_time = time.time() - start_time
    processed_images_count += 1

    # Calculate remaining time
    remaining_images_count = total_output_images - processed_images_count
    estimated_remaining_time = elapsed_time * remaining_images_count

    # Convert estimated remaining time to hours, minutes, and seconds
    minutes, seconds = divmod(estimated_remaining_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Print the estimated remaining time
    print(f"****** Processed {processed_images_count}/{total_output_images} images ******")
    print(f"****** Estimated remaining time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds ******")

# Summary
print("\n===== SUMMARY =====")
print("Images generated and saved with all models for each image in the incoming_images directory.")