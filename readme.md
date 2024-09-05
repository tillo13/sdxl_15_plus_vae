# VAE and Stable Diffusion 1.5 Image Generation Loop

## Overview

This script, `vae_sd1.5_image_gen_loop.py`, generates images using Variational Autoencoder (VAE) and Stable Diffusion 1.5 (SD1.5) models in a loop. It leverages the IP-Adapter for enhanced face ID processing. The script ensures the setup of necessary dependencies, checks for sufficient disk space, and processes images with specified models to generate high-fidelity outputs.

## Requirements

- Python 3.7 or higher
- A virtual environment (optional but recommended)
- Required Python packages (diffusers, requests, torch, etc.)

## Features

- Uses VAE and SD1.5 models for image generation.
- Utilizes face ID processing with IP-Adapter.
- Ensures sufficient disk space before downloading models.
- Loops through all images in the `incoming_images` directory multiple times based on the specified number of loops.
- Configurable global variables for prompts, image dimensions, and generation parameters.

## Setup

1. **Clone Repository and Set Up Virtual Environment (Optional)**:
    ```sh
    cd vae_sd1.5_image_gen_pipeline
    python -m venv kumori_venv
    source kumori_venv/bin/activate  # On Windows use `.\kumori_venv\Scripts\activate`
    ```

2. **Install Dependencies**:
    The script will automatically install required dependencies during execution.

3. **Ensure Necessary Files**:
    Place any initial images for processing in the `shared/incoming_images` directory.

## Usage

1. **Set Global Variables**:
    Edit the following section in `vae_sd1.5_image_gen_loop.py` to configure prompts, image dimensions, and generation parameters:
    ```python
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
    NUM_INFERENCE_STEPS = 50  # Number of inference steps for image generation

    # Loop configuration
    NUM_LOOPS = 3  # Number of loops to repeat the process

    # ###########################
    ```

2. **Run the Script**:
    Execute the script to start the image generation process:
    ```sh
    python vae_sd1.5_image_gen_loop.py
    ```

## Output

- **Generated Images**:
  The generated images will be saved in the `generated_images` directory.

- **Debug Images**:
  Any debug images will be saved in the `debug_images` directory (if enabled in the code).

## Notes

- Ensure your system has enough disk space as required by the script.
- The script checks for the existence of model files before attempting to download them.

## References

- [IP-Adapter on GitHub](https://github.com/tencent-ailab/IP-Adapter)
- [Hugging Face IP-Adapter-FaceID Models](https://huggingface.co/h94/IP-Adapter-FaceID/tree/main)