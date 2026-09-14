
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import os
import glob
import subprocess


def colorize_image(input_image):

    if input_image is None:
        return None, None, None

    input_dir = "gradio_input"
    output_dir = "gradio_output"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Remove previous files
    for f in glob.glob(os.path.join(input_dir, "*")):
        os.remove(f)

    for f in glob.glob(os.path.join(output_dir, "*")):
        os.remove(f)

    # Convert uploaded image
    original = Image.fromarray(input_image).convert("RGB")

    # Create black-and-white version
    gray = cv2.cvtColor(
        np.array(original),
        cv2.COLOR_RGB2GRAY
    )

    black_white = Image.fromarray(gray).convert("RGB")

    # Save input for DDColor
    input_path = os.path.join(input_dir, "input.jpg")
    original.save(input_path)

    # Run DDColor
    subprocess.run(
        [
            "python",
            "scripts/infer.py",
            "--model_name",
            "ddcolor_paper",
            "--input",
            input_dir,
            "--output",
            output_dir
        ],
        check=True
    )

    # Find output image
    output_files = []

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        output_files.extend(
            glob.glob(os.path.join(output_dir, ext))
        )

    if not output_files:
        raise Exception("DDColor did not generate an output image.")

    result = Image.open(output_files[0]).convert("RGB")

    # Slight color enhancement
    result_np = np.array(result)

    hsv = cv2.cvtColor(
        result_np,
        cv2.COLOR_RGB2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * 1.15,
        0,
        255
    ).astype(np.uint8)

    enhanced = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    final_image = Image.fromarray(enhanced)

    # Save final result
    final_image.save("final_colorized_image.png")

    return original, black_white, final_image


# Create interface
with gr.Blocks(title="AI Image Colorization") as demo:

    gr.Markdown(
        """
        # 🎨 AI Image Colorization Using Generative AI

        ### Restore realistic colors to black-and-white photographs using DDColor AI.

        Upload an image and compare the original, grayscale, and AI colorized versions.
        """
    )

    input_image = gr.Image(
        type="numpy",
        label="📤 Upload Image"
    )

    colorize_button = gr.Button(
        "✨ Colorize Image",
        variant="primary"
    )

    gr.Markdown("## 🖼️ Before & After Comparison")

    with gr.Row():

        original_output = gr.Image(
            type="pil",
            label="📷 Original Image"
        )

        grayscale_output = gr.Image(
            type="pil",
            label="⚫ Black & White"
        )

        colorized_output = gr.Image(
            type="pil",
            label="🎨 AI Colorized"
        )

    colorize_button.click(
        fn=colorize_image,
        inputs=input_image,
        outputs=[
            original_output,
            grayscale_output,
            colorized_output
        ]
    )

    gr.Markdown(
        """
        ---

        ### 🔬 How It Works

        **Input Image → DDColor Deep Learning Model → Color Prediction → OpenCV Enhancement → Final Image**

        **Technologies:** Python • PyTorch • OpenCV • DDColor • Gradio
        """
    )


demo.launch(share=True)
