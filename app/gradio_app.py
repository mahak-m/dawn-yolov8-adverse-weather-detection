import os
from pathlib import Path
from functools import lru_cache
from collections import Counter

import gradio as gr
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# MODEL CHECKPOINTS
# --------------------------------------------------

MODELS = {
    "Baseline YOLOv8": "/content/drive/MyDrive/csc490_runs/baseline_s108/weights/best.pt",
    "Inverse Square-Root Weighted YOLOv8": "/content/drive/MyDrive/csc490_runs/weighted_sqrt_s108/weights/best.pt",
    "Inverse-Frequency Weighted YOLOv8": "/content/drive/MyDrive/csc490_runs/weighted_inverse_s108/weights/best.pt",
}


# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------

@lru_cache(maxsize=3)
def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    return YOLO(model_path)


# --------------------------------------------------
# RUN DETECTION
# --------------------------------------------------

def run_detection(image, selected_model, confidence, iou):

    if image is None:
        return [], [], "Please upload an image first."

    selected_models = (
        list(MODELS.items())
        if selected_model == "Compare All Models"
        else [(selected_model, MODELS[selected_model])]
    )

    gallery_images = []
    detection_rows = []
    summaries = []

    try:
        for model_name, model_path in selected_models:

            model = load_model(model_path)

            results = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                verbose=False
            )

            result = results[0]

            # Generate annotated image
            annotated = result.plot()
            annotated = Image.fromarray(annotated[:, :, ::-1])

            gallery_images.append(
                (annotated, model_name)
            )

            # Extract detections
            counts = Counter()

            if result.boxes is not None:
                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]
                    score = float(box.conf[0].item())

                    x1, y1, x2, y2 = [
                        round(float(v), 2)
                        for v in box.xyxy[0].tolist()
                    ]

                    counts[class_name] += 1

                    detection_rows.append([
                        model_name,
                        class_name,
                        round(score, 4),
                        x1,
                        y1,
                        x2,
                        y2
                    ])

            total = sum(counts.values())

            summaries.append(
                f"### {model_name}\n"
                f"- Total detections: **{total}**\n"
                f"- Class counts: {dict(counts) if counts else 'No detections'}"
            )

    except Exception as error:
        return [], [], f"**Error:** {str(error)}"

    summary = "\n\n".join(summaries)

    if not detection_rows:
        summary += "\n\nNo objects detected in the uploaded image."

    return gallery_images, detection_rows, summary


# --------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------

with gr.Blocks(title="Adverse Weather Vehicle Detection") as demo:

    gr.Markdown(
        """
        # Vehicle Detection in Adverse Weather

        ### Class-Weighted YOLOv8

        Upload an image captured in adverse weather conditions
        and evaluate vehicle detection using three YOLOv8 configurations.

        Compare standard classification loss against two
        class-weighting strategies.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            input_image = gr.Image(
                label="Upload an Image",
                type="pil",
                height=350
            )

            model_choice = gr.Dropdown(
                choices=[
                    "Compare All Models",
                    *MODELS.keys()
                ],
                value="Compare All Models",
                label="Select Model"
            )

            confidence = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            iou = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.05,
                label="IoU Threshold"
            )

            detect_button = gr.Button(
                "Run Detection",
                variant="primary"
            )

        with gr.Column(scale=1):

            output_gallery = gr.Gallery(
                label="Detection Results",
                columns=2,
                height=450
            )

    gr.Markdown("## Detection Summary")

    output_summary = gr.Markdown()

    output_table = gr.Dataframe(
        headers=[
            "Model",
            "Class",
            "Confidence",
            "X1",
            "Y1",
            "X2",
            "Y2"
        ],
        label="Detection Details",
        interactive=False,
        wrap=True
    )

    detect_button.click(
        fn=run_detection,
        inputs=[
            input_image,
            model_choice,
            confidence,
            iou
        ],
        outputs=[
            output_gallery,
            output_table,
            output_summary
        ]
    )

    gr.Markdown(
        """
        ---
        **Project:** Vehicle Detection in Adverse Weather

        **Dataset:** DAWN (Detection in Adverse Weather Nature)

        **Models:** Baseline YOLOv8, Inverse Square-Root Weighted YOLOv8,
        and Inverse-Frequency Weighted YOLOv8.

        **Note:** Image-level detection counts are not equivalent to
        validation metrics such as precision, recall, or mAP.
        """
    )


# --------------------------------------------------
# LAUNCH APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True
    )
