import os
import io

import streamlit as st
import pandas as pd
import cv2

from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DAWN Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection in Adverse Weather")
st.markdown(
    "Compare the original YOLOv8 model with our "
    "class-weighted YOLOv8 model on adverse-weather images."
)

st.divider()


# --------------------------------------------------
# 2. MODEL PATHS
# --------------------------------------------------

BASELINE_PATH = os.getenv(
    "BASELINE_MODEL",
    "/content/drive/MyDrive/dawn_models/baseline/best.pt"
)

WEIGHTED_PATH = os.getenv(
    "WEIGHTED_MODEL",
    "/content/drive/MyDrive/dawn_models/weighted/best.pt"
)


# --------------------------------------------------
# 3. LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# --------------------------------------------------
# 4. RUN DETECTION
# --------------------------------------------------

def run_detection(model, image, confidence, iou):

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        verbose=False
    )

    result = results[0]

    # Convert annotated image from BGR to RGB
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())

            class_name = result.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "Class": class_name,
                "Confidence": round(confidence_score, 3),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1)
            })

    detection_df = pd.DataFrame(detections)

    return annotated_image, detection_df


# --------------------------------------------------
# 5. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Compare Both Models",
        "Baseline YOLOv8",
        "Class-Weighted YOLOv8"
    ]
)


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

st.header("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an adverse-weather road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        io.BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    st.subheader("Original Image")
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------
    # 7. LOAD SELECTED MODELS AND RUN INFERENCE
    # --------------------------------------------------

    if st.button("Run Vehicle Detection", type="primary"):

        models_to_run = []

        if model_choice in [
            "Compare Both Models",
            "Baseline YOLOv8"
        ]:
            models_to_run.append(
                ("Baseline YOLOv8", BASELINE_PATH)
            )

        if model_choice in [
            "Compare Both Models",
            "Class-Weighted YOLOv8"
        ]:
            models_to_run.append(
                ("Class-Weighted YOLOv8", WEIGHTED_PATH)
            )

        try:

            model_results = {}

            with st.spinner("Loading models and running detection..."):

                for model_name, model_path in models_to_run:

                    if not os.path.exists(model_path):
                        raise FileNotFoundError(
                            f"Model checkpoint not found: {model_path}"
                        )

                    model = load_model(model_path)

                    annotated, detections = run_detection(
                        model,
                        image,
                        confidence,
                        iou
                    )

                    model_results[model_name] = {
                        "image": annotated,
                        "detections": detections
                    }

            st.success("Detection completed successfully!")

            # --------------------------------------------------
            # 8. DISPLAY MODEL PREDICTIONS
            # --------------------------------------------------

            st.header("Detection Results")

            columns = st.columns(len(model_results))

            for column, (model_name, output) in zip(
                columns,
                model_results.items()
            ):

                with column:

                    st.subheader(model_name)

                    st.image(
                        output["image"],
                        use_container_width=True
                    )

                    st.metric(
                        "Objects Detected",
                        len(output["detections"])
                    )

            st.divider()

            # --------------------------------------------------
            # 9. DISPLAY DETECTION DETAILS
            # --------------------------------------------------

            st.header("Detection Details")

            for model_name, output in model_results.items():

                st.subheader(model_name)

                detections = output["detections"]

                if detections.empty:

                    st.info("No objects detected in this image.")

                else:

                    st.dataframe(
                        detections,
                        use_container_width=True,
                        hide_index=True
                    )

            # --------------------------------------------------
            # 10. MODEL COMPARISON
            # --------------------------------------------------

            if len(model_results) == 2:

                st.divider()
                st.header("Model Comparison")

                comparison_data = []

                for model_name, output in model_results.items():

                    detections = output["detections"]

                    class_counts = (
                        detections["Class"].value_counts().to_dict()
                        if not detections.empty
                        else {}
                    )

                    average_confidence = (
                        detections["Confidence"].mean()
                        if not detections.empty
                        else 0
                    )

                    comparison_data.append({
                        "Model": model_name,
                        "Total Detections": len(detections),
                        "Average Confidence": round(
                            average_confidence, 3
                        ),
                        "Detected Classes": ", ".join(
                            sorted(class_counts.keys())
                        ) if class_counts else "None"
                    })

                comparison_df = pd.DataFrame(comparison_data)

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These results compare predictions on one uploaded image. "
                    "They are not substitutes for validation metrics such as "
                    "mAP, precision, or recall."
                )

        except Exception as error:

            st.error(f"An error occurred: {error}")

else:

    st.info(
        "Upload an image to begin vehicle detection."
    )


# --------------------------------------------------
# 11. PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("About This Project"):

    st.markdown("""
    This project investigates vehicle detection under adverse
    weather conditions using the DAWN dataset and YOLOv8.

    **Models compared:**

    - Baseline YOLOv8
    - YOLOv8 with class-weighted classification loss

    **Project objective:**

    Investigate whether class weighting improves detection
    performance on underrepresented object categories.

    **Evaluation metrics:**

    - Precision
    - Recall
    - mAP@50
    - mAP@50-95
    - Per-class Average Precision
    """)
