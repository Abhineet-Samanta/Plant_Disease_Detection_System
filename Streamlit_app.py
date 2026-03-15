import streamlit as st
import requests
import base64

st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# -------------------------
# BACKGROUND IMAGE WITH OVERLAY
# -------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Make all text white */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}

        /* Glass card effect */
        .block-container {{
            background: rgba(0,0,0,0.35);
            padding: 30px;
            border-radius: 15px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local(r"C:\Users\Admin\Pictures\Agriculture.jpg")

# -------------------------
# HEADER
# -------------------------
st.title("Plant Disease Detection System")
st.write("Upload a plant leaf image to detect possible diseases using AI.")

st.divider()

# -------------------------
# PLANT SELECTION
# -------------------------
plant = st.selectbox(
    "Select Plant Type",
    ["Select Plant", "Potato", "Bell Pepper", "Corn"]
)

# -------------------------
# SHOW PLANT IMAGE
# -------------------------
if plant == "Potato":
    st.image(
        r"C:\Users\Admin\Pictures\Potato.jpg",
        caption="Potato Crop",
        use_container_width=True
    )

elif plant == "Bell Pepper":
    st.image(
        r"C:\Users\Admin\Pictures\Pepper.jpg",
        caption="Bell Pepper Plant",
        use_container_width=True
    )

elif plant == "Corn":
    st.image(
        r"C:\Users\Admin\Pictures\Corn.webp",
        caption="Corn Field",
        use_container_width=True
    )

# -------------------------
# IMAGE UPLOAD
# -------------------------
if plant != "Select Plant":

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Leaf Image", width=400)

        if plant == "Potato":
            url = "http://127.0.0.1:8006/predict/potato"

        elif plant == "Bell Pepper":
            url = "http://127.0.0.1:8006/predict/pepper"

        elif plant == "Corn":
            url = "http://127.0.0.1:8006/predict/corn"

        with st.spinner("Analyzing image..."):

            response = requests.post(
                url,
                files={"file": uploaded_file.getvalue()}
            )

        if response.status_code == 200:

            result = response.json()

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Plant:", plant)

            with col2:
                st.write("Disease:", result["class"])

            confidence = round(result["confidence"] * 100, 2)

            st.progress(int(confidence))
            st.write("Confidence:", confidence, "%")

        else:
            st.error("Server error. Please check FastAPI server.")