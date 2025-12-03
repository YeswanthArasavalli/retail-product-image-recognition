from io import BytesIO

import requests
import streamlit as st
from PIL import Image
import pandas as pd


DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Retail Product Classifier (API)", layout="centered")
st.title("🛒 Retail Product Classifier – API Client")

st.caption(
    "Frontend in Streamlit → Backend model served via Flask API.\n"
    "Make sure `python -m src.inference_api` is running before using this page."
)


api_url = st.text_input("Prediction API URL", value=DEFAULT_API_URL)

uploaded = st.file_uploader(
    "Upload a product image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # show preview
    image = Image.open(BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    
    uploaded.seek(0)

    if st.button("🔮 Predict"):
        try:
            with st.spinner("Contacting API and running model..."):
                files = {
                    "image": (uploaded.name, uploaded.read(), uploaded.type)
                }
                response = requests.post(api_url, files=files, timeout=60)

            if response.status_code != 200:
                st.error(f"API returned status {response.status_code}: {response.text}")
            else:
                data = response.json()

                pred_class = data.get("predicted_class", "N/A")
                conf = float(data.get("confidence", 0.0))
                top5 = data.get("top5", [])

                st.subheader("Top-1 Prediction")
                st.markdown(f"**Class:** `{pred_class}`")
                st.markdown(f"**Confidence:** `{conf*100:.2f}%`")

                if top5:
                    st.subheader("Top-5 Probabilities")
                    df = pd.DataFrame(
                        {
                            "class": [t["class"] for t in top5],
                            "score": [float(t["score"]) for t in top5],
                        }
                    )
                    df.set_index("class", inplace=True)
                    st.bar_chart(df)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. "
                "Is `python -m src.inference_api` running on the given URL?"
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")

else:
    st.info("Upload a product image to get predictions from the API.")
