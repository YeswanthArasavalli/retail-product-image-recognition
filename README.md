# 🛒 Retail Product Recognition – Deep Learning (EfficientNet-B0 · PyTorch)

This project is an end-to-end image classification system for retail SKU identification using the **Retail Product Checkout (RPC)** dataset (~200+ product categories).

It includes:

- Dataset preparation pipeline (COCO → cropped classification dataset)
- Model training using PyTorch + EfficientNet-B0
- Evaluation with classification reports and confusion matrix
- Interactive Streamlit app to upload product images and view predictions
- (Optional) Flask inference API for deployment use cases

**Live Demo:** [Try it here](https://retail-image-recognition.streamlit.app/)

---

## 🚀 Demo

Run the UI:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/retail_streamlit_app.py
```

Then open:

- http://localhost:8501

Upload a product image to view:

- Top-1 prediction
- Confidence score
- Top-5 probabilities bar chart

---

## 📂 Project Structure

```
retail-product-image-recognition/
├── src/
│   ├── prepare_dataset.py
│   ├── app.py
│   ├── train_retail_classifier_torch.py
│   ├── evaluate_test_torch.py
│   ├── inference_api.py
│   └── retail_streamlit_app.py
├── models_torch/
│   └── best_model_efficientnet_b0.pth
├── eval_outputs/
│   ├── classification_report.json
│   ├── confusion_matrix.csv
│   └── test_predictions_detailed.csv
├── requirements.txt
├── predictions_test.csv
├── test_api_request.py
├── .gitignore
└── README.md
```

---

## 🧠 Model

**Architecture:** EfficientNet-B0
**Framework:** PyTorch
**Image size:** 224×224
**Classes:** ~200 retail SKUs
**Training:** Mixed precision (AMP), AdamW optimizer

**Final performance:**

- Top-1 validation accuracy: ~84.9%
- Training hardware: NVIDIA RTX 2050 (Windows local GPU)

---

## 📊 Evaluation Outputs

Generated reports include:

- Precision, recall, F1-score (per class)
- Confusion matrix
- Predictions CSV for Tableau or BI analysis
- Correct vs misclassified breakdown

---

## 📦 Use Cases

- Automated retail checkout
- Inventory analytics and shelf insights
- Retail product catalog cleanup
- Template for computer vision MLOps / deployment pipelines

---

## 🏁 Next Improvements

- ONNX / TensorRT optimization
- Real-time webcam inference
- Self-supervised or contrastive retraining to improve robustness

---

## 📜 License

This project uses processed annotations from the RPC dataset (not included in this repository).
All training results and code belong to the author; please respect dataset and license terms from the original RPC source.

---

## 👤 Author

**Yeswanth Arasavalli** — Data Scientist (Computer Vision · Deep Learning)
- Portfolio: https://yeswantharasavalli.me
- LinkedIn: https://linkedin.com/in/yeswantharasavalli
- Email: yeswantharasavalli@gmail.com
