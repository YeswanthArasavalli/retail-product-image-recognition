import requests

url = "http://127.0.0.1:8000/predict"

image_path = "D:/Projects/retail_image_recognition/data_cls/test/74_drink/coke.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.json())
