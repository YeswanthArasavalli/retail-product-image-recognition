from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data_cls")
MIN_BOX_SIZE = 16  

splits = ["train2019", "val2019", "test2019"]
json_files = {
    "train2019": "instances_train2019.json",
    "val2019": "instances_val2019.json",
    "test2019": "instances_test2019.json",
}


def load_coco_annotations(json_path: Path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    return images, categories, anns_by_image


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def crop_and_save(image_path: Path, bbox, out_path: Path):
    """
    bbox: [x,y,w,h] in COCO format
    """
    x, y, w, h = bbox
    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return False

    with Image.open(image_path).convert("RGB") as img:
        x2, y2 = x + w, y + h
        crop = img.crop((x, y, x2, y2))
        ensure_dir(out_path.parent)
        crop.save(out_path, format="JPEG")
    return True


def process_split(split_name: str):
    print(f"\nProcessing split: {split_name}")
    json_path = RAW_DIR / json_files[split_name]
    images, categories, anns_by_image = load_coco_annotations(json_path)

    # train2019 -> train, val2019 -> val, test2019 -> test
    out_split_dir = OUT_DIR / split_name.replace("2019", "")

    for img_id, img_info in tqdm(images.items()):
        file_name = img_info["file_name"]
        image_path = RAW_DIR / split_name / file_name

        anns = anns_by_image.get(img_id, [])
        for i, ann in enumerate(anns):
            cat_id = ann["category_id"]
            cat_name = categories[cat_id]
            bbox = ann["bbox"]  # [x,y,w,h]

            base_name = Path(file_name).stem
            out_dir = out_split_dir / cat_name
            out_name = f"{base_name}_ann{i}.jpg"
            out_path = out_dir / out_name

            crop_and_save(image_path, bbox, out_path)


if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    for split in splits:
        process_split(split)
    print("\nDone. Cropped classification dataset created under data_cls/")
