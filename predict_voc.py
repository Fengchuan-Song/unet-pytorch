import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from nets.unet import Unet
from utils.utils import cvtColor, preprocess_input, resize_image


IMAGE_EXTENSIONS = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
VOC_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict semantic segmentation masks in VOC format.")
    parser.add_argument("--model_path", type=str, default='/data/UNet/U-Net/weights/best_epoch_weights_ep037_mIoU0.789.pth', help="Path to trained .pth weights.")
    parser.add_argument("--input_path", type=str, default='/data_ssd/datasets/WaterScenes/MIPC_SemanticSegmentation/2007_test.txt', help="RGB image file, image directory, or txt file.")
    parser.add_argument("--output_dir", type=str, default="/data/UNet/predict_results/SegmentationClass", help="VOC-style output directory.")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes, including background.")
    parser.add_argument("--backbone", type=str, default="vgg", choices=["vgg", "resnet50"], help="UNet backbone.")
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if it is available.")
    parser.add_argument("--copy_images", action="store_true", help="Copy source images to output JPEGImages.")
    return parser.parse_args()


def collect_image_paths(input_path):
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".txt":
        image_paths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                image_paths.append(Path(line))
        return image_paths
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_model(model_path, num_classes, backbone, device):
    model = Unet(num_classes=num_classes, pretrained=False, backbone=backbone)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        state_dict[key] = value

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def predict_mask(model, image, input_shape, device):
    image = cvtColor(image)
    original_h, original_w = np.array(image).shape[:2]

    image_data, resized_w, resized_h = resize_image(image, (input_shape[1], input_shape[0]))
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
        0,
    )

    with torch.no_grad():
        images = torch.from_numpy(image_data).to(device)
        pred = model(images)[0]
        pred = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy()
        top = int((input_shape[0] - resized_h) // 2)
        left = int((input_shape[1] - resized_w) // 2)
        pred = pred[top: top + resized_h, left: left + resized_w]
        pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1).astype(np.uint8)
    return pred


def save_voc_png(mask, save_path):
    palette = []
    for color in VOC_COLORS:
        palette.extend(color)
    palette.extend([0] * (256 * 3 - len(palette)))

    image = Image.fromarray(mask, mode="P")
    image.putpalette(palette)
    image.save(save_path)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    image_paths = collect_image_paths(args.input_path)
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {args.input_path}")

    output_dir = Path(args.output_dir) / "VOC2007"
    seg_dir = output_dir / "SegmentationClass"
    img_dir = output_dir / "JPEGImages"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, args.num_classes, args.backbone, device)

    for image_path in tqdm(image_paths, desc="Predict"):
        image_id = image_path.stem
        image = Image.open(image_path).convert("RGB")
        mask = predict_mask(model, image, args.input_shape, device)

        save_voc_png(mask, seg_dir / f"{image_id}.png")
        if args.copy_images:
            shutil.copy2(image_path, img_dir / image_path.name)

    print(f"Saved VOC prediction masks to: {seg_dir}")


if __name__ == "__main__":
    main()
