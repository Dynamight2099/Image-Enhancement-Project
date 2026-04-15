# Underwater Image Enhancement using Hybrid ResUNet–Transformer

## Overview

This project presents a deep learning model for underwater image enhancement using a hybrid ResUNet–Transformer architecture. The model improves image quality by addressing common underwater issues such as color distortion, low contrast, and poor visibility.

The approach combines convolutional layers for local feature extraction with a Transformer module for capturing global context, resulting in improved visual quality and structural consistency.

---

## Results

The model was evaluated on two standard underwater datasets: UIEB and EUVP.

| Dataset | PSNR  | SSIM | LPIPS | UIQM |
| ------- | ----- | ---- | ----- | ---- |
| UIEB    | 25.81 | 0.86 | 0.12  | 3.89 |
| EUVP    | 27.54 | 0.90 | 0.07  | 3.99 |

**Key observations:**

* Improved perceptual quality (lower LPIPS)
* Strong structural preservation (high SSIM)
* Better performance on heavily degraded images (EUVP)

---

## Dataset

The project uses the following datasets:

* UIEB (Underwater Image Enhancement Benchmark)
* EUVP (Enhancement of Underwater Visual Perception)

Download link:
https://drive.google.com/drive/folders/1QeaCoxrXQA5C0W2Ra3aV7RGtzstEihoW?usp=drive_link

---

## Notes

* The model is trained on a combined dataset and evaluated separately
* A pretrained YOLO model is used to assess feature visibility after enhancement
* The project is implemented using PyTorch

---

## Author

Zayan Zubair
