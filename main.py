# %%
import torch

print("Python version check:")
import sys
print(sys.version)

print("\nTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %%
# ============================================================
# Cell 1: Basic imports and environment check
# This cell sets up all libraries we will use throughout
# the project and checks whether the GPU is available.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from PIL import Image

# Metrics and utilities
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_metric

from tqdm import tqdm

# Check device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# ============================================================
# Cell 2: Image transformations
# This defines how input and reference images are preprocessed
# before being fed into the model.
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize all images to same size
    transforms.ToTensor()            # Convert to tensor and scale to [0, 1]
])


# %%
# ============================================================
# Cell 3: Unified Dataset class for UIEB + EUVP
# This dataset loads paired (raw, reference) underwater images
# from multiple datasets and presents them as one dataset.
# ============================================================

class UnderwaterDataset(Dataset):
    def __init__(self, dataset_roots, transform=None):
        """
        Args:
            dataset_roots (list): List of dataset root directories
                                  e.g. ['datasets/UIEB', 'datasets/EUVP']
            transform (callable): Image transformations
        """
        self.image_pairs = []
        self.transform = transform

        for root in dataset_roots:
            raw_dir = os.path.join(root, 'raw')
            ref_dir = os.path.join(root, 'reference')

            raw_images = sorted(os.listdir(raw_dir))
            ref_images = sorted(os.listdir(ref_dir))

            # Safety check
            assert len(raw_images) == len(ref_images), \
                f"Mismatch in {root}: raw={len(raw_images)}, reference={len(ref_images)}"

            for raw_name, ref_name in zip(raw_images, ref_images):
                raw_path = os.path.join(raw_dir, raw_name)
                ref_path = os.path.join(ref_dir, ref_name)

                self.image_pairs.append((raw_path, ref_path))

        print(f"Total image pairs loaded: {len(self.image_pairs)}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        raw_path, ref_path = self.image_pairs[idx]

        raw_img = Image.open(raw_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        if self.transform:
            raw_img = self.transform(raw_img)
            ref_img = self.transform(ref_img)

        return raw_img, ref_img


# %%
# ============================================================
# Cell 4: Create dataset instance
# This combines UIEB and EUVP into one training dataset.
# ============================================================

dataset_roots = [
    '../datasets/UIEB',
    '../datasets/EUVP'
]

train_dataset = UnderwaterDataset(
    dataset_roots=dataset_roots,
    transform=image_transform
)


# %%
# ============================================================
# Cell 5: DataLoader
# This handles batching, shuffling, and fast data loading.
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,      # Safe batch size for RTX 4060
    shuffle=True,      # Mix UIEB and EUVP samples
    num_workers=0,     # Parallel data loading
    pin_memory=True   # Faster transfer to GPU
)


# %%
# ============================================================
# Cell 6: Visual sanity check
# Displays a raw image and its reference counterpart.
# ============================================================

raw_batch, ref_batch = next(iter(train_loader))

# Take the first image in the batch
raw_img = raw_batch[0].permute(1, 2, 0).cpu().numpy()
ref_img = ref_batch[0].permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Raw Input")
plt.imshow(raw_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Reference")
plt.imshow(ref_img)
plt.axis("off")

plt.tight_layout()
plt.show()


# %%
# ============================================================
# Cell 7: Residual Block
# This block learns a residual mapping instead of full features.
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut connection (1x1 conv if channels change)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


# %%
# ============================================================
# Cell 8: Encoder Block
# Downsamples the feature maps while increasing depth.
# ============================================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = self.res_block(x)
        downsampled = self.pool(features)
        return features, downsampled


# %%
# ============================================================
# Cell 9: Decoder Block
# Upsamples and merges features from the encoder.
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.up(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.res_block(x)
        return x


# %%
# ============================================================
# Cell 10: ResUNet Model
# Full encoder-decoder architecture with residual blocks.
# ============================================================

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Output
        out = torch.sigmoid(self.final_conv(d1))
        return out


# %%
# ============================================================
# Cell 11: Model sanity check
# Verifies output shape and GPU compatibility.
# ============================================================

model = ResUNet().to(device)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    dummy_output = model(dummy_input)

print("Input shape :", dummy_input.shape)
print("Output shape:", dummy_output.shape)


# %%
# ============================================================
# Cell 12: VGG16 feature extractor for perceptual loss
# ============================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg.eval()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.l1_loss(pred_features, target_features)


# %%
# ============================================================
# Cell 13: Composite loss (MSE + SSIM + Perceptual)
# ============================================================

class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual = VGGPerceptualLoss()

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        perceptual_loss = self.perceptual(pred, target)

        total_loss = (
            0.7 * mse_loss +
            0.2 * ssim_loss +
            0.1 * perceptual_loss
        )

        return total_loss


# %%
#Checking if loss function is ready 
criterion = CompositeLoss().to(device)

dummy_pred = torch.rand(1, 3, 256, 256).to(device)
dummy_gt   = torch.rand(1, 3, 256, 256).to(device)

loss_val = criterion(dummy_pred, dummy_gt)
print("Composite loss:", loss_val.item())


# %%
# ============================================================
# Cell 14: Transformer Bottleneck
# Adds global context modeling at the deepest layer
# ============================================================

class TransformerBottleneck(nn.Module):
    def __init__(self, channels, num_heads=8, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Flatten spatial dimensions
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # Transformer
        x = self.transformer(x)

        # Restore spatial dimensions
        x = x.permute(0, 2, 1).view(B, C, H, W)

        return x


# %%
# ============================================================
# Cell 15: Hybrid ResUNet with Transformer Bottleneck
# ============================================================

class ResUNetTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # CNN Bottleneck
        self.bottleneck_conv = ResidualBlock(512, 1024)

        # Transformer Bottleneck
        self.bottleneck_transformer = TransformerBottleneck(
            channels=1024,
            num_heads=8,
            num_layers=2
        )

        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck_conv(p4)
        b = self.bottleneck_transformer(b)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = torch.sigmoid(self.final_conv(d1))
        return out


# %%
# ============================================================
# Cell 16: Hybrid model sanity check
# ============================================================

hybrid_model = ResUNetTransformer().to(device)
hybrid_model.eval()

dummy_input = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    dummy_output = hybrid_model(dummy_input)

print("Hybrid output shape:", dummy_output.shape)


# %%
# ============================================================
# Cell 17: Optimizer and learning rate scheduler
# ============================================================

learning_rate = 1e-4
num_epochs = 50 

optimizer = torch.optim.Adam(
    hybrid_model.parameters(),
    lr=learning_rate
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)


# %%
# ============================================================
# Cell 18: Training loop for the hybrid model
# ============================================================

criterion = CompositeLoss().to(device)

best_loss = float('inf')

for epoch in range(num_epochs):
    hybrid_model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for raw_imgs, ref_imgs in progress_bar:
        raw_imgs = raw_imgs.to(device)
        ref_imgs = ref_imgs.to(device)

        optimizer.zero_grad()

        outputs = hybrid_model(raw_imgs)
        loss = criterion(outputs, ref_imgs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.6f}")

    scheduler.step(epoch_loss)

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            hybrid_model.state_dict(),
            '../checkpoints/best_hybrid_resunet_transformer.pth'
        )
        print("Best model saved.")


# %%
# ============================================================
# Cell 19: Load best trained model
# ============================================================

hybrid_model.load_state_dict(
    torch.load('../checkpoints/best_hybrid_resunet_transformer.pth', map_location=device)
)

hybrid_model.eval()
print("Best hybrid model loaded.")


# %%
# ============================================================
# Cell 20: Load best trained hybrid model for evaluation
# ============================================================

hybrid_model.load_state_dict(
    torch.load('../checkpoints/best_hybrid_resunet_transformer.pth', map_location=device)
)
hybrid_model.eval()

print("Hybrid model loaded for evaluation.")


# %%
# ============================================================
# Cell 21: Metric storage initialization
# ============================================================

psnr_scores = []
ssim_scores = []
lpips_scores = []
uiqm_scores = []
uciqe_scores = []


# %%
# ============================================================
# Cell 22: LPIPS setup
# ============================================================

import lpips

lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()


# %%
# ============================================================
# Cell 23: UIQM and UCIQE computation helpers
# ============================================================

import cv2

def compute_uciqe(img):
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0] / 255.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    chroma = np.sqrt(a**2 + b**2)

    return (
        0.4680 * np.std(chroma) +
        0.2745 * np.mean(chroma) +
        0.2576 * np.mean(L)
    )


def compute_uiqm(img):
    img = (img * 255).astype(np.uint8)

    # Colorfulness
    rg = img[:, :, 0] - img[:, :, 1]
    yb = 0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2]

    colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + \
                   0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)

    # Sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharpness = np.var(cv2.Laplacian(gray, cv2.CV_64F))

    # Contrast
    contrast = np.std(gray)

    return 0.0282 * colorfulness + 0.2953 * sharpness + 3.5753 * contrast


# %%
# ============================================================
# Cell 24: Evaluation loop on combined dataset
# ============================================================


with torch.no_grad():
    for raw_imgs, ref_imgs in tqdm(train_loader, desc="Evaluating"):
        raw_imgs = raw_imgs.to(device)
        ref_imgs = ref_imgs.to(device)

        outputs = hybrid_model(raw_imgs)

        for i in range(outputs.size(0)):
            pred = outputs[i].cpu().permute(1, 2, 0).numpy()
            ref  = ref_imgs[i].cpu().permute(1, 2, 0).numpy()

            # PSNR & SSIM
            psnr_scores.append(psnr(ref, pred, data_range=1.0))
            ssim_scores.append(
                ssim_metric(ref, pred, channel_axis=2, data_range=1.0)
            )

            # LPIPS
            pred_t = outputs[i:i+1]
            ref_t  = ref_imgs[i:i+1]
            lpips_scores.append(lpips_model(pred_t, ref_t).item())

            # UIQM & UCIQE
            uiqm_scores.append(compute_uiqm(pred))
            uciqe_scores.append(compute_uciqe(pred))


# %%
# ============================================================
# Cell 24.1: Clean UCIQE values (remove inf / nan)
# ============================================================

uciqe_array = np.array(uciqe_scores)

# Keep only finite values
uciqe_clean = uciqe_array[np.isfinite(uciqe_array)]

print("Original UCIQE samples :", len(uciqe_array))
print("Valid UCIQE samples    :", len(uciqe_clean))
print("Discarded samples      :", len(uciqe_array) - len(uciqe_clean))


# %%
# ============================================================
# Cell 25: Final evaluation metrics (UCIQE removed)
# ============================================================

print("===== Final Hybrid Model Evaluation (UIEB + EUVP Combined) =====")

print(f"PSNR  : {np.mean(psnr_scores):.4f}")
print(f"SSIM  : {np.mean(ssim_scores):.4f}")
print(f"LPIPS : {np.mean(lpips_scores):.4f}")
print(f"UIQM  : {np.mean(uiqm_scores):.4f}")


# %%
# ============================================================
# Cell 26: Visual comparison ( Raw vs Hybrid vs Reference )
# ============================================================

num_samples = 10
shown = 0

plt.figure(figsize=(12, 4 * num_samples))

with torch.no_grad():
    for raw_imgs, ref_imgs in train_loader:
        raw_imgs = raw_imgs.to(device)
        ref_imgs = ref_imgs.to(device)
        outputs = hybrid_model(raw_imgs)

        for i in range(raw_imgs.size(0)):
            if shown >= num_samples:
                break

            raw_img = raw_imgs[i].cpu().permute(1, 2, 0).numpy()
            out_img = outputs[i].cpu().permute(1, 2, 0).numpy()
            ref_img = ref_imgs[i].cpu().permute(1, 2, 0).numpy()

            plt.subplot(num_samples, 3, 3*shown + 1)
            plt.imshow(raw_img); plt.axis("off"); plt.title("Raw")

            plt.subplot(num_samples, 3, 3*shown + 2)
            plt.imshow(out_img); plt.axis("off"); plt.title("Hybrid")

            plt.subplot(num_samples, 3, 3*shown + 3)
            plt.imshow(ref_img); plt.axis("off"); plt.title("Reference")

            shown += 1

        if shown >= num_samples:
            break

plt.tight_layout()
plt.show()


# %%
# ============================================================
# Cell 27: Normalized Metric Distributions 
# Metrics: PSNR, SSIM, LPIPS, UIQM
# ============================================================

def normalize(values):
    values = np.array(values)
    return (values - values.min()) / (values.max() - values.min())


# Normalize metrics
psnr_norm  = normalize(psnr_scores)
ssim_norm  = normalize(ssim_scores)
lpips_norm = normalize(lpips_scores)
uiqm_norm  = normalize(uiqm_scores)

plt.figure(figsize=(12, 8))

# PSNR
plt.subplot(2, 2, 1)
plt.hist(psnr_norm, bins=30, color='tab:blue', alpha=0.8, edgecolor='black')
plt.axvline(psnr_norm.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.title("Normalized PSNR Distribution")
plt.xlabel("Normalized PSNR")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# SSIM
plt.subplot(2, 2, 2)
plt.hist(ssim_norm, bins=30, color='tab:green', alpha=0.8, edgecolor='black')
plt.axvline(ssim_norm.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.title("Normalized SSIM Distribution")
plt.xlabel("Normalized SSIM")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# LPIPS
plt.subplot(2, 2, 3)
plt.hist(lpips_norm, bins=30, color='tab:orange', alpha=0.8, edgecolor='black')
plt.axvline(lpips_norm.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.title("Normalized LPIPS Distribution (Lower is Better)")
plt.xlabel("Normalized LPIPS")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# UIQM
plt.subplot(2, 2, 4)
plt.hist(uiqm_norm, bins=30, color='tab:purple', alpha=0.8, edgecolor='black')
plt.axvline(uiqm_norm.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.title("Normalized UIQM Distribution")
plt.xlabel("Normalized UIQM")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# %%
# ============================================================
# Cell 28: Metric Summary Bar Chart
# ============================================================

metric_names = ['PSNR', 'SSIM', 'LPIPS', 'UIQM']
metric_means = [
    psnr_norm.mean(),
    ssim_norm.mean(),
    lpips_norm.mean(),
    uiqm_norm.mean()
]

plt.figure(figsize=(7, 4))
plt.bar(metric_names, metric_means,
        color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple'],
        edgecolor='black')

plt.title("Normalized Average Evaluation Metrics (Hybrid Model)")
plt.ylabel("Normalized Metric Value")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()



