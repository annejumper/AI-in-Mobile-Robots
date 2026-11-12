"""
The finger-counting network: 3 convolution layers, each followed by ReLU and 2x2 max pooling,
then one linear layer that turns everything into 6 scores (0-5 fingers).

    input   3 x 64 x 64     colour image of the red square
    conv1  16 x 64 x 64     16 filters, 5x5   -> ReLU -> maxpool -> 16 x 32 x 32
    conv2  32 x 32 x 32     32 filters, 3x3   -> ReLU -> maxpool -> 32 x 16 x 16
    conv3  64 x 16 x 16     64 filters, 3x3   -> ReLU -> maxpool -> 64 x  8 x  8
    flatten                 64*8*8 = 4096 numbers
    linear                  4096 -> 6 scores  (softmax turns them into probabilities)

Trained with cross-entropy loss, because this is choosing a category, not predicting a number.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

from config import IMG_SIZE, CLASSES


class FingerCNN(nn.Module):
    """
    4 conv blocks (conv -> batch norm -> ReLU), the first 3 followed by a 2x2 max-pool:
        64x64x3 -> 32x32x16 -> 16x16x32 -> 8x8x64 -> 8x8x128
    then global average pooling: each of the 128 features is averaged over the whole image,
    so the answer depends on how much "finger" the network sees, not where the hand is.
    Batch norm keeps every layer's numbers in a steady range (faster, steadier training,
    less sensitive to lighting).
    """
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(c_in, c_out, k, pool=True):
            layers = [nn.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False),
                      nn.BatchNorm2d(c_out), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(
            *block(3, 16, 5), *block(16, 32, 3), *block(32, 64, 3), *block(64, 128, 3, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)   # randomly drops numbers while training so it can't memorise
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x))
        x = torch.flatten(x, 1)
        return self.classifier(self.dropout(x))


def preprocess(bgr_crop):
    """OpenCV crop (BGR, any size) -> 3 x 64 x 64 tensor in [-1, 1]. Used for training AND running."""
    img = cv2.resize(bgr_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy((img - 0.5) / 0.5).permute(2, 0, 1)


def load_model(path, device="cpu"):
    model = FingerCNN()
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, bgr_crop, device="cpu"):
    """(fingers, confidence, all 6 probabilities) for one crop."""
    probs = torch.softmax(model(preprocess(bgr_crop).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    k = int(probs.argmax())
    return CLASSES[k], float(probs[k]), probs
