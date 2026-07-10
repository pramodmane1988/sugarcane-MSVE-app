import torch
import torch.nn as nn
import torch.nn.functional as F


class SugarcaneMSVE(nn.Module):
    def __init__(self, input_channels=3, feature_dims=512):
        super(SugarcaneMSVE, self).__init__()

        self.feature_dims = feature_dims

        # -----------------------------
        # Scale 1 (Coarse Features)
        # -----------------------------
        self.scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # -----------------------------
        # Scale 2 (Medium Features)
        # -----------------------------
        self.scale2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # -----------------------------
        # Scale 3 (Fine Features)
        # -----------------------------
        self.scale3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((14, 14))
        )

        # -----------------------------
        # Attention Module
        # -----------------------------
        self.attention = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # -----------------------------
        # Feature Fusion
        # -----------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(896, feature_dims, kernel_size=1),
            nn.BatchNorm2d(feature_dims),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

    def forward(self, x):

        print("\n========== Forward Pass ==========")

        print("Input Shape :", x.shape)

        feat1 = self.scale1(x)
        print("Scale-1 Output :", feat1.shape)

        feat2 = self.scale2(feat1)
        print("Scale-2 Output :", feat2.shape)

        feat3 = self.scale3(feat2)
        print("Scale-3 Output :", feat3.shape)

        feat1_resized = F.interpolate(
            feat1,
            size=(14, 14),
            mode='bilinear',
            align_corners=False
        )

        feat2_resized = F.interpolate(
            feat2,
            size=(14, 14),
            mode='bilinear',
            align_corners=False
        )

        print("Scale-1 Resized :", feat1_resized.shape)
        print("Scale-2 Resized :", feat2_resized.shape)

        concatenated = torch.cat(
            [feat1_resized, feat2_resized, feat3],
            dim=1
        )

        print("Concatenated Shape :", concatenated.shape)

        attention_weights = self.attention(concatenated)

        print("Attention Shape :", attention_weights.shape)

        feat1_weighted = feat1_resized * attention_weights[:, 0:1, :, :]
        feat2_weighted = feat2_resized * attention_weights[:, 1:2, :, :]
        feat3_weighted = feat3 * attention_weights[:, 2:3, :, :]

        weighted_concat = torch.cat(
            [feat1_weighted, feat2_weighted, feat3_weighted],
            dim=1
        )

        print("Weighted Concatenation :", weighted_concat.shape)

        fused_features = self.fusion(weighted_concat)

        print("Final Feature Shape :", fused_features.shape)
        print("==================================\n")

        return {
    "scale1": feat1,
    "scale2": feat2,
    "scale3": feat3,
    "attention": attention_weights,
    "fused": fused_features
}