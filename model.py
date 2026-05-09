# For the time being we will use pre-made module for captchas to train our model

# Simple LLM generated cnn model to test if the dataset is behaving well
import torch
import torch.nn as nn


class SimpleCaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(SimpleCaptchaModel, self).__init__()

        # 1. Feature Extractor (CNN)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces size: 200x50 -> 100x25

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces size: 100x25 -> 50x12
        )

        # 2. Recurrent Layer (LSTM)
        # We treat the width of the image (50) as our sequence length
        self.lstm = nn.LSTM(input_size=64 * 12, hidden_size=128,
                            num_layers=2, bidirectional=True, batch_first=True)

        # 3. Output Layer
        # Bidirectional LSTM has hidden_size * 2
        self.classifier = nn.Linear(128 * 2, num_chars)

    def forward(self, x):
        # x shape: [batch, 3, height, width]
        x = self.conv_layer(x)

        # Reshape for LSTM: [batch, width, features]
        # We permute so the width (the sequence) is the second dimension
        x = x.permute(0, 3, 1, 2)
        batch, seq_len, channels, height = x.size()
        x = x.view(batch, seq_len, channels * height)

        x, _ = self.lstm(x)
        x = self.classifier(x)

        return x


# Testing the flow
if __name__ == "__main__":
    # Assuming your vocab size is 16 (0-9, +, -, *, /, =, blank)
    model = SimpleCaptchaModel(num_chars=16)

    # Fake batch: 4 images, 3 channels, 50px height, 200px width
    test_input = torch.randn(4, 3, 50, 200)
    output = model(test_input)

    print(f"Input Shape: {test_input.shape}")
    print(f"Output Shape: {output.shape}")  # Should be [batch, seq_len, num_chars]