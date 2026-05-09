import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CaptchaDataset
from model import SimpleCaptchaModel
import os


# Helper function to turn model outputs back into readable sequences
def decode_predictions(logits, blank_label=0):
    # We want the most likely character at each time step
    _, max_indices = torch.max(logits, 2)
    max_indices = max_indices.permute(1, 0)  # Swap to [Batch_Size, Sequence_Length]

    predictions = []
    for row in max_indices:
        decoded = []
        last_char = None
        for idx in row:
            idx = idx.item()
            # CTC rules: ignore blanks and consecutive duplicates
            if idx != blank_label and idx != last_char:
                decoded.append(idx)
            last_char = idx
        predictions.append(decoded)
    return predictions


# Main training function
def train_full_dataset():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Image setup
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data
    dataset = CaptchaDataset(csv_file="image_data.csv",
                             root_dir="diverse_math_dataset/",
                             transform=transform)

    # Split data 90/10
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    print(f"Total images: {total_size}")
    print(f"Training on: {train_size}, Validation on: {val_size}")

    # Init model, 17 classes total
    model = SimpleCaptchaModel(num_chars=17).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            logits = model(images)
            logits = logits.permute(1, 0, 2)
            logits = logits.log_softmax(2)

            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=50, dtype=torch.long)
            target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
            targets = torch.cat([l for l in labels])

            optimizer.zero_grad()
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Finished Epoch {epoch + 1} - Avg Training Loss: {avg_loss:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        correct_sequences = 0
        total_sequences = 0

        # No gradients needed for checking accuracy
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)

                val_logits = model(val_images)
                val_logits = val_logits.permute(1, 0, 2)

                # Get predictions using our helper function
                preds = decode_predictions(val_logits)

                # Compare each prediction to the actual label
                for i in range(len(preds)):
                    target_list = val_labels[i].tolist()
                    if preds[i] == target_list:
                        correct_sequences += 1
                    total_sequences += 1

        accuracy = (correct_sequences / total_sequences) * 100
        print(f"Validation Accuracy: {accuracy:.2f}%\n")

        # Save checkpoint
        torch.save(model.state_dict(), "captcha_model_latest.pth")

    print("Finished all epochs.")


if __name__ == "__main__":
    if os.path.exists("image_data.csv"):
        train_full_dataset()
    else:
        print("Error: image_data.csv not found")