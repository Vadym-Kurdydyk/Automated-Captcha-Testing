import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CaptchaDataset
from model import SimpleCaptchaModel
import os



def train_full_dataset():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])




    #Load the data and do the preparations
    dataset = CaptchaDataset(csv_file="image_data.csv",
                             root_dir="diverse_math_dataset/",
                             transform=transform)

    size = len(dataset)
    train_size = int(0.8* size)
    test_size = size-train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train_set, shuffle=True, num_workers=2, batch_size=64)
    dataloader_test = DataLoader(test_set, shuffle=False, num_workers=2, batch_size=64)

    #Initiate the model
    model = SimpleCaptchaModel(num_chars=17).to(device)


    # Initiate loss function and an optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()  # Get the model in a training mod
        running_loss = 0.0  # Reset the running loss every epoch
        for batch in dataloader_train:
            # Get the data and move it to GPU
            images, labels = batch

            images = images.to(device)


            # Forward pass
            logits = model(images)
            logits = logits.permute(1, 0, 2)
            logits = logits.log_softmax(2)

            batch_size = images.size(0)

            input_lengths = torch.full(size=(batch_size,), fill_value=50, dtype=torch.long)


            #Define parameters needed for ctc loss
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            targets = torch.cat([torch.tensor(label) for label in labels])

            # Calculate loss
            optimizer.zero_grad()
            loss = criterion(logits, targets, input_lengths, target_lengths)

            #Calculate gradients
            loss.backward()

            #Update steps
            optimizer.step()



if __name__ == "__main__":
    train_full_dataset()