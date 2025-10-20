import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd


class ConvAE_Encoder(nn.Module):
    def __init__(self, n_channels=1, latent_dim=32, max_filters=128):
        super(ConvAE_Encoder, self).__init__()
        self.max_filters = max_filters
        self.n_channels = n_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, self.max_filters//8, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.BatchNorm2d(self.max_filters//8),
            nn.ReLU(),

            nn.Conv2d(self.max_filters//8, self.max_filters//4, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
            nn.BatchNorm2d(self.max_filters//4),
            nn.ReLU(),

            nn.Conv2d(self.max_filters//4, self.max_filters//2, kernel_size=3, stride=2, padding=1),  # 7x7 → 4x4
            nn.BatchNorm2d(self.max_filters//2),
            nn.ReLU(),
            
            nn.Conv2d(self.max_filters//2, self.max_filters, kernel_size=3, stride=2, padding=1),  # 4x4 → 2x2
            nn.BatchNorm2d(self.max_filters),
            nn.ReLU()
        )

        self.fc = nn.Linear(self.max_filters * 2 * 2, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class ConvAE_Decoder(nn.Module):
    def __init__(self, n_channels=1, latent_dim=32, max_filters=128):
        super(ConvAE_Decoder, self).__init__()
        
        self.n_channels = n_channels
        self.max_filters = max_filters
        self.fc = nn.Linear(latent_dim, self.max_filters * 2 * 2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.max_filters, self.max_filters//2, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2x2 → 4x4
            nn.BatchNorm2d(self.max_filters//2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.max_filters//2, self.max_filters//4, kernel_size=3, stride=2, padding=1, output_padding=int(self.n_channels == 3)),  # 7x7 → 14x14
            nn.BatchNorm2d(self.max_filters//4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(self.max_filters//4, self.max_filters//8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.BatchNorm2d(self.max_filters//8),
            nn.ReLU(),

            nn.ConvTranspose2d(self.max_filters//8, self.n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 → 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.max_filters, 2, 2)
        x = self.decoder(x)
        return x

class ConvAE(nn.Module):
    def __init__(self,n_channels=1, latent_dim=32, max_filters=128):
        super(ConvAE, self).__init__()
        self.encoder = ConvAE_Encoder(n_channels, latent_dim, max_filters)
        self.decoder = ConvAE_Decoder(n_channels, latent_dim, max_filters)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Data loading and transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

def reconstruct_AE(dataset_name, model_name, latent_dim=128, max_filters=1024, batch_size=64, epochs=20, learning_rate=1e-3):
    if dataset_name == 'mnist':
        train_val_data = datasets.MNIST(root='./visual_experiments/data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./visual_experiments/data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_val_data = datasets.CIFAR10(root='./visual_experiments/data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./visual_experiments/data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_val_data))

    # Indices for training and validation
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(train_val_data)))

    # Create Subsets
    train_dataset = Subset(train_val_data, train_indices)
    val_dataset = Subset(train_val_data, val_indices)

    # Hyperparameters
    n_channels = 1 if dataset_name == 'mnist' else 3 

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ConvAE(n_channels, latent_dim, max_filters)
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        _ = model.train()
        running_loss = 0.0
        for data in train_loader:
            images, _ = data  # We only need the images for AE, not the labels
            images = images.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            reconstruction = model(images)

            # Compute loss
            loss = criterion(reconstruction, images)
            running_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation step
        _ = model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                images, _ = data
                images = images.to(device)
                reconstruction = model(images)
                loss = criterion(reconstruction, images)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


    test_samples_indices = []
    for i in range(10):
        # get the first 10 samples of each class
        test_samples_indices.extend(np.where(np.array(test_data.targets) == i)[0][:1])
        
    test_samples = torch.utils.data.Subset(test_data, test_samples_indices)
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False)

    # Reconstruct the test examples
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in test_loader:
            test_images, _ = data
            test_images = test_images.to(device)

            # Perform reconstruction
            reconstructions = model(test_images)
    
    # Save test examples as CSV tabular
    reconstructions = reconstructions.view(reconstructions.size(0), -1)
    reconstructions = reconstructions.cpu().numpy()
    reconstructions = pd.DataFrame(reconstructions)
    reconstructions.to_csv(f'visual_experiments/reconstructions/{dataset_name}/method_comparison/{model_name}.csv', index=False)

reconstruct_AE('mnist', 'ConvAE', epochs=20, latent_dim=32)

