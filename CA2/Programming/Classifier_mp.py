# Import Packages 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import socket
from contextlib import closing
import time
import datetime
import os



# Check for GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Preprocessing

transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


#Functions

def setup(rank, world_size, master_port, backend, timeout):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
       
def load_data(rank, world_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Train
    train_set = STL10(root="/storage/dmls/stl10_data", split="train", download=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                               sampler=train_sampler, 
                                               batch_size=batch_size, 
                                               shuffle=False, 
                                               persistent_workers=True, 
                                               num_workers=1, 
                                               pin_memory=True)

    # Test
    test_set = STL10(root="/storage/dmls/stl10_data", split="test", download=False, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                              sampler=test_sampler, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              persistent_workers=True, 
                                              num_workers=1, 
                                              pin_memory=True)

    return train_loader, test_loader


# Model

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Convolutional 
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Convolutional 
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # FC
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)  
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    # Training Function

def train_model(rank, world_size,master_port,backend , timeout, num_epochs, batch_size):
    setup(rank, world_size, master_port, backend, timeout)
    torch.cuda.set_device(rank)
    train_loader, test_loader = load_data(rank,world_size, batch_size)
    model = SimpleCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        ddp_model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cuda_mem = torch.cuda.max_memory_allocated(device=rank)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        ddp_model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = ddp_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    end_time = time.time()
    print(f"Rank {rank}, Training Time: {end_time - start_time:.2f} seconds")
    print(f"Rank {rank}, CUDA Memory Usage: {cuda_mem / (1024 ** 2):.2f} MB")
    print(f"Rank {rank}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    num_epochs = 10

    batch_size = 32
    backend = 'nccl'

  # world_size = 2  # Number of GPUs
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    timeout = datetime.timedelta(seconds=10)

    start_time = time.time()
    mp.spawn(train_model, nprocs=world_size, args=(world_size, master_port, backend, timeout, num_epochs , batch_size), join=True)
    end_time = time.time()

    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f'Batch size:{batch_size}')
    print(f'Backend:{backend}')
