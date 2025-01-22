import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from sklearn.metrics import accuracy_score

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def load_data():
    train_x = np.load('train_data/train_x.npy').astype(np.float32)
    train_y = np.load('train_data/train_y.npy')
    test_x = np.load('test_data/test_x.npy').astype(np.float32)
    test_y = np.load('test_data/test_y.npy')

    return train_x, train_y, test_x, test_y

def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    train_x, train_y, test_x, test_y = load_data()
    num_classes = len(np.unique(train_y))
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    # Create DistributedSampler
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, sampler=test_sampler,shuffle=False)

    device = torch.device("cpu")
    model = SimpleModel(input_size=512, hidden_size=32, output_size=num_classes).to(device)
    ddp_model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    print(f"Training Started for Rank = {rank}")

    start_time = time.time()
    num_epochs = 10
    final_train_accuracy = 0
    final_test_accuracy = 0

    for epoch in range(num_epochs):
        # Training Phase
        ddp_model.train()
        train_sampler.set_epoch(epoch)  
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Evaluation Phase
        ddp_model.eval()
        epoch_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = ddp_model(batch_x)
                loss = criterion(outputs, batch_y)

                epoch_test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += batch_y.size(0)
                correct_test += (predicted == batch_y).sum().item()

        test_loss = epoch_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        final_train_accuracy = train_accuracy
        final_test_accuracy = test_accuracy

    total_time = time.time() - start_time

    if rank == 0:
        print(f"Final Train Accuracy: {final_train_accuracy:.2f}%")
        print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
        print(f"Total Training Time: {total_time:.2f} seconds")

        CHECKPOINT_PATH = "model_checkpoint.pth"
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    train(rank, world_size)
