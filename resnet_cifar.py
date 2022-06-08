import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
import os
import time

NUM_WORKERS = int(os.cpu_count() / 2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64

def cifar_dataloader():
    CIFAR_MEAN, CIFAR_STD = (0.491, 0.482, 0.446), (0.247, 0.243, 0.262)
    train_transforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    test_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    train = DataLoader(datasets.CIFAR10("./cifar_data", transform=train_transforms, download=True), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test = DataLoader(datasets.CIFAR10("./cifar_data", transform=test_transforms, train=False, download=True), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    return train, test


class Trainer:
    def __init__(self, model, epochs, device=None):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.train_data, self.test_data = cifar_dataloader()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # if not test-only
        if epochs > 0:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            self.epochs = epochs
            self.out_dir = "./checkpoints/"

    def run_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def run_epoch(self):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            # if epoch%5==0:
                # self.evaluate(self.model, 4)

            for inputs, targets in self.train_data:
                self.run_batch(inputs, targets)

            if self.scheduler is not None:
                self.scheduler.step()
    
    def evaluate(self, max_batch=None):
        L = 0
        A = 0
        t0 = time.time()
        with torch.inference_mode():
            for b, (x, y) in enumerate(self.test_data):
                if max_batch and b == max_batch:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                acc = accuracy(preds, y)
                L += loss.item()
                A += acc.item()
        elapsed = time.time() - t0
        L /= b
        A /= b
        print(f"Loss: {L} \nAccuracy: {A}")
        print("="*20)
        print(f"Time taken ({b * BATCH_SIZE} CIFAR test samples): {elapsed}")

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.out_dir)
        print("Model state dict saved at model.pth")


