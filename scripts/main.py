import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12*4*4)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

def num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train():
    batch_size_list = [100, 500, 1000]
    lr_list = [0.001, 0.01]
    shuffle_list = [True, False]

    for batch_size in batch_size_list:
        for lr in lr_list:
            for shuffle in shuffle_list:
                
                network = Network()
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
                optimiser = optim.Adam(network.parameters(), lr=lr)

                images, labels = next(iter(train_loader))
                grid = torchvision.utils.make_grid(images)

                tb = SummaryWriter()

                comment = f' batchsize ={batch_size} lr ={lr} shuffle ={shuffle}'
                tb = SummaryWriter(comment=comment)
                tb.add_image('images', grid)
                tb.add_graph(network, images)

                for epoch in range(10):

                    total_loss = 0
                    total_correct = 0

                    for batch in train_loader: 
                        images, labels = batch

                        preds = network(images)
                        loss = F.cross_entropy(preds, labels)

                        optimiser.zero_grad()
                        loss.backward()
                        optimiser.step()

                        total_loss += loss.item() * batch_size
                        total_correct += num_correct(preds, labels)

                    tb.add_scalar('loss', total_loss, epoch)
                    tb.add_scalar('number correct', total_correct, epoch)
                    tb.add_scalar('accuracy', total_correct/len(train_set), epoch)

                    for name, weight in network.named_parameters():
                        tb.add_histogram(name, weight, epoch)
                        tb.add_histogram(f'{name}.grad' , weight.grad, epoch)

                    print('epoch:', epoch , 'loss:', total_loss, 'total correct:', total_correct)

                tb.close()