import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from models.LeNet import LeNet
import dataloader


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # Load Data
    dataset = dataloader.MyMnist(args.dataset)
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * args.train_rate),
                                                         len(dataset) - int(len(dataset) * args.train_rate)],
                                               generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=True)
    test_iter = iter(test_loader)
    test_label_vectors, test_labels, test_imgs = next(test_iter)
    test_label_vectors = test_label_vectors.to(device)
    test_labels = test_labels.to(device)
    test_imgs = test_imgs.to(device)

    # Define net,loss function and optimizer
    Net = LeNet()
    Net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=0.0005)

    # Train model
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (label_vectors, labels, imgs) in enumerate(train_loader):
            # print(label_vectors.shape, labels.shape, imgs.shape)
            imgs = imgs.to(device)
            label_vectors = label_vectors.to(device)
            optimizer.zero_grad()

            result = Net(imgs)
            '''
            print(result.shape)
            print(label_vectors)
            print(result)
            exit(0)
            '''
            loss = loss_function(result, label_vectors)
            loss.backward()
            optimizer.step()
            # print(result)
            # print(loss)
            running_loss += loss
            if step % 100 == 99:
                # print(label_vectors)
                # print(result)
                with torch.no_grad():
                    result = Net(test_imgs)
                    predicts = torch.max(result, dim=1)[1]
                    accuracy = (predicts == test_labels).sum()

                    print(
                        f"epoch:{epoch}, step:{step}, train_loss:{running_loss / 50}, test_accuracy:{accuracy / 5000}")
                    running_loss = 0.0
    print('Finished training.')
    torch.save(Net.state_dict(), "results/LeNet.pth")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("-t", "--train_rate", default=0.7, type=float, help="rate to separate data into train and test")
    parser.add_argument("--dataset", default="datas/sign_mnist_combine.csv", help="path for the dataset")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
