import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from models.LeNet import LeNet
import dataloader
import argparse

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # Load and split Data
    dataset = dataloader.MyMnist(args.dataset)
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * args.train_rate),
                                                         len(dataset) - int(len(dataset) * args.train_rate)],
                                               generator=generator)
    test_size = len(dataset) - int(len(dataset) * args.train_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)
    test_iter = iter(test_loader)
    test_label_vectors, test_labels, test_imgs = next(test_iter)
    test_labels = test_labels.to(device)
    test_imgs = test_imgs.to(device)

    # Define net,loss function and optimizer
    Net = LeNet()
    Net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.learning_rate)

    # Train model
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (label_vectors, labels, imgs) in enumerate(train_loader):
            imgs = imgs.to(device)
            label_vectors = label_vectors.to(device)
            optimizer.zero_grad()
            result = Net(imgs)

            # Update Net
            loss = loss_function(result, label_vectors)
            loss.backward()
            optimizer.step()
            running_loss += loss

            # Test net
            if step % args.step == args.step-1:
                with torch.no_grad():
                    result = Net(test_imgs)
                    predicts = torch.max(result, dim=1)[1]
                    accuracy = (predicts == test_labels).sum()

                    print(f"epoch:{epoch}, step:{step}, "
                          f"train_loss:{running_loss / 50}, test_accuracy:{accuracy / test_size}")
                    running_loss = 0.0
    print('Finished training.')
    torch.save(Net.state_dict(), args.save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("-t", "--train_rate", default=0.7, type=float, help="rate to separate data into train and test")
    parser.add_argument("--dataset", default="datas/sign_mnist_combine.csv", help="path for the dataset")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="learning rate for the net")
    parser.add_argument("-s", "--save_path", default="results/LeNet.pth", help="path to save the model")
    parser.add_argument("-S", "--step", default=100, type=int, help="step to print training info")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    inp_args = parse_args()
    train(inp_args)
