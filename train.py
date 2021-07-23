import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import os
import time

from repVGG import Net
import config
from graph_noisy_loss import graph_noisy_loss
from contrastive_loss import ContrastiveLoss


def train():
    train_dataset = dataset.ImageFolder(config.train_dataset_path, transform=config.train_transform)
    train_dataloader = DataLoader(train_dataset, config.train_batch_size, shuffle=True, drop_last=True)
    test_dataset = dataset.ImageFolder(config.test_dataset_path, transform=config.test_transform)
    test_dataloader = DataLoader(test_dataset, config.test_batch_size, shuffle=False)
    au_dataset = dataset.ImageFolder(config.au_dataset_path, transform=config.train_transform)
    au_dataloader = DataLoader(au_dataset, config.train_batch_size, shuffle=True)

    # define GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    if os.path.exists(config.model_path) is not True:
        net = Net().to(device)
    else:
        # net = torch.load(config.model_path)
        net = Net().to(device)

    cross_loss = nn.CrossEntropyLoss()
    con_loss = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4)

    best_acc = 0.
    for epoch in range(config.epochs):
        net.train()
        sum_loss = 0.
        correct = 0.
        total = 0.
        con_batch = config.train_batch_size // 2
        length = len(train_dataloader)
        since = time.time()

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            au_inputs, au_labels = next(iter(au_dataloader))
            au_inputs, au_labels = au_inputs.to(device), au_labels.to(device)
            labels1, labels2 = labels[con_batch:], labels[:con_batch]
            labels_con = torch.tensor(labels1.cpu().numpy() == labels2.cpu().numpy(), dtype=torch.float32).to(device)
            optimizer.zero_grad()

            outputs, adj = net(inputs)
            outputs1, outputs2 = outputs[con_batch:], outputs[:con_batch]
            au_outputs = net.au_forward(au_inputs)

            loss1 = cross_loss(outputs, labels)
            loss2 = graph_noisy_loss(labels.cpu(), adj.cpu())
            loss3 = con_loss(outputs1, outputs2, labels_con) * 0.1
            loss4 = cross_loss(au_outputs, au_labels) * 0.3
            loss = loss1 + loss2 + loss3 + loss4

            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            sum_loss += loss.item()

            _, pre = torch.max(outputs.data, 1)

            total += outputs.size(0)
            correct += torch.sum(pre == labels.data)
            train_acc = correct / total

            print('[epoch:%d, iter:%d] Loss: %f | Loss1: %f | Loss2: %f | Loss3: %f | Loss4: %f | Acc: %f | Time: %f'
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss/(i + 1), loss1, loss2, loss3, loss4, train_acc, time.time() - since))

        # scheduler.step()

        # start to test
        if epoch % 1 == 0:
            print("start to test:")
            with torch.no_grad():
                correct = 0.
                total = 0.
                loss = 0.
                for i, data in enumerate(test_dataloader):
                    net.eval()
                    inputs_test, labels_test = data
                    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                    outputs_test, _ = net(inputs_test)
                    loss += cross_loss(outputs_test, labels_test)

                    # present_max, pred = torch.max(outputs.data, 1)
                    _, pred = torch.max(outputs_test.data, 1)

                    total += labels_test.size(0)
                    correct += torch.sum(pred == labels_test.data)
                test_acc = correct / total
                print('test_acc:', test_acc, '| time', time.time() - since)
                model_name = 'vgg19_bn_graph_over_classifying'
                f = open('{}.txt'.format(model_name), 'a')
                f.write(str(test_acc.item()) + '\n')
                f.close()
                if test_acc.item() > best_acc and epoch > 100:
                    best_acc = test_acc.item()
                    torch.save(net, '{}-{}.pkl'.format(model_name, epoch))


def test():
    test_dataset = dataset.ImageFolder(config.test_dataset_path, transform=config.test_transform)
    test_dataloader = DataLoader(test_dataset, config.test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load(config.model_path)

    with torch.no_grad():
        correct = 0.
        total = 0.
        count = 0
        labels_true = []
        labels_pred = []
        presents = []
        features_list = []
        f = open('error_index.txt', 'w')
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, features = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(pred == labels)
            for index_ in range(len(labels)):
                index = index_ + 1
                label_true = int(labels[index_])
                label_pred = int(pred[index_])
                if label_true != label_pred:
                    f.write(str(label_true) + ' ' + str(index + count - (label_true + 1) * 25) + '\n')
            count += len(labels)
            for label in labels:
                labels_true.append(label.cpu().data.numpy())
            for label in pred:
                labels_pred.append(label.cpu().data.numpy())
            for feature in features:
                features_list.append(feature)
            for present in outputs:
                presents.append(present.cpu().data.numpy())
            print('Extracted {}-th feature'.format(count))
        np.save('labels_true.npy', labels_true)
        np.save('labels_pred.npy', labels_pred)
        np.save('presents.npy', presents)
        np.save('features.npy', features_list)
        test_acc = correct / total
        f.close()
        print(test_acc)


if __name__ == '__main__':
    test()
