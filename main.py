from dataset import NetflixDataset
import random
import torch
import torchvision
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    dataset = NetflixDataset("../dataset/filenames.json", "train")
    val_dataset = NetflixDataset("../dataset/filenames.json", "val")
    n_classes = dataset.n_classes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                             shuffle=True, num_workers=0)
    print(len(dataset))

    ids = [x for x in range(len(dataset))]
    random.shuffle(ids)

    # for id in ids:
    #     img = dataset.__getitem__(id)
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
    model.cuda()

    # Training loop
    epochs = 20
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    for epoch in range(epochs):
        print("Epoch", epoch)
        model.train()
        if epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = 0.00001
        correct = 0
        total = 0
        total_loss = 0
        n_batches = 0
        for (imgs, labels) in tqdm(dataloader):
            # print(step)
            imgs = imgs.cuda()

            labels = labels.cuda()
            optimizer.zero_grad()
            preds = model(imgs)

            loss = loss_fn(preds, labels)
            loss.backward()
            total_loss += loss.item()

            _, preds = torch.max(preds, 1)
            preds = preds.cpu()
            labels = labels.cpu()
            correct += np.sum([labels[j] == preds[j] for j in range(len(labels))])
            total += len(labels)
            n_batches += 1

            optimizer.step()
        accuracy = correct / total
        total_loss = total_loss / n_batches
        train_accuracy.append(accuracy)
        train_loss.append(total_loss)

        model.eval()
        correct = 0
        total = 0
        n_batches = 0
        total_loss = 0
        with torch.no_grad():
            for step, (imgs, labels) in enumerate(val_dataloader):
                imgs = imgs.cuda()

                labels = labels.cuda()
                preds = model(imgs)
                loss = loss_fn(preds, labels)
                total_loss += loss.item()
                _, preds = torch.max(preds, 1)
                preds = preds.cpu()
                labels = labels.cpu()
                correct += np.sum([labels[j] == preds[j] for j in range(len(labels))])
                total += len(labels)
                n_batches += 1
        accuracy = correct / total
        total_loss = total_loss / n_batches
        val_accuracy.append(accuracy)
        val_loss.append(total_loss)

    with open("statistics35s.dat", "wb") as f:
        for stat in [train_accuracy, train_loss, val_accuracy, val_loss]:
            print(stat)
            np.save(f, stat)

    torch.save(model.state_dict(), "model35s.pth")

    model.eval()
    test_dataset = NetflixDataset("../dataset/filenames.json", "test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                             shuffle=True, num_workers=0)

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_dataloader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()
            print("total", len(targets))
            print("correct", np.sum([targets[j] == preds[j] for j in range(len(targets))]))
