from dataset import NetflixDataset
import cv2
import random
import torch
import torchvision
import numpy as np

if __name__ == "__main__":

    dataset = NetflixDataset("../dataset/filenames.json", "train")
    n_classes = dataset.n_classes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=0)
    print(len(dataset))

    ids = [x for x in range(len(dataset))]
    random.shuffle(ids)

    # for id in ids:
    #     img = dataset.__getitem__(id)
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    print(model)
    model.cuda()
    model.train()

    # Training loop
    epochs = 3
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        print(epoch)

        for step, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.cuda()

            labels = labels.cuda()
            optimizer.zero_grad()
            preds = model(imgs)

            output = loss(preds, labels)
            output.backward()

            optimizer.step()

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
