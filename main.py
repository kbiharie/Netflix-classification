from dataset import NetflixDataset
import random
import torch
import torchvision
import numpy as np
import cv2
from tqdm import tqdm


def calculate_loss_accuracy(model, imgs, labels, purpose="val"):
    """
    Calculate the loss and amount correct of a batch
    :param model: The neural network
    :param imgs: The images of the batch
    :param labels: The corresponding labels of the batch
    :param purpose: Whether the batch is used to train or validate, "train" or "val"
    :return: Batch loss and amount correct in batch
    """

    # Put images and labels on the gpu
    imgs = imgs.cuda()
    labels = labels.cuda()

    # Predicted labels of the images
    preds = model(imgs)
    # The loss of the predictions
    loss = loss_fn(preds, labels)

    # Backpropagate if the model is training
    if purpose == "train":
        loss.backward()

    # Find label prediction
    _, preds = torch.max(preds, 1)
    # Put predictions and labels on the cpu
    preds = preds.cpu()
    labels = labels.cpu()

    # Calculate amount of correctly predicted results
    correct = np.sum([labels[j] == preds[j] for j in range(len(labels))])

    # Return loss and amount correct
    return loss.item(), correct


def show_random_images(train_dataset):
    """
    Test function to show images of the train dataset
    :param train_dataset:
    :return:
    """

    # Randomize the order of the train dataset ids
    ids = [x for x in range(len(train_dataset))]
    random.shuffle(ids)

    # Show images
    for id in ids:
        img, label = train_dataset.__getitem__(id)
        img = img.cpu().permute(1, 2, 0)
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", img)
        cv2.waitKey(0)


if __name__ == "__main__":

    json_file = "../dataset/filenames_animated.json"
    model_name = "model_animated_10"

    # Train and validation datasets
    train_dataset = NetflixDataset(json_file, "train")
    val_dataset = NetflixDataset(json_file, "val")
    # Number of classes in the dataset
    n_classes = train_dataset.n_classes
    # Train and validation dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                             shuffle=True, num_workers=0)

    # Test functions to show random images from the train dataset
    # show_random_images(train_dataset)

    # Load a pretrained resnet50 model
    model = torchvision.models.resnet50(pretrained=True)
    # Edit the last layer to the amount of classes our dataset has
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
    # Put the model on the gpu
    model.cuda()

    # Training loop
    # Amount of epochs to train
    epochs = 10
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Arrays to put train and validation accuracies and losses in
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    # For every epoch
    for epoch in range(epochs):
        # Print the current epoch
        print("Epoch", epoch)

        # Put the model in train mode
        model.train()

        # Lower the learning rate every 10 epochs
        if (epoch + 1) % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] /= 10
        # Variables necessary to calculate the accuracy
        correct = 0
        total_loss = 0

        # Training loop
        for (imgs, labels) in tqdm(train_dataloader):
            # print(step)
            # Reset the gradients
            optimizer.zero_grad()

            # Retrieve the loss and amount of correct of the batch
            batch_loss, batch_correct = calculate_loss_accuracy(model, imgs, labels, "train")

            # Add loss and correct
            total_loss += batch_loss
            correct += batch_correct

            # Step function
            optimizer.step()

        # Append epoch loss and accuracy for the statistics
        train_accuracy.append(correct / len(train_dataset))
        train_loss.append(total_loss / len(train_dataloader))

        # Put model in validation mode
        model.eval()

        # Reset correct and loss variables
        correct = 0
        total_loss = 0

        # Without gradient changes
        with torch.no_grad():

            # Validation loop
            for step, (imgs, labels) in enumerate(val_dataloader):
                # Retrieve the loss and amount of correct of the batch
                batch_loss, batch_correct = calculate_loss_accuracy(model, imgs, labels)

                # Add loss and correct
                total_loss += batch_loss
                correct += batch_correct

        # Append epoch loss and accuracy for the statistics
        val_accuracy.append(correct / len(val_dataset))
        val_loss.append(total_loss / len(val_dataloader))

    # Save the training and validation accuracies and losses to a file
    with open("models/" + model_name + ".dat", "wb") as f:
        for stat in [train_accuracy, train_loss, val_accuracy, val_loss]:
            print(stat)
            np.save(f, stat)

    # Save the trained model
    torch.save(model.state_dict(), "models/" + model_name + ".pth")

    # Put the model in validation mode
    model.eval()
    # Test dataset
    test_dataset = NetflixDataset(json_file, "test")
    # Test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                             shuffle=True, num_workers=0)

    # Test loop
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
