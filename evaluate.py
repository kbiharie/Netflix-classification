import torchvision
import torch
from dataset import NetflixDataset

if __name__ == "__main__":

    torch.manual_seed(0)

    # Test dataset
    test_dataset = NetflixDataset("../dataset/filenames_animated.json", "test")
    # Test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                                  shuffle=False, num_workers=0)
    n_classes = test_dataset.n_classes

    # Load a pretrained resnet50 model
    model = torchvision.models.resnet50(pretrained=True)
    # Edit the last layer to the amount of classes our dataset has
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=n_classes, bias=True)
    # Put the model on the gpu
    model.cuda()

    model.load_state_dict(torch.load("models/model_animated_10.pth"))
    model.eval()
    accuracies = {}

    for i in range(n_classes):
        accuracies[i] = 0
    with torch.no_grad():
        for step, (imgs, labels) in enumerate(test_dataloader):
            print(step)

            imgs = imgs.cuda()

            # Predicted labels of the images
            preds = model(imgs)

            # Find label prediction
            _, preds = torch.max(preds, 1)
            preds = preds.cpu()

            correct = [labels[j] == preds[j] for j in range(len(labels))]

            for i in range(len(correct)):
                accuracies[labels[i].item()] += int(correct[i])
            break
    print(accuracies)

    accuracies_show = {}
    for show_name in test_dataset.shows:
        accuracies_show[show_name] = accuracies[test_dataset.shows[show_name]]

    for show_name in accuracies_show:
        print(show_name, accuracies_show[show_name])
