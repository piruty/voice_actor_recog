import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable


from models.NN import Network
from models.DatasetLoader import make_data_set


train_loader, test_loader, classes = make_data_set('./files/data.txt')

print('start train')
model = Network()

loss_fn = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

def testAccurary():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            (images, labels) = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy = (100 * accuracy / total)
    return accuracy


def train(num_epochs):
    best_accuracy = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            # print(labels)
            loss = loss_fn(outputs, labels.float())
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

# we want to save the model if the accuracy is the best
model.save()

import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[torch.argmax(labels[j])]
                               for j in range(10)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted:   ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))

# Let's build our model
train(5)
print('Finished Training')

# Test which classes performed well
# testModelAccuracy()

# Let's load the model we just created and test the accuracy per label
model = Network()
path = "./files/model.pth"
model.load_state_dict(torch.load(path))

# Test with batch of images
testBatch()
