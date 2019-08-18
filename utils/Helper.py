import time

import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from matplotlib import pyplot as plt
from PIL import Image


def prepare_loader(root_train, root_test, train_transform, test_transforms,
                   batch_size=64, num_workers=0):
    """
        Helper function for prepare data loader by splitting images into test and valid
        :param root_train: train data directory
        :param root_test: train data directory
        :param train_transform: train transform
        :param test_transforms: test transform
        :param batch_size: batch size, default 64
        :param num_workers: num of worker, default 0
        :return: train loader and test loader, classes and classes to idx
        """

    # data set
    train_data = datasets.ImageFolder(root_train, transform=train_transform)
    test_data = datasets.ImageFolder(root_test, transform=test_transforms)

    print("Train size:{}".format(len(train_data)))
    print("Valid size:{}".format(len(test_data)))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers)

    return [train_loader, test_loader, train_data.classes, train_data.class_to_idx]


def prepare_loader_split(root, train_transform, test_transforms,
                         batch_size=64, test_size=0.2, num_workers=0):
    """
    Helper function for prepare data loader by splitting images into test and valid
    :param root: train data directory
    :param train_transform: train transform
    :param test_transforms: test transform
    :param batch_size: batch size, default 64
    :param test_size: test split percentage, default 20%
    :param num_workers: num of worker, default 0
    :return: train loader and test loader
    """

    # data set
    train_data = datasets.ImageFolder(root, transform=train_transform)
    test_data = datasets.ImageFolder(root, transform=test_transforms)

    # obtain training indices that will be used for validation
    num_train = len(train_data)

    # mix data
    # index of num of train
    indices = list(range(num_train))
    # random the index
    np.random.shuffle(indices)
    split = int(np.floor(test_size * num_train))
    # divied into two part
    train_idx, test_idx = indices[split:], indices[:split]

    # define the sampler
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # prepare loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        sampler=test_sampler, num_workers=num_workers)

    print("Train size:{}".format(num_train))
    print("Valid size:{}".format(len(test_data)))

    return [train_loader, test_loader, test_data.classes, test_data.class_to_idx]


def imshow(img, mean=None, std=None):
    """
    Helper function to show image
    :param img: image data
    :param mean: mean, default None
    :param std: std, default NOne
    :return: None
    """

    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])

    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    inp = img.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


def visualize(loader, classes, num_of_image=2, fig_size=(25, 5)):
    """
    Helper function for visualize data
    :param loader: data loader
    :param classes: lis of classes
    :param num_of_image: number of image to show, default 2
    :param fig_size: figure size, default (25,5)
    :return: None
    """
    data_iter = iter(loader)
    images, labels = data_iter.next()

    fig = plt.figure(figsize=fig_size)
    for idx in range(num_of_image):
        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])
        # denormalize first
        img = images[idx] / 2 + 0.5
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))  # transpose
        ax.imshow(img, cmap='gray')
        ax.set_title(classes[labels[idx]])


def load_latest_model(model, name="model.pt"):
    """
    Helper function for Load model
    :param model: current model
    :param name: model name
    :return: loaded model default model.pt
    """
    model.load_state_dict(torch.load(name))
    return model


def save_current_model(model, name='model.pt'):
    """
    Helper function for save model
    :param model: current model
    :param name: model name, default model.pt
    :return: None
    """
    torch.save(model.state_dict(), name)


def save_check_point(model, epoch, train_loader, classes, optimizer, scheduler=None,
                     path=None, name='model.pt'):
    """
    Helper function for save check point. save everything like epoch
    optimizer state and also model state
    :param model: current model
    :param epoch: total epoch
    :param train_loader: train data loader for extract class_to_idx
    :param classes: total classes in your datasets
    :param optimizer: optimizer
    :param scheduler: scheduler if any, default None
    :param path: path for saving model, default None
    :param name: model name, default model.pt
    :return: None
    """

    class_to_idx = train_loader.dataset.class_to_idx

    try:
        classifier = model.classifier
    except AttributeError:
        classifier = model.fc

    checkpoint = {
        'class_to_idx': class_to_idx,
        'class_to_name': classes,
        'epochs': epoch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if path is None:
        d = model
    else:
        d = path + "/" + name

    torch.save(checkpoint, d)
    print(f"Model saved at {d}")


def load_checkpoint(path, model, optimizer_name='adam', lr=0.003, momentum=None,
                    scheduler=None, step=2, gamma=0.1):
    """
    Helper function for load check point
    :param path: path of saved model
    :param model: current model
    :param optimizer_name: optimizer name, default Adam
    :param lr: learning rate, used to create optimizer
    :param momentum: momentum if you use SGD optimizer, default None
    :param scheduler: StepLR scheduler, if you want to create scheduler, default None
    :param step: Period of learning rate decay, default 2
    :param gamma: Multiplicative factor of learning rate decay. default: 0.1
    :return: model, optimizer and scheduler(if scheduler is not None)
    """

    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Load in checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")

    # Extract classifier
    classifier = checkpoint['classifier']
    # set classifier
    try:
        check = model.classifier
    except AttributeError:
        check = False

    if check is not False:
        model.classifier = classifier
    else:
        model.fc = classifier

    # Extract others
    model.cat_to_name = checkpoint['class_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        if momentum is not None:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')
    print(f'Model has been trained for {model.epochs} epochs.')

    if scheduler is not None:
        return [model, optimizer, scheduler]
    else:
        return [model, optimizer]


def freeze_parameters(model):
    """
    Helper function for freeze parameter
    :param model: current model
    :return: new model with freeze parameters
    """
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze(model):
    """
    Helper function for unfreeze parameters
    :param model: current model
    :return: new model with unfreeze parameters
    """
    for param in model.parameters():
        param.requires_grad = True

    return model


def unfreeze_last_layer(model, last_layer_name='classifier'):
    """
    Helper function for unfreeze parameters
    :param model: current model
    :param last_layer_name: last layer name of the model
    :return: new model with unfreeze parameters
    """

    if last_layer_name.lower() == 'classifier':
        for param in model.classifier.parameters():
            param.requires_grad = True

    if last_layer_name.lower() == 'fc':
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def train(model, train_loader, test_loader,
          epochs, optimizer, criterion, scheduler=None,
          name="model.pt", path=None):
    """
    Helper function for train model
    :param model: current model
    :param train_loader: train data loader
    :param test_loader: test data loader
    :param epochs: number of epoch
    :param optimizer: optimizer
    :param criterion: loss function
    :param scheduler: scheduler, default None
    :param name: model name, default model.pt
    :param path: model saved location, default None
    :return: model, list of train loss and test loss
    """

    # compare overfitted
    train_loss_data, valid_loss_data = [], []
    # check for validation loss
    valid_loss_min = np.Inf
    # calculate time
    since = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        total = 0
        correct = 0
        e_since = time.time()

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training

        for images, labels in train_loader:
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            log_ps = model(images)
            # calculate the loss
            loss = criterion(log_ps, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        print("\t\tGoing for validation")
        model.eval()  # prep model for evaluation
        for data, target in test_loader:
            # Move input and label tensors to the default device
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss_p = criterion(output, target)
            # update running validation loss
            valid_loss += loss_p.item() * data.size(0)
            # calculate accuracy
            proba = torch.exp(output)
            top_p, top_class = proba.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(test_loader.dataset)

        # calculate train loss and running loss
        train_loss_data.append(train_loss * 100)
        valid_loss_data.append(valid_loss * 100)

        print("\tTrain loss:{:.6f}..".format(train_loss),
              "\tValid Loss:{:.6f}..".format(valid_loss),
              "\tAccuracy: {:.4f}".format(correct / total * 100))

        if scheduler is not None:
            scheduler.step()  # step up scheduler

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), name)
            valid_loss_min = valid_loss
            # save to google drive
            if path is not None:
                torch.save(model.state_dict(), path)

        # Time take for one epoch
        time_elapsed = time.time() - e_since
        print('\tEpoch:{} completed in {:.0f}m {:.0f}s'.format(
            epoch + 1, time_elapsed // 60, time_elapsed % 60))

    # compare total time
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model
    model = load_latest_model(model, name)

    # return the model
    return [model, train_loss_data, valid_loss_data]


def train_faster_log(model, train_loader, test_loader,
                     epochs, optimizer, criterion, scheduler=None, print_every=5):
    """
    Helper function for train model. This model print log after a certain interval
    in every epoch.
    :param model: current model
    :param train_loader: train data loader
    :param test_loader: test data loader
    :param epochs: number of epoch
    :param optimizer: optimizer
    :param criterion: loss function
    :param scheduler scheduler, default None
    :param print_every: print log interval
    :return:
    """

    steps = 0
    running_loss = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"\tTrain loss: {running_loss / print_every:.3f}.. "
                      f"\tTest loss: {test_loss / len(test_loader):.3f}.. "
                      f"\tTest accuracy: {accuracy / len(test_loader):.3f}")
                running_loss = 0
                model.train()

    if scheduler is not None:
        scheduler.step()

    return model


def check_overfitted(train_loss, test_loss):
    """
    Helper function for check over fitting
    :param train_loss: list of train loss
    :param test_loss: list of test loss
    :return: None
    """
    plt.plot(train_loss, label="Training loss")
    plt.plot(test_loss, label="validation loss")
    plt.legend(frameon=False)


def test_per_class(model, test_loader, criterion, classes):
    """
    Helper function for testing per class
    :param model: current model
    :param test_loader: test loader
    :param criterion: loss function
    :param classes: list of classes
    :return: None
    """

    total_class = len(classes)

    test_loss = 0.0
    class_correct = list(0. for i in range(total_class))
    class_total = list(0. for i in range(total_class))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()  # prep model for evaluation

    for data, target in test_loader:
        # Move input and label tensors to the default device
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target) - 1):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(total_class):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


def test(model, loader, criterion=None):
    """
    Helper function for test result. This function use torch.mean()
    :param model: current result
    :param loader: test data loader
    :param criterion: loss function to track loss, default None
    :return: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        model.eval()

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            if criterion is not None:
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    if criterion is not None:
        print("Test Loss:{:.6f}".format(test_loss),
              "\nAccuracy: {:.4f}".format(accuracy / len(loader) * 100))
    else:
        print("Accuracy: {:.4f}".format(accuracy / len(loader) * 100))


def test_with_single_image(model, file, transform, classes):

    """

    :param model:
    :param file:
    :param transform:
    :param classes:
    :return:
    """

    file = Image.open(file).convert('RGB')

    img = transform(file).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        out = model(img.to(device))
        ps = torch.exp(out)
        top_p, top_class = ps.topk(1, dim=1)
        value = top_class.item()
        print("Value:", value)
        print(classes[value])
        plt.imshow(np.array(file))
        plt.show()
