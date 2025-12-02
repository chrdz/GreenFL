import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class TwoLinearLayers(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, bias=False):
        super(TwoLinearLayers, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_classes = output_dimension

        self.fc1 = nn.Linear(input_dimension, hidden_dimension, bias=bias)
        self.fc2 = nn.Linear(hidden_dimension, output_dimension, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class Cifar10CNN(nn.Module):
#     def __init__(self, num_classes):
#         super(Cifar10CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.fc1 = nn.Linear(64 * 5 * 5, 2048)
#         self.output = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = self.output(x)
#         return x
    
class Cifar10CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)         
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)             
        self.drop  = nn.Dropout(0.3)
        self.out   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.out(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.

    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        self.rnn.flatten_parameters()

        encoded = self.encoder(input_)
        output, (hidden, cell) = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)

        hidden = hidden.permute(1, 0, 2)  # change to (B, N_LAYERS, T)
        cell = cell.permute(1, 0, 2)  # change to (B, N_LAYERS, T)

        return output, (hidden, cell)


def replace_batchnorm_with_groupnorm(model, max_groups=32):
    """
    Replace BatchNorm2d layers in a model with GroupNorm.
    Ensures num_channels is divisible by num_groups.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # Choose num_groups such that it divides num_channels
            num_groups = min(max_groups, num_channels)  # Limit groups to max_groups or num_channels
            while num_channels % num_groups != 0:
                num_groups -= 1  # Decrease groups until divisible

            # Replace with GroupNorm
            setattr(model, name, nn.GroupNorm(num_groups, num_channels))
        elif len(list(module.children())) > 0:  # Recursively check children
            replace_batchnorm_with_groupnorm(module, max_groups)
            

def get_mobilenet(num_classes):
    """
    creates MobileNet model with `num_classes` outputs

    :param num_classes:

    :return:
        model (nn.Module)

    """
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    replace_batchnorm_with_groupnorm(model, max_groups=32)

    return model

def replace_bn_with_gn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            gn = nn.GroupNorm(num_groups=32, num_channels=num_features)  # Adjust num_groups as needed
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module)


def get_resnet18(num_classes):
    """
    creates ResNet18 model with num_classes outputs

    :param num_classes:

    :return:
        model (nn.Module)

    """
    print("Using ResNet-18 model")
    model = models.resnet18(pretrained=True)

    # Replace BatchNorm with GroupNorm
    replace_bn_with_gn(model)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def get_custom_model(num_classes=10):
    model = Cifar10CNN(num_classes=num_classes)
    # replace_batchnorm_with_groupnorm(model, max_groups=32)

    return model