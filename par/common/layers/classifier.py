from torch import nn


def make_classifier(backbone, feature_size, num_classes):
    classifier = nn.Linear(feature_size, num_classes)
    nn.init.normal_(classifier.weight, std=0.001)
    nn.init.constant_(classifier.bias, 0.0)

    return classifier
