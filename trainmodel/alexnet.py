import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    # def __init__(self, num_classes=1000):
    #     super(AlexNet, self).__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #     )
    #     self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    #     self.classifier = nn.Sequential(
    #         nn.Dropout(),
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),
    #     )
    #     self.fc = nn.Linear(4096, num_classes)

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # 使用3x3卷积
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2池化
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 再次池化
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 自适应池化，输出为6x6
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.pool(out)  # 第一次池化
        out = self.conv_block2(out)
        out = self.pool(out)  # 第二次池化
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.pool2(out)  # 第三次池化
        out = self.avgpool(out)  # 自适应池化
        out = out.reshape(out.size(0), -1)  # 展平
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)

        new_dict = {}
        for k, v in state_dict.items():
            if 'classifier.6' not in k:
                new_dict[k] = v
            else:
                new_dict[k.replace('classifier.6', 'fc')] = v

        model.load_state_dict(new_dict)
    return model