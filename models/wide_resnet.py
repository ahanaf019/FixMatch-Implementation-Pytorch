import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph
from torchinfo import summary

class BasicBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride, dropout_rate=0.0):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.kernel_size = (3, 3)
        self.stride = stride
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.in_filters),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.in_filters,
                out_channels=self.out_filters,
                kernel_size=(self.kernel_size),
                stride=self.stride,
                padding=1
            ),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm2d(self.out_filters),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(
                in_channels=self.out_filters,
                out_channels=self.out_filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1
            ),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_filters != out_filters:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(self.in_filters),
                nn.Conv2d(
                    in_channels=self.in_filters,
                    out_channels=self.out_filters,
                    kernel_size=(1, 1),
                    stride=stride,
                ),
            )
    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, in_channels, base_filters, k, N, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.base_filters = base_filters
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.base_filters,
                kernel_size=(3,3),
                padding=1
            )
        )

        # =========================================================
        self.conv2 = nn.Sequential(
            BasicBlock(
                in_filters=self.base_filters,
                out_filters=self.base_filters * k,
                stride=1, dropout_rate=dropout_rate,
            ),
            *[
            BasicBlock(
                in_filters=self.base_filters * k,
                out_filters=self.base_filters * k,
                stride=1, dropout_rate=dropout_rate,
            ) for _ in range(N - 1)
        ])

        # =========================================================
        self.conv3 = nn.Sequential(
            BasicBlock(
                in_filters=self.base_filters * k,
                out_filters=self.base_filters * 2 * k,
                stride=2, dropout_rate=dropout_rate,
            ),
            *[
            BasicBlock(
                in_filters=self.base_filters * 2 * k,
                out_filters=self.base_filters * 2 * k,
                stride=1, dropout_rate=dropout_rate,
            ) for _ in range(N - 1)
        ])

        # =========================================================
        self.conv4 = nn.Sequential(
            BasicBlock(
                in_filters=self.base_filters * 2 * k,
                out_filters=self.base_filters * 4 * k,
                stride=2, dropout_rate=dropout_rate,
            ),
            *[
            BasicBlock(
                in_filters=self.base_filters * 4 * k,
                out_filters=self.base_filters * 4 * k,
                stride=1, dropout_rate=dropout_rate,
            ) for _ in range(N - 1)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_filters * 4 * k, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x




if __name__ == '__main__':
    model = WideResNet(in_channels=3, base_filters=16, k=2, N=6).to('cuda')
    
    conv_counter = 0
    for (name, module) in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_counter += 1
    print(conv_counter)
    # print(model)
    # summary(model, input_size=[1, 3, 32, 32])

    # y = model(torch.rand(size=[1, 32, 112, 112]).to('cuda'))
    # graph = draw_graph(model, input_size=(1, 32, 112, 112), expand_nested=True)
    # graph.visual_graph.render("resnet18_arch", format="png")