import torch.nn as nn
import torch
# **Define ResNet-1D Architecture**
# BasicBlock for ResNet1D
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.3):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)  # Conv1
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)  # Conv2
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out

# ResNet1D Architecture
class ResNet1D(nn.Module):
    def __init__(self, block, layers,embedding_dim, in_channels=64, num_classes=1000, dropout_prob=0.3):
        super(ResNet1D, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(embedding_dim, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1, dropout_prob=0.3):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_prob))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # (batch, in_channels, L/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.maxpool(x)  # (batch, in_channels, L/4)

        x = self.layer1(x)  # (batch, 64, L/4)
        x = self.layer2(x)  # (batch, 128, L/8)
        x = self.layer3(x)  # (batch, 256, L/16)
        x = self.layer4(x)  # (batch, 512, L/32)

        x = self.avgpool(x)  # (batch, 512, 1)
        x = torch.flatten(x, 1)  # (batch, 512)
        x = self.fc(x)  # (batch, num_classes)

        return x

# Define the ResNet1DEmbedding model
class ResNet1DEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, dropout_prob=0.3):
        super(ResNet1DEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], embedding_dim,in_channels=64, num_classes=num_classes,dropout_prob=dropout_prob)
        # ** Dropout **
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  # (batch, max_length, embedding_dim)
        x = x.permute(0, 2, 1)  # Rearrange to (batch, embedding_dim, sequence_length)
        x = self.resnet(x)  # Output from ResNet1D: (batch, num_classes)
        # Apply additional fully connected layers if needed
        # If you want to keep the ResNet's fully connected layer, you can skip this
        # x = self.fc_layers(x)
        return x
