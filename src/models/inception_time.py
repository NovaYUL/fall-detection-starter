import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=(9,19,39), use_bn=True):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            in_conv = bottleneck_channels
        else:
            in_conv = in_channels

        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_conv, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1)) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_bn = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(x_bn) for conv in self.conv_list]
        pool_out = self.conv_pool(self.maxpool(x))
        x = torch.cat(conv_outs + [pool_out], dim=1)
        x = self.bn(x)
        return self.relu(x)

class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_blocks=6, out_channels=32, bottleneck_channels=32, use_residual=True, n_classes=1, dropout=0.2):
        super().__init__()
        blocks = []
        self.use_residual = use_residual
        self.shortcut_layers = nn.ModuleList()
        ch = in_channels
        for i in range(num_blocks):
            block = InceptionBlock(
                in_channels=ch,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels
            )
            blocks.append(block)
            out_ch = out_channels * 4
            if use_residual and (i % 3 == 2):
                self.shortcut_layers.append(nn.Sequential(
                    nn.Conv1d(ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch)
                ))
                ch = out_ch
            else:
                self.shortcut_layers.append(None)
                ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.final_bn = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            out = block(x)
            if self.use_residual and (i % 3 == 2):
                sc = self.shortcut_layers[i](x)
                out = F.relu(out + sc)
            x = out
        x = self.final_bn(x)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)
        return logits
