# -*- coding: utf-8 -*-
import torch
from torch import nn


def get_ocr_model(config, pretrained=True):
    backbone = get_resnet34_backbone(pretrained=pretrained)
    return RecognitionModel(backbone, **config['model']['params'])


def get_resnet34_backbone(pretrained=True):
    m = torch.hub.load('pytorch/vision:v0.7.0', 'resnet34', pretrained=pretrained)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class RecognitionModel(nn.Module):

    def __init__(self, feature_extractor, time_feature_count, lstm_hidden, lstm_len, n_class):
        super(RecognitionModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, n_class)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        return x
