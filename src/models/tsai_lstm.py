import torch.nn as nn


class TsaiLSTMClassifier(nn.Module):
    """
    A thin wrapper around timeseriesAI/tsai's LSTMPlus.
    Input: (B, C, L) where C=features, L=seq_len
    Output: logits (B,)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        try:
            from tsai.all import LSTMPlus  # type: ignore
        except Exception as e:
            raise ImportError("tsai is not installed. Please run: pip install tsai") from e
        self.net = LSTMPlus(c_in=input_dim, c_out=1)

    def forward(self, x):
        y = self.net(x)  # (B, 1)
        return y.squeeze(-1)
