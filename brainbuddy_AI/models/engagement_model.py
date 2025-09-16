import torch.nn as nn

class EngagementModel(nn.Module):
    def __init__(self, cnn_feat_dim=512, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # logit
        )

    def forward(self, cnn_feats):
        # cnn_feats: (B, T, D)
        _, (hn, _) = self.lstm(cnn_feats)   # hn: (1, B, H)
        x = hn.squeeze(0)                   # (B, H)
        return self.fc(x)                   # (B, 1)



