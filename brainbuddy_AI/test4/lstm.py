import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: [B, T, 2H]
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)  # [B, T, 1]
        weighted = lstm_out * attn_weights  # [B, T, 2H]
        context = weighted.sum(dim=1)       # [B, 2H]
        return context, attn_weights

class BiLSTMAttnModel(nn.Module):
    def __init__(self, input_size, hidden_size, dynamic_size, num_classes=5, num_layers=1, dropout=0.3):
        super().__init__()

        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.attn = AttentionLayer(hidden_size)

        self.fc_dynamic = nn.Sequential(
            nn.Linear(dynamic_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_dyn):
        # x_seq: [B, T, D], x_dyn: [B, D_dyn]
        lstm_out, _ = self.bilstm(x_seq)            # lstm_out: [B, T, 2H]
        h_seq, attn_weights = self.attn(lstm_out)   # h_seq: [B, 2H]

        h_dyn = self.fc_dynamic(x_dyn)              # [B, 16]
        h = torch.cat([h_seq, h_dyn], dim=1)        # [B, 2H + 16]

        out = self.fc_fusion(h)                     # [B, num_classes]
        return out
