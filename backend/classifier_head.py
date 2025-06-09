import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class classifier(nn.Module):
    """
    Joint LSTM + linear-layer classification head.
    Processes a sequence of DINOv2 embeddings (one 768-dim vector per frame)
    of length `seq_len` and produces a classification for the center frame.
    """

    def __init__(
        self,
        in_features,   # dimensionality of each DINOv2 embedding (e.g., 768)
        out_features,  # number of behavior classes
        seq_len=31     # length of the input sequence (must be odd)
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Linear “motion-agnostic” branch: directly projects each embedding to class logits
        self.lin1 = nn.Linear(in_features, out_features)

        # Linear “embedding → LSTM input” projection
        self.lin0 = nn.Linear(in_features, 256)

        # Linear projection after LSTM output (bidirectional hidden size is 2×64 = 128)
        self.lin2 = nn.Linear(128, out_features)
        
        # BatchNorm across the feature dimension, applied to each frame’s embedding
        # BatchNorm1d normalizes over [batch, features], so we permute accordingly
        self.batch_norm = nn.BatchNorm1d(in_features)

        # Single-layer bidirectional LSTM:
        # input_size = 256 (from lin0), hidden_size = 64, num_layers = 1
        # bidirectional=True doubles the hidden dimension → outputs of size 128
        self.lstm = nn.LSTM(256, 64, num_layers=1, batch_first=True, bidirectional=True)

        # Half sequence length (number of frames on either side of center)
        self.hsl = seq_len // 2
        # “sub-window” size for averaging around center: ±sw frames
        self.sw = 5


    def forward_linear(self, x):
        """
        Linear branch: for each frame embedding in the sequence, project
        to logits, then average over a small centered window of frames.
        Input x shape: (batch_size, seq_len, in_features)
        Returns: (batch_size, out_features)
        """
        # Project each frame’s embedding to class logits → shape: (B, seq_len, out_features)
        x_proj = self.lin1(x)

        # Slice out the centered window of size (2*sw + 1) around index hsl
        # We take frames [hsl - sw : hsl + sw + 1], then average over that window
        windowed_logits = x_proj[:, self.hsl - self.sw : self.hsl + self.sw + 1, :]

        # Mean over the window dimension → final linear logits per example
        return windowed_logits.mean(dim=1)


    def forward_lstm(self, x):
            """
            LSTM branch: run the projected sequence through LSTM,
            then take a small centered window of LSTM hidden outputs,
            average them, and project to class logits.
            Input x shape: (batch_size, seq_len, 256)
            Returns:
              - class logits (batch_size, out_features)
              - averaged LSTM features from the centered window (batch_size, 128)
            """
            # Run through LSTM → outputs of shape (batch_size, seq_len, 128)
            lstm_out, _ = self.lstm(x)

            # Extract the centered window around hsl
            center_window_raw_lstm = lstm_out[:, self.hsl - self.sw : self.hsl + self.sw + 1, :] # (B, 2*sw+1, 128)
            
            # Average over that window dimension
            avg_latent = center_window_raw_lstm.mean(dim=1) # Averaged latent features shape: (batch_size, 128)

            # Project these averaged features to class logits → shape: (batch_size, out_features)
            logits = self.lin2(avg_latent)

            # Return both the projected logits and the AVERAGED latent window features
            return logits, avg_latent


    def forward(self, x):
        """
        Training-time forward pass with noise injection (“dropout” of
        random embedding dimensions).
        Input x shape: (batch_size, seq_len, in_features)
        Returns:
          - lstm_logits: (batch_size, out_features)
          - linear_logits: (batch_size, out_features)
          - rawm: the averaged LSTM window output (batch_size, 128)
        """
        # x permuted to (batch_size, in_features, seq_len) for BatchNorm1d
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Randomly pick between 64 and 256 feature channels to “drop” by replacing them with noise
        # This injects noise into embeddings as a form of regularization
        amount = random.randint(64, 256)
        rand_inds = torch.randperm(x.size(2))[:amount]  # choose random channel indices
        x[:, :, rand_inds] = torch.randn_like(x[:, :, rand_inds]).to(x.device)

        # Get the linear branch logits (averaged over centered window)
        linear_logits = self.forward_linear(x)

        # Project the noised embeddings to 256-dim for LSTM input
        x_lstm = self.lin0(x)

        # Subtract the mean over the sequence dimension so that the LSTM sees zero-mean inputs
        x_lstm = x_lstm - x_lstm.mean(dim=1, keepdim=True)

        # Run LSTM branch
        lstm_logits, rawm = self.forward_lstm(x_lstm)

        # Return the two logits outputs and the averaged latent features
        return lstm_logits, linear_logits, rawm


    def forward_nodrop(self, x):
        """
        Inference-time forward pass: no noise injection.
        Input x shape: (batch_size, seq_len, in_features)
        Returns:
          - combined logits: sum of LSTM branch and linear branch → (batch_size, out_features)
        """
        # BatchNorm (zero-mean, unit-variance) on embeddings
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Linear branch logits
        linear_logits = self.forward_linear(x)

        # Project embeddings to 256-dim for LSTM input
        x_lstm = self.lin0(x)

        # Zero-mean across the sequence dimension
        x_lstm = x_lstm - x_lstm.mean(dim=1, keepdim=True)

        # LSTM branch
        lstm_logits, _ = self.forward_lstm(x_lstm) # Second output (avg_latent) is not used here

        # Sum both branches to get final logits
        return lstm_logits + linear_logits