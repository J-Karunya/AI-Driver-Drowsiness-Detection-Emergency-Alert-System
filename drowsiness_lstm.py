import torch
import torch.nn as nn


class DrowsinessLSTM(nn.Module):

    def __init__(
        self,
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.30,
        num_classes=2
    ):

        super(DrowsinessLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Because the LSTM is bidirectional:
        # output size = hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                32
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                num_classes
            )
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length, 9)

        out, _ = self.lstm(x)

        # Take the last timestep
        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        return out


def get_model(
    model_path=None,
    device="cpu"
):

    model = DrowsinessLSTM(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.30,
        num_classes=2
    )

    if model_path:

        try:

            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )

            model.load_state_dict(
                state_dict
            )

            print(
                f"✓ Loaded trained LSTM from "
                f"{model_path}"
            )

        except Exception as e:

            print(
                f"ERROR: Could not load model "
                f"from {model_path}"
            )

            print(
                f"Reason: {e}"
            )

            raise RuntimeError(
                "The LSTM model architecture does not "
                "match the saved model weights."
            )

    model.to(device)
    model.eval()

    return model