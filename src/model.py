import torch


class MaleFemaleModel(torch.nn.Module):
    def __init__(
            self,
            input_size=80,
            time_size=100,
            output_size=1,
            conv2d_filters=32,
    ):
        super(MaleFemaleModel, self).__init__()

        self.conv_in = torch.nn.Sequential(
            torch.nn.Conv2d(1, conv2d_filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv2d_filters, conv2d_filters, kernel_size=(3, 3), stride=(2, 2),
                            padding=(1, 1)),
            torch.nn.ReLU(),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Linear(conv2d_filters * (input_size // 4) * (time_size // 4 + 1), 100),
            torch.nn.Linear(100, output_size),
        )

    def forward(self, x):
        x = self.conv_in(x)
        b, c, t, f = x.size()
        x = self.conv_out(x.contiguous().view(b, t * c * f))
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    x = torch.rand([3, 1, 64, 201])
    model = MaleFemaleModel(input_size=64, time_size=201)
    output = model(x)
    print(output.shape)
