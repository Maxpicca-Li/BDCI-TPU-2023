import torch
from ninasr import NinaSR

if __name__ == "__main__":
    model = torch.jit.load("./models/ninasr_jit.pt")

    ninasr_model = NinaSR(scale=4, n_resblocks=10, n_feats=64, expansion=2.0)

    for key, value in model.state_dict().items():
        print(key, value.shape)

    # random_input = torch.randn(1, 3, 200, 200)
    # print(model(random_input))