import torch
from models import ESPCNSeq
model = ESPCNSeq(img_channels=3,upscale_factor=4)

model.load_state_dict(torch.load("./models/espcnseq/espcnseq_500_reduced-channels.pt"))
#model.load_state_dict(torch.load("./models/espcnseq/espcnseq_1.pt")) 

#model = torch.jit.load("./models/espcnseq/espcnseq_1.pt")

# print(model)

model.eval() 

random_input = torch.randn(1, 3, 200, 200)

# print(model(random_input))

traced_script_module = torch.jit.trace(model, random_input)

traced_script_module.save("./models/espcnseq/espcnseq-reduced-channels-jit.pt")