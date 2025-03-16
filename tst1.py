

import torch

from thop import profile

from Net.DnCNN import DNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DNNs = DNN().to(device)
DNNs.load_state_dict(torch.load("./DnCNN/SaveModel/25900_0.0008.pth"))
DNNs.eval()


input = torch.randn(1, 3, 8, 8).cuda()
flops, params = profile(DNNs, inputs=(input,))

print(flops, params)




# 28458221568.0 559363.0




