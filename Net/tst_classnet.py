
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # self.lastOut = nn.Linear(32, 3)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     )
        self.value = nn.Conv2d(128, 1, 1)
        self.policy = nn.Conv2d(128, 5, 1)
        # arch_util.initialize_weights([self.CondNet], 0.1)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
    def forward(self, x):
        conv = self.CondNet(x)
        value = self.value(conv)
        policy = self.policy(conv)
        value = self.upsample(value)
        policy = self.upsample(policy)
        print(value.size())
        print(policy.size())
        print("TTTTTT")
        return value
import time
# test
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
x4 = torch.randn(1, 3, 32, 32)
x1s = torch.cat((x1, x2, x3, x4), 0)
model = Classifier()
out = model(x1)
t1 = time.time()
out = model(x1s)
# out2 = model(x2)
# out3 = model(x3)
# out4 = model(x4)
t2 = time.time()
t3 = time.time()
x = torch.randn(1, 3, 64, 64)
out5 = model(x)
t4 = time.time()
print(t2-t1)
print(t4-t3)
# print(out)