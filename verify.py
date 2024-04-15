import torch
import math

# d_model = 64
# print(torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
#              device=torch.device('cpu')).shape)


result = torch.tensor([[[1.], [2.], [3.]]])
print(result.shape)

print(result)
result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
print(result)