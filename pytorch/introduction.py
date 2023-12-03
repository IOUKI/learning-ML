# pytorch introduction

import torch

# tensor: 可以存放很多數字的資料型態
x = torch.tensor(87)
print(x)
print(x.ndim)

x = torch.tensor([1, 2, 3])
print(x)
print(x.ndim)

x = torch.tensor([[1, 2, 3], [5, 6, 7]])
print(x)
print(x.ndim)