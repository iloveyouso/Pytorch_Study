# Tensor Basics.py

import numpy as np 
import torch

#Initialization of Tensors 
data = [[1,2],[3,4]]
x1 = torch.tensor(data)
np_array = np.array(data)
x2 = torch.from_numpy(np_array)
x_ones = torch.ones_like(x2)
x_rand= torch.rand_like(x2,dtype=torch.float)

shape = (2,3,)
rand_tensor = torch.rand(shape)
zero_tensor = torch.zeros(shape)
ones_tensor = torch.ones(shape)

############################################
#Attributes of a Tensor
tensor = torch.rand(3,4)

print(f"Shape: {tensor.shape}")
print(f"dtype: {tensor.dtype}")
print(f"Device: {tensor.device}")

tensor = tensor.to("cuda")
print(f"Device Again: {tensor.device} \n")

############################################
# OPERATIONS ON TENSORS
# Indexing and Slicing
tensor = torch.ones(4,4)
print(f"First Row: {tensor[0]}")
print(f"First Column: {tensor[:,0]}")
print(f"Last Column: {tensor[:,-1]}")
tensor[:,1]=0
print(tensor)
t1= torch.cat([tensor, tensor, tensor],dim=1)
print(t1)

# Tensor Multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3= torch.rand_like(y1)
torch.matmul(tensor,tensor.T,out=y3)
print(y1)

# Element-wise Product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(z1)

# Aggregation of Tensor
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place Operations (Operations that store the result ino the operand: _)
tensor.add_(5)
print(tensor)
