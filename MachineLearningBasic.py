import torch

# SCALAR
scalar=torch.tensor(7) #scalar input=7
print("\n",scalar)
print(scalar.ndim) #no. of dimention does scalar have
print(scalar.item()) #Get tensor back as python int

# VECTOR
vector=torch.tensor([6,9]) #vector input=7x+7y
print("\n",vector)
print(vector.ndim)
print(vector[1])
print(vector.shape)

# MATRIX
matrix=torch.tensor([[7,8],
                     [9,10]]) #matrix input
print("\n",matrix)
print(matrix.ndim)
print(matrix[1])
print(matrix.shape) #tensor([[rows, column]])

# TENSOR
tensor=torch.tensor([[[1,2,3],
                      [3,6,9],
                      [10,11,12]]]) # tensor input
print("\n",tensor)
print(tensor.ndim)
print(tensor.shape)

# RANDOM TENSOR
random_tensor=torch.rand(size=(3,4)) #rows=3 columns=4
print("\n",random_tensor)
print(random_tensor.ndim)

# CREATING A RANDOM TENSOR WITH SIMILAR SHAPE TO AN IMAGE TENSOR
random_image_size_tensor=torch.rand(size=(224,224,3)) #size=(height, width, colorChannels) where colorChannels are (R, G, B)
print("\n",random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)

# zeros and ones
zeros=torch.zeros(size=(3,4))
print("\n",zeros)
print(zeros*random_tensor)
ones=torch.ones(size=(3,4))
print(ones)
print(ones.dtype)

# Creating range of tensors
zero_to_nine=torch.arange(0,10)
print("\n",zero_to_nine)
one_to_ten=torch.arange(1,11)
print(one_to_ten)
one_to_thousand_with_50steps=torch.arange(start=1, end=1000, step=50)
print(one_to_thousand_with_50steps)

# Creating tensor-like
one_to_ten=torch.zeros_like(input=one_to_ten)
print(one_to_ten)

# **NOTE:** tensor datatypes is one of the 3 big errors you'll run into with pytorch & deep learning 
# 1. tensor not right datatype
# 2. tensor not right shape
# 3. tensor not on the right device

float32_tensor=torch.tensor([3.0, 6.0, 9.0],
                            dtype=None, #what data type is the tensor (default=float32)
                            device=None, #what device is your tensor on (default=None=cpu)
                            requires_grad=False) #whether or not to track gradiants with this tensor operation
print("\n",float32_tensor)
print(float32_tensor.dtype)
float16_tensor=torch.tensor([2.0, 4.0, 6.0],
                            dtype=torch.float16,
                            device="cuda",
                            requires_grad=False)
print(float16_tensor)
print(float16_tensor.dtype)

# Converting float32 to float16 tensor
float16_tensor=float32_tensor.type(torch.float16)
print("\n",float16_tensor)
print(float16_tensor*float32_tensor)

# let
int_32_tensor=torch.tensor([2,3,4],
                           dtype=torch.int32)
print(int_32_tensor)
print(float32_tensor*int_32_tensor)

# 1. tensor not right datatype - to get datatype from tensor, can us tensor.dtype
# 2. tensor not right shape - to get shape from a tensor, can use tensor.shape
# 3. tensor not on the right device - to get device from a tensor, can use tensor.device
some_tensor=torch.rand(3, 4)
print("\n",some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device of tensor: {some_tensor.device}")

# Manipulating tensors (tensor operations)
# It includes:
# Addition
# Subtraction
# Multiplication (element-wise)
# Division
# Matrix multiplication
tensor=torch.tensor([1,2,3])
print("\n",tensor+10)
print(tensor-10) #Slow
print(tensor*10)
print(tensor/10)
#OR
print(torch.add(tensor, 10))
print(torch.sub(tensor, 10)) #Fast
print(torch.mul(tensor, 10))
print(torch.div(tensor, 10))

# Matrix multiplication
# 2 main ways of performing multiplication in neural networks and deep learning:
# 1. Element wise multiplication
print("\n",tensor, "*", tensor)
print(f"Equals: {tensor*tensor}")
# 2. Matrix multiplication (dot product)
print(torch.matmul(tensor, tensor))
#OR
print(tensor @ tensor)

# There are two main rules that performing matrix multiplication needs to satisfy:
# 1. Inner Dimension Should Match
# torch.Size([3, 2]) @ torch.Size([3, 4]) won't work
# torch.Size([3, 2]) @ torch.Size([2, 5]) will work
# torch.Size([2, 3]) @ torch.Size([3, 1]) will work
matrixmul=torch.matmul(torch.rand(2,3), torch.rand(3,6))
print(matrixmul)
# 2. The resulting matrix has the shape of the Outer Dimension
print(matrix.shape)
# In matrix A*B != B*A
# To Fix our tensor shape issues, we can manipulate the shape of one of our tensors using a TRANSPOSE
# Transpose switches the axis or dimension of the matrix
A=torch.rand(2,3)
B=torch.rand(2,3)
#print(A @ B) will not work cuz the inner dimension are not same
#we have to transpose any one of them
print("\n",A)
print(A.T)
print(B)
print(B.T)
print(A.T @ B)
print(A @ B.T)

# Finding Min, Max, Mean, Sum, etc (tensor aggregations)
tensorA=torch.arange(10, 110, 10) #(start, end, steps)
print(tensorA)
print("\n",f"Minimum: {torch.min(tensorA)}")
print(f"Minimum: {tensorA.min()}")
print(f"Maximum: {torch.max(tensorA)}")
print(f"Maximum: {tensorA.max()}")
print(f"Mean: {torch.mean(tensorA.type(torch.float32))}") #mean can calculated only for floating numbers or complex one
print(f"Mean: {tensorA.type(torch.float32).mean()}")
print(f"Sum: {torch.sum(tensorA)}")
print(f"Sum: {tensorA.sum()}")

#positional min, man and mean. Shows the position of the min or max value in the tensor
print("\n",tensorA.argmin())
print(tensorA.argmax())
print(torch.argmin(tensorA))
print(torch.argmax(tensorA))
print(tensorA[0])
print(tensorA[9])

# Reshaping, stacking, squeezing and unsqueezing tensors:
# Reshaping - reshapes the input tensor to a defined shape
x=torch.arange(1, 11)
print("\n",x, x.shape)
x_reshaped=x.reshape(1,10) # only works when (1*10 = actual x size)
print(x_reshaped, x_reshaped.shape)

# View - return a view of an input tensor of certain shape but keep the same memory as the original tensor
z=x.view(1,10)
print("\n",z,z.shape) # changing z changes x(bcuz a view of tensor shares the same memory as the original input)
z[:, 0]=5
print(z,x)

# Stacking - combine multiple tensors on top of eachother (vstack) or side by side (hstack)
x_stacked=torch.stack([x,x,x,x],dim=0) #hstack
print("\n",x_stacked)
x_stacked=torch.stack([x,x,x,x],dim=1) #vstack
print("\n",x_stacked)

# Squeezwe - removes all one dimension from a tensor
print("\n"f"Previous tensor and its shape: ",x_reshaped,x_reshaped.shape)
x_squeezed=x_reshaped.squeeze()
print("\n"f"After removing extra dimensions from x_reshape: ",x_squeezed,x_squeezed.shape)

# unsqueeze - adds a one dimension to a target tensor
print("\n"f"Previous tensor and its shape: ",x_squeezed,x_squeezed.shape)
x_unsqueezed=x_squeezed.unsqueeze(dim=0) #need the dimension input to unsqeeze the tenor (adding the dimension)
print("\n"f"After unsqueezing the squeesed tensor be like: ",x_unsqueezed,x_unsqueezed.shape)
x_unsqueezed=x_squeezed.unsqueeze(dim=1)
print("\n"f"After unsqueezing the squeesed tensor be like: ",x_unsqueezed,x_unsqueezed.shape)

# Permute - return a view of the input with dimension permuted (swapped) in a certain way
x_og=torch.rand(size=(3,4,2)) #[height, width, colorchannel] 
print("\n",x_og,x_og.shape)
x_permuted=x_og.permute(2,0,1) #shifting-->[height-->2, width-->0, colorchannel-->1]
print("\n",x_permuted,x_permuted.shape)