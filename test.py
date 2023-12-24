# NOTE: import torch to include some shared lib for custom function
import torch
from tiny_api_c import hello as hello_c
from tiny_api_c import vector_add as vector_add_c
from tiny_api_cuda import hello as hello_cuda
from tiny_api_cuda import vector_add as vector_add_cuda

def test_api_bind():
    hello_c()
    hello_cuda()

def test_data_bind():
    a = torch.tensor([1., 2., 3., 4., 5.], device="cuda")
    b = torch.tensor([5., 4., 3., 2., 1.], device="cuda")
    ref_out = a + b
    cpp_out = vector_add_c(a, b)
    cuda_out = vector_add_cuda(a, b)
    print(ref_out)
    print(cpp_out)
    print(cuda_out)

def main():
    test_api_bind()
    test_data_bind()

if __name__ == "__main__":
    main()
