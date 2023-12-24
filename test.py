# NOTE: import torch to include some shared lib
import torch
from tiny_api_c import hello as hello_c
from tiny_api_cuda import hello as hello_cuda

def main():
    hello_c()
    hello_cuda()

if __name__ == "__main__":
    main()
