
# Pytorch cuda binding tutorial

This repository contains a cheatsheet code for binding C and CUDA function for pytorch. Read pytorch [doc](https://pytorch.org/tutorials/advanced/cpp_extension.html#) for more detail.


## Usage

Build

```
python build.py install
```

[Use the custom package](./test.py)

```python
# NOTE: import torch to include some shared lib
import torch
from tiny_api_c import hello as hello_c
from tiny_api_cuda import hello as hello_cuda

def main():
    hello_c()
    hello_cuda()

if __name__ == "__main__":
    main()
```


