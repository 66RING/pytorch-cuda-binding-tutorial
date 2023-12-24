
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

## Roadmap

- [x] api binding
- [x] torch data binding

### API binding

- Use `PYBIND11_MODULE` to bind API

```cpp
#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")
  m.def("hello", &hello, "Prints hello world from cuda file");
  m.def("vector_add", &vector_add, "Add two vectors on cuda");
}
```

### data binding

- `torch::Tensor` as tensor type
- `tensor.data()` to get the underlaying pointer
- `AT_DISPATCH_FLOATING_TYPES()` to determing data type. e.g. `AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ([&]{your_kernel_call<scalar_t>();}))`
    * the typename `scalar_t` is needed for `AT_DISPATCH_FLOATING_TYPES`
    * [more](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h)

`AT_DISPATCH_FLOATING_TYPES()` can be done by some thing like this.

```cpp
switch (tensor.type().scalarType()) {
  case torch::ScalarType::Double:
    return function<double>(tensor.data<double>());
  case torch::ScalarType::Float:
    return function<float>(tensor.data<float>());
  ...
}
```



