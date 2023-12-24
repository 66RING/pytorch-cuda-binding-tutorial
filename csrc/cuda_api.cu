#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/python.h>

#include "cuda_api.h"

__global__ void hello_cuda123(char *out) {
  int i = threadIdx.x;
  out[i] += i;
}

void hello() {
  std::string str = "0000";
  int length = str.length();
  char *device_str;

  cudaMalloc((void **)&device_str, length + 1);
  // data copy to device
  cudaMemcpy(device_str, str.data(), length + 1, cudaMemcpyHostToDevice);

  hello_cuda123<<<1, 4>>>(device_str);
  // wait until cuda kernel finish
  cudaDeviceSynchronize();

  // data copy back from device
  cudaMemcpy(str.data(), device_str, length + 1, cudaMemcpyDeviceToHost);
  std::cout << "Hello, World from cuda! " << str << std::endl;

  cudaFree(device_str);
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  auto out = torch::zeros_like(a);
  auto size = a.size(0);
  // NOTE: AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)
  // We need a way of determining at runtime what type a tensor is and then
  // selectively call functions with the corresponding correct type signature.
  AT_DISPATCH_FLOATING_TYPES(a.type(), "anyname", ([&] {
                               vector_add_cuda_kernel<scalar_t><<<1, size>>>(
                                   a.data<scalar_t>(), b.data<scalar_t>(),
                                   out.data<scalar_t>());
                             }));
  return out;
}

template <typename scalar_t>
__global__ void vector_add_cuda_kernel(const scalar_t *a, const scalar_t *b,
                                       scalar_t *out) {
  const int index = threadIdx.x;
  out[index] = a[index] + b[index];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")
  m.def("hello", &hello, "Prints hello world from cuda file");
  m.def("vector_add", &vector_add, "Add two vectors on cuda");
}
