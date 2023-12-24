#include <iostream>
#include <torch/python.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")
  m.def("hello", &hello, "Prints hello world from cuda file");
}
