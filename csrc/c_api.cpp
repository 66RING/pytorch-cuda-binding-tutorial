#include <torch/python.h>
#include <iostream>

void hello() {
    std::cout << "Hello, World from c!" << std::endl;
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")
  m.def("hello", &hello, "Prints hello world from c file");
  m.def("vector_add", &vector_add, "Adds two vectors");
}

