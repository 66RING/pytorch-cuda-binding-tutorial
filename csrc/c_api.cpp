#include <torch/python.h>
#include <iostream>

void hello() {
    std::cout << "Hello, World from c!" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")
  m.def("hello", &hello, "Prints hello world from c file");
}

