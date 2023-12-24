#include <torch/python.h>

// NOTE:tensor malloc as device before we call
// e.g. data.to("cuda") in python
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void vector_add_cuda_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out);
