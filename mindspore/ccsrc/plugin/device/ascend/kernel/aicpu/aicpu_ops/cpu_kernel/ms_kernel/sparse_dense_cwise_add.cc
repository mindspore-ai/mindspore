#include "sparse_dense_cwise_add.h"
#include <iostream>
#include "utils/kernel_util.h"
#include "utils/sparse_dense_cwise_utils.h"

namespace aicpu {
namespace {
const char *kSparseDenseCwiseAdd = "SparseDenseCwiseAdd";

#define SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DTYPE, TYPE, CTX)         \
  case (DTYPE): {                                                     \
    uint32_t result = SparseDenseCwiseOpCompute<TYPE>(CTX);           \
    if (result != KERNEL_STATUS_OK) {                                 \
      KERNEL_LOG_ERROR("SparseDenseCwiseAdd kernel compute failed."); \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }
}  // namespace

uint32_t SparseDenseCwiseAddKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(CheckParams(ctx), "SparseDenseCwiseAdd check params failed.");

  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SPARSE_DENSE_CWISE_ADD_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("SparseDenseCwiseAdd kernel data type %s not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSparseDenseCwiseAdd, SparseDenseCwiseAddKernel);
}  // namespace aicpu