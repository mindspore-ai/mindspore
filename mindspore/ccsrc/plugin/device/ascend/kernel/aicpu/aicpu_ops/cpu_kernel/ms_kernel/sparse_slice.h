#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_

#include <string>
#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
class SparseSliceCpuKernel : public CpuKernel {
 public:
  SparseSliceCpuKernel() = default;
  ~SparseSliceCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
  uint32_t SparseSliceParamCheck(Tensor *indices, Tensor *values, Tensor *shape, Tensor *start, Tensor *size);
};
}  // namespace aicpu
#endif
