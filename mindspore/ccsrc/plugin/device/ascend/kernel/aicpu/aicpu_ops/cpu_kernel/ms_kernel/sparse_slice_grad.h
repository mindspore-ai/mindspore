#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPARSE_OPS_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class SparseSliceGradCpuKernel : public CpuKernel {
 public:
  SparseSliceGradCpuKernel() = default;
  ~SparseSliceGradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
  uint32_t SparseSliceGradParamCheck(Tensor *backprop_val_grad, Tensor *indices, Tensor *start, Tensor *new_indices);

  template <typename T>
  uint32_t GradCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif