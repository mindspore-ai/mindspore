#ifndef AICPU_KERNELS_SPARSE_DENSE_CWISE_MUL_H_
#define AICPU_KERNELS_SPARSE_DENSE_CWISE_MUL_H_

#include "utils/sparse_dense_cwise_utils.h"

namespace aicpu {
class SparseDenseCwiseMulKernel : public SparseDenseCwiseOpKernel<MulOp> {
 public:
  SparseDenseCwiseMulKernel() = default;
  ~SparseDenseCwiseMulKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};

}  // namespace aicpu
#endif