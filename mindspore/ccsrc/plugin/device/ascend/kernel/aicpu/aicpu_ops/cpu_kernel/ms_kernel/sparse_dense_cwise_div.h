#ifndef AICPU_KERNELS_SPARSE_DENSE_CWISE_DIV_H_
#define AICPU_KERNELS_SPARSE_DENSE_CWISE_DIV_H_

#include "utils/sparse_dense_cwise_utils.h"

namespace aicpu {
class SparseDenseCwiseDivKernel : public SparseDenseCwiseOpKernel<DivOp> {
 public:
  SparseDenseCwiseDivKernel() = default;
  ~SparseDenseCwiseDivKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};

}  // namespace aicpu
#endif