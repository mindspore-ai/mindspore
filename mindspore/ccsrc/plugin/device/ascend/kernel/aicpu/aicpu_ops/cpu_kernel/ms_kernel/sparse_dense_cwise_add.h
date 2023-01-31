#ifndef AICPU_KERNELS_SPARSE_DENSE_CWISE_ADD_H_
#define AICPU_KERNELS_SPARSE_DENSE_CWISE_ADD_H_

#include "utils/sparse_dense_cwise_utils.h"

namespace aicpu {
class SparseDenseCwiseAddKernel : public SparseDenseCwiseOpKernel<AddOp> {
 public:
  SparseDenseCwiseAddKernel() = default;
  ~SparseDenseCwiseAddKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;
};

}  // namespace aicpu
#endif