

#ifndef AICPU_UtILS_SPARSE_DENSE_CWISE_UTILS_H_
#define AICPU_UtILS_SPARSE_DENSE_CWISE_UTILS_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
struct AddOp {
  static std::string Name() { return "Add"; }
};

struct DivOp {
  static std::string Name() { return "Div"; }
};

struct MulOp {
  static std::string Name() { return "Mul"; }
};

template <typename Op>
class SparseDenseCwiseOpKernel : public CpuKernel {
 public:
  SparseDenseCwiseOpKernel() = default;
  ~SparseDenseCwiseOpKernel() override = default;

 protected:
  virtual uint32_t Compute(CpuKernelContext &ctx) override = 0;

  static uint32_t CheckParams(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpSpecialCompute(BcastShapeType type, CpuKernelContext &ctx);
  template <typename T>
  uint32_t SparseDenseCwiseOpSpecialComputeComplex(BcastShapeType type, CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeOp(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ComputeOpComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpNoBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpNoBcastComputeComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpBcastComputeComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SparseDenseCwiseOpCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
