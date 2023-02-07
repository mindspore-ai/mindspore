#ifndef AICPU_KERNELS_NORMALIZED_TRIDIAGONAL_SOLVE_H
#define AICPU_KERNELS_NORMALIZED_TRIDIAGONAL_SOLVE_H

#include "cpu_ops_kernel.h"
#include <vector>

namespace aicpu {
class TridiagonalSolveCpuKernel : public CpuKernel {
 public:
  TridiagonalSolveCpuKernel() = default;
  ~TridiagonalSolveCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  Tensor *diags_tensor_ = nullptr;
  Tensor *rhs_tensor_ = nullptr;
  AttrValue *partial_pivoting = nullptr;
  Tensor *output_tensor_ = nullptr;
  size_t matrix_num;
  int64_t data_size;
  uint32_t res;
  int32_t diags_rank;
  int32_t rhs_rank;
  int32_t diags_size;
  int32_t rhs_size;
  std::shared_ptr<TensorShape> diags_shape;
  std::shared_ptr<TensorShape> rhs_shape;
  std::vector<int64_t> diags_dimsize;
  std::vector<int64_t> rhs_dimsize;
  DataType diags_type_ = DT_DOUBLE;
  DataType rhs_type_ = DT_DOUBLE;
  DataType data_type_ = DT_DOUBLE;
  int DimSize0 = 0;
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  uint32_t choosedatatype_(CpuKernelContext &ctx, size_t nth_batch, int i);

  template <typename T>
  uint32_t DoCompute1(CpuKernelContext &ctx, size_t nth_batch, int i);

  template <typename T>
  uint32_t DoCompute2(CpuKernelContext &ctx, size_t nth_batch, int i);

};  // namespace aicpu
#endif
}
