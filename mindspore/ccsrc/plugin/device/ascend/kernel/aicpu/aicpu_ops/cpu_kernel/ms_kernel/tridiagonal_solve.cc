/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tridiagonal_solve.h"
#include <iostream>
#include "Eigen/Core"
#include "Eigen/LU"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"

using namespace Eigen;
using namespace std;

namespace {
const char *TRIDIAGONALSOLVE = "TridiagonalSolve";
// 是否启用多线程的标识分界点
const int64_t kParallelDataNumSameShape = 8 * 1024;
}  // namespace

// 定义命名空间aicpu
namespace aicpu {

// 读取输入输出以及exception抛出
uint32_t TridiagonalSolveCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  // get 输入输出指针
  diags_tensor_ = ctx.Input(0);
  rhs_tensor_ = ctx.Input(1);
  output_tensor_ = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, 2, 1), "Less check input and output number failed.");
  // get shape指针
  std::shared_ptr<TensorShape> diags_shape = diags_tensor_->GetTensorShape();
  KERNEL_CHECK_NULLPTR(diags_shape, KERNEL_STATUS_PARAM_INVALID, "Get shape of input[0], diags failed");
  std::shared_ptr<TensorShape> rhs_shape = rhs_tensor_->GetTensorShape();
  KERNEL_CHECK_NULLPTR(rhs_shape, KERNEL_STATUS_PARAM_INVALID, "Get shape of input[1], rhs failed");

  // get 输入维度
  int32_t diags_rank = diags_shape->GetDims();
  int32_t rhs_rank = rhs_shape->GetDims();

  // get diags和rhs矩阵的尺寸
  rhs_size =
    rhs_tensor_->GetTensorShape()->GetDimSize(rhs_rank - 1) * rhs_tensor_->GetTensorShape()->GetDimSize(rhs_rank - 2);
  diags_size = diags_tensor_->GetTensorShape()->GetDimSize(diags_rank - 1) *
               diags_tensor_->GetTensorShape()->GetDimSize(diags_rank - 2);

  // get shape的size vector
  std::vector<int64_t> diags_dimsize = diags_shape->GetDimSizes();
  std::vector<int64_t> rhs_dimsize = rhs_shape->GetDimSizes();

  // get partial_pivoting
  partial_pivoting = ctx.GetAttr("partial_pivoting");

  // get diags_type_和rhs_type_
  diags_type_ = static_cast<DataType>(diags_tensor_->GetDataType());
  rhs_type_ = static_cast<DataType>(rhs_tensor_->GetDataType());

  // get data_type_
  data_type_ = rhs_type_;

  // exception抛出

  //  1) 维度小于2
  if (diags_rank < 2) {
    KERNEL_LOG_ERROR("Expected diags to have rank at least 2, got[%d]", diags_rank);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // 2) diags和rhs维度不匹配
  if (rhs_rank != diags_rank) {
    KERNEL_LOG_ERROR("Expected the rank of rhs to be [%d] or [%d], got [%d]", diags_rank - 1, diags_rank, rhs_rank);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  //  3) diags没有三行
  DimSize0 = diags_shape->GetDimSize(diags_rank - 2);
  if (DimSize0 != 3) {
    KERNEL_LOG_ERROR("Expected 3 diagonals got [%d]", DimSize0);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // 4) batch_size不一致
  for (int i = 0; i < diags_rank - 2; i++) {
    if (diags_dimsize[i] != rhs_dimsize[i]) {
      KERNEL_LOG_ERROR("Batch shapes of diags and rhs are incompatible");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  //  5) diags和rhs类型不一致
  if (diags_type_ != rhs_type_) {
    KERNEL_LOG_ERROR("The type of diags and rhs are incompatible");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  //  6) 输入为空
  if (diags_dimsize.size() == 0 || rhs_dimsize.size() == 0) {
    KERNEL_LOG_ERROR("The input is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // 7)diags和rhs长度不匹配
  int DimSize1 = diags_shape->GetDimSize(diags_rank - 1);
  int RhsSize0 = rhs_shape->GetDimSize(rhs_rank - 2);
  if (DimSize1 != RhsSize0) {
    KERNEL_LOG_ERROR("The length of diags and rhs are incompatible");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // 8) 输入的数据类型无法处理
  if (diags_type_ != DT_FLOAT && diags_type_ != DT_DOUBLE && diags_type_ != DT_COMPLEX64 &&
      diags_type_ != DT_COMPLEX128) {
    KERNEL_LOG_ERROR("The type of inputs are invalid");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

// 根据数据类型为模板函数赋不同的值,并且根据partial_pivoting的值选择不同的计算方法
uint32_t TridiagonalSolveCpuKernel::choosedatatype_(CpuKernelContext &ctx, size_t nth_batch, int i) {
  if (partial_pivoting->GetBool()) {
    switch (data_type_) {
      case DT_FLOAT: {
        res = DoCompute1<float>(ctx, nth_batch, i);
        break;
      }
      case DT_DOUBLE: {
        res = DoCompute1<double>(ctx, nth_batch, i);
        break;
      }
      case DT_COMPLEX64: {
        res = DoCompute1<std::complex<float>>(ctx, nth_batch, i);
        break;
      }
      case DT_COMPLEX128: {
        res = DoCompute1<std::complex<double>>(ctx, nth_batch, i);
        break;
      }
      default: {
        KERNEL_LOG_ERROR(
          "Tridiagonal-solve op support input tensor type: float、double、complex64、complex128,should not be tensor "
          "type [%s]",
          data_type_);
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else {
    switch (data_type_) {
      case DT_FLOAT:
        res = DoCompute2<float>(ctx, nth_batch, i);
        break;
      case DT_DOUBLE:
        res = DoCompute2<double>(ctx, nth_batch, i);
        break;
      case DT_COMPLEX64:
        res = DoCompute2<std::complex<float>>(ctx, nth_batch, i);
        break;
      case DT_COMPLEX128:
        res = DoCompute2<std::complex<double>>(ctx, nth_batch, i);
        break;
      default: {
        KERNEL_LOG_ERROR(
          "Tridiagonal-solve op support input tensor type: float、double、complex64、complex128,should not be tensor "
          "type [%s]",
          data_type_);
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  return res;
}

// 当partial_pivoting的值为true时的计算函数
template <typename T>
uint32_t TridiagonalSolveCpuKernel::DoCompute1(CpuKernelContext &ctx, size_t nth_batch, int i) {
  // 计算变量的尺寸
  int rhs_rank = rhs_tensor_->GetTensorShape()->GetDims();
  int diags_rank = diags_tensor_->GetTensorShape()->GetDims();
  const int batch = rhs_tensor_->GetTensorShape()->GetDimSize(rhs_rank - 1);
  const int n = diags_tensor_->GetTensorShape()->GetDimSize(diags_rank - 1);

  // 计算分片后该次运算的起始地址
  auto a = static_cast<T *>(diags_tensor_->GetData());
  auto b = static_cast<T *>(rhs_tensor_->GetData());
  auto value = reinterpret_cast<T *>(output_tensor_->GetData());
  if (i == -1) {
    a += nth_batch * diags_size;
    b += nth_batch * rhs_size;
    value += nth_batch * rhs_size;
  } else {
    a += i * diags_size;
    b += i * rhs_size;
    value += i * rhs_size;
  }

  const T zero = 0;

  // 用于计算的中间变量
  Array<T, Dynamic, 3> u(n, 3);

  // 输入superdiags,diags,subdiags
  Array<T, Dynamic, 1> superdiag(n);
  Array<T, Dynamic, 1> diag(n);
  Array<T, Dynamic, 1> subdiag(n);

  // 输入rhs
  Array<T, Dynamic, Dynamic> rhs(n, batch);

  // 计算结果x
  Array<T, Dynamic, Dynamic> x(n, batch);

  // 将输入数据装载进变量
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < batch; j++) {
      rhs(i, j) = *(b + i * batch + j);
    }
  }

  for (int i = 0; i < n; i++) {
    superdiag(i) = *(a + i);
    diag(i) = *(a + n + i);
    subdiag(i) = *(a + 2 * n + i);
  }

  // 计算过程
  u(0, 0) = diag(0);
  u(0, 1) = superdiag(0);
  x.row(0) = rhs.row(0);

  for (int i = 0; i < n - 1; ++i) {
    if (abs(u(i, 0)) >= abs(subdiag(i + 1))) {
      // No row interchange.
      if (u(i, 0) == zero) {
        KERNEL_LOG_ERROR("The first element of diag should not be zero");
        return KERNEL_STATUS_PARAM_INVALID;
      }
      const T factor = subdiag(i + 1) / u(i, 0);
      u(i + 1, 0) = diag(i + 1) - factor * u(i, 1);
      x.row(i + 1) = rhs.row(i + 1) - factor * x.row(i);
      if (i != n - 2) {
        u(i + 1, 1) = superdiag(i + 1);
        u(i, 2) = 0;
      }
    } else {
      // Interchange rows i and i + 1.
      const T factor = u(i, 0) / subdiag(i + 1);
      u(i, 0) = subdiag(i + 1);
      u(i + 1, 0) = u(i, 1) - factor * diag(i + 1);
      u(i, 1) = diag(i + 1);
      x.row(i + 1) = x.row(i) - factor * rhs.row(i + 1);
      x.row(i) = rhs.row(i + 1);
      if (i != n - 2) {
        u(i, 2) = superdiag(i + 1);
        u(i + 1, 1) = -factor * superdiag(i + 1);
      }
    }
  }
  if (u(n - 1, 0) == zero) {
    KERNEL_LOG_ERROR("The last element of diag should not be zero");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // 计算最终结果并且存入相应的输出地址中
  x.row(n - 1) /= u(n - 1, 0);
  for (int j = 0; j < batch; j++) {
    *(value + (n - 1) * batch + j) = x(n - 1, j);
  }

  x.row(n - 2) = (x.row(n - 2) - u(n - 2, 1) * x.row(n - 1)) / u(n - 2, 0);
  for (int j = 0; j < batch; j++) {
    *(value + (n - 2) * batch + j) = x(n - 2, j);
  }

  for (int i = n - 3; i >= 0; --i) {
    x.row(i) = (x.row(i) - u(i, 1) * x.row(i + 1) - u(i, 2) * x.row(i + 2)) / u(i, 0);
    for (int j = 0; j < batch; j++) {
      *(value + i * batch + j) = x(i, j);
    }
  }

  // KERNEL_LOG_INFO("TridiagonalSolveCpuKernel::DoCompute end! ");
  return KERNEL_STATUS_OK;
}

// 当partial_pivoting的值为false时的计算函数
template <typename T>
uint32_t TridiagonalSolveCpuKernel::DoCompute2(CpuKernelContext &ctx, size_t nth_batch, int i) {
  // 计算变量的尺寸
  int rhs_rank = rhs_tensor_->GetTensorShape()->GetDims();
  int diags_rank = diags_tensor_->GetTensorShape()->GetDims();
  const int batch = rhs_tensor_->GetTensorShape()->GetDimSize(rhs_rank - 1);
  const int n = diags_tensor_->GetTensorShape()->GetDimSize(diags_rank - 1);
  // 计算分片后该次运算的起始地址
  auto a = static_cast<T *>(diags_tensor_->GetData());
  auto b = static_cast<T *>(rhs_tensor_->GetData());
  auto value = reinterpret_cast<T *>(output_tensor_->GetData());
  if (i == -1) {
    a += nth_batch * diags_size;
    b += nth_batch * rhs_size;
    value += nth_batch * rhs_size;
  } else {
    a += i * diags_size;
    b += i * rhs_size;
    value += i * rhs_size;
  }

  // 用于计算的中间变量
  Array<T, Dynamic, 3> u(n, 3);

  // 输入superdiags,diags,subdiags
  Array<T, Dynamic, 1> superdiag(n);
  Array<T, Dynamic, 1> diag(n);
  Array<T, Dynamic, 1> subdiag(n);

  // 输入rhs
  Array<T, Dynamic, Dynamic> rhs(n, batch);

  // 计算结果x
  Array<T, Dynamic, Dynamic> x(n, batch);

  const T zero = 0;

  // 计算过程
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < batch; j++) {
      rhs(i, j) = *(b + i * batch + j);
    }
  }

  for (int i = 0; i < n; i++) {
    superdiag(i) = *(a + i);
    diag(i) = *(a + n + i);
    subdiag(i) = *(a + 2 * n + i);
  }

  if (diag(0) == zero) {
    KERNEL_LOG_ERROR("The first element of diag should not be zero");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  u(0) = superdiag(0) / diag(0);
  x.row(0) = rhs.row(0) / diag(0);
  for (int i = 1; i < n; ++i) {
    auto denom = diag(i) - subdiag(i) * u(i - 1);
    if (denom == zero) {
      KERNEL_LOG_ERROR("The diag should not be zero");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    u(i) = superdiag(i) / denom;
    x.row(i) = (rhs.row(i) - subdiag(i) * x.row(i - 1)) / denom;
  }
  for (int i = n - 2; i >= 0; --i) {
    x.row(i) -= u(i) * x.row(i + 1);
  }

  // 计算最终结果并且存入相应的输出地址中
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < batch; j++) {
      *(value + i * batch + j) = x(i, j);
    }
  }

  KERNEL_LOG_INFO("TridiagonalSolveCpuKernel::DoCompute end! ");
  return KERNEL_STATUS_OK;
}

// 主函数
uint32_t TridiagonalSolveCpuKernel::Compute(CpuKernelContext &ctx) {
  res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  data_size = ctx.Input(0)->NumElements();
  matrix_num = ctx.Input(0)->NumElements() / diags_size;

  // 判断多线程
  if (data_size >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    // 使用CpuKernelUtils::GetCPUNum接口获取AI CPU的核数
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    // 若AI CPU中核数大于数据量，以数据量作为max_core_num
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    // 多线程的lambda函数
    auto shared_tridiagonalsolve = [&](size_t start, size_t end) {
      for (size_t nth_batch = start; nth_batch < end; nth_batch++) res = choosedatatype_(ctx, nth_batch, -1);
    };
    CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shared_tridiagonalsolve);
  } else {
    // 若数据量小于8K，不进行分片，使用单AI CPU核进行计算。
    for (size_t nth_batch = 0; nth_batch < matrix_num; nth_batch++) res = choosedatatype_(ctx, -1, nth_batch);
  }
  return res;
}

// 注册该算子实现
REGISTER_CPU_KERNEL(TRIDIAGONALSOLVE, TridiagonalSolveCpuKernel);
}  // namespace aicpu