/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "count_nonzero.h"

#include <algorithm>
#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCountNonZeroInputNum = 1;
const uint32_t kCountNonZeroOutputNum = 1;
const int64_t kParallelNum = 2 * 1024;
const char *kCountNonZero = "CountNonZero";

// The following code is used to handle the general case.
// Params to use in ParallelIterator construction.
std::vector<int64_t> cnz_dims;
std::vector<int64_t> cnz_transposed_shape;
int64_t cnz_stride;

// Class def of ParallelIterator.
class ParallelIterator {
 public:
  ParallelIterator(std::vector<int64_t> transposed_shape, std::vector<int64_t> dims,
                   const std::vector<int64_t> &input_shape);
  ~ParallelIterator() = default;
  void Next();
  void Set(int64_t pos);
  inline int64_t Get() const { return _pos; };

 private:
  int64_t _dimension{0};
  std::vector<int64_t> _coord;
  std::vector<int64_t> _shape;
  std::vector<int64_t> _strides;
  std::vector<int64_t> _back_strides;
  std::vector<int64_t> _dims;
  int64_t _pos{0};
};

ParallelIterator::ParallelIterator(std::vector<int64_t> transposed_shape, std::vector<int64_t> dims,
                                   const std::vector<int64_t> &input_shape)
    : _dimension(transposed_shape.size()),
      _coord(transposed_shape.size(), 0),
      _shape(transposed_shape),
      _strides(transposed_shape.size(), 1),
      _back_strides(transposed_shape.size(), 1),
      _dims(dims),
      _pos(0) {
  std::vector<int64_t> strides(_dimension, 1);
  for (int64_t i = _dimension - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * input_shape[i + 1];
  }
  for (int64_t i = _dimension - 1; i >= 0; --i) {
    _strides[i] = strides[_dims[i]];
    _back_strides[i] = (_shape[i] - 1) * _strides[i];
  }
}
void ParallelIterator::Set(int64_t pos) {
  for (int64_t i = _dimension - 1; i >= 0 && pos != 0; --i) {
    _coord[i] = pos % _shape[i];
    _pos += _coord[i] * _strides[i];
    pos /= _shape[i];
  }
}
void ParallelIterator::Next() {
  for (int64_t i = _dimension - 1; i >= 0; --i) {
    if (_coord[i] + 1 == _shape[i]) {
      _coord[i] = 0;
      _pos -= _back_strides[i];
    } else {
      _coord[i]++;
      _pos += _strides[i];
      break;
    }
  }
}

// The two structs is used for tag dispatch in IsNonZero.
template <class T>
struct is_complex_t : std::false_type {};
template <class T>
struct is_complex_t<std::complex<T>> : std::true_type {};

template <class T>
int64_t IsNonZero(T val, std::true_type) {
  return val.real() != 0 || val.imag() != 0 ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
}
template <class T>
int64_t IsNonZero(T val, std::false_type) {
  return val != static_cast<T>(0) ? static_cast<int64_t>(1) : static_cast<int64_t>(0);
}
}  // namespace

namespace aicpu {
template <class T>
uint32_t CountNonZeroComputeImpl(CpuKernelContext &ctx) {
  Tensor *x_tensor = ctx.Input(kFirstInputIndex);
  Tensor *y_tensor = ctx.Output(kFirstOutputIndex);
  const T *x_ptr = reinterpret_cast<const T *>(x_tensor->GetData());
  int64_t *y_ptr = reinterpret_cast<int64_t *>(y_tensor->GetData());
  int64_t data_num = y_tensor->NumElements();
  int64_t input_data_num = x_tensor->NumElements();
  std::vector<int64_t> input_shape = x_tensor->GetTensorShape()->GetDimSizes();

  // For scalar_reduction, start=0, end=input_data_num, cannot be parallelized.
  auto count_nonzero_scalar_shard = [&](int64_t start, int64_t end) {
    y_ptr[0] = 0;
    for (int64_t i = start; i < end; ++i) {
      y_ptr[0] += IsNonZero<T>(x_ptr[i], is_complex_t<T>{});
    }
  };

  // For general case. Can be parallelized but performance is not good.
  auto count_nonzero_shard = [&](int64_t start, int64_t end) {
    ParallelIterator iter(cnz_transposed_shape, cnz_dims, input_shape);
    iter.Set(start * cnz_stride);
    for (int64_t i = start; i < end; ++i) {
      int64_t reduce_initial = static_cast<int64_t>(0);
      for (int64_t j = 0; j < cnz_stride; ++j) {
        reduce_initial += IsNonZero<T>(x_ptr[iter.Get()], is_complex_t<T>{});
        iter.Next();
      }
      y_ptr[i] = reduce_initial;
    }
  };

  if (data_num == 1) {
    count_nonzero_scalar_shard(0, input_data_num);
  } else if (data_num > kParallelNum) {
    CpuKernelUtils::ParallelFor(ctx, data_num, 1, count_nonzero_shard);
  } else {
    count_nonzero_shard(0, data_num);
  }
  return KERNEL_STATUS_OK;
}

uint32_t CountNonZeroDimsCheckAndParse(CpuKernelContext &ctx) {
  std::vector<int64_t> input_shape = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes();
  int64_t input_rank = input_shape.size();
  std::vector<int64_t> dims{};

  auto dims_attr = ctx.GetAttr("dims");
  if (dims_attr != nullptr) {
    dims = dims_attr->GetListInt();
  }
  if (dims.size() == 0) {
    for (int64_t i = 0; i < input_rank; ++i) {
      dims.push_back(i);
    }
  }
  // Check dims in [-x_rank, x_rank)
  for (auto &dim : dims) {
    if (dim < 0) {
      dim += input_rank;
    }
    KERNEL_CHECK_FALSE(dim < input_rank && dim >= 0, KERNEL_STATUS_PARAM_INVALID,
                       "[CountNonZero] dims must be in [-x_rank, x_rank).");
  }
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
  int64_t stride_ = static_cast<int64_t>(1);
  std::vector<int64_t> transposed_shape(input_rank);
  // axes is the transpose of indices.
  // For example, if input_rank = 5, dims = [1, 3],
  // then axes =[0, 2, 4] + [1, 3].
  // Initial value if axes is [?, ?, ?, ?, ?]
  std::vector<int64_t> axes_(input_rank);
  int64_t j = static_cast<int64_t>(0), k = static_cast<int64_t>(0);
  // Put dim indices to keep to front of axes and calculate stride.
  // After this operation, axes becomes [0, 2, 4] + [?, ?],
  // and stride becomes 1 * 2 * 4
  for (int64_t i = 0; i < input_rank; i++) {
    if (j == static_cast<int64_t>(dims.size()) || i != dims[j]) {
      axes_[k] = i;
      ++k;
    } else {
      stride_ *= input_shape[i];
      ++j;
    }
  }
  // Put dim indices to reduce to back of axes.
  // After this operation, axes becomes [0, 2, 4] + [1, 3]
  for (auto &dim : dims) {
    axes_[k] = dim;
    ++k;
  }
  // Calculate transposed_shape using axes.
  // For example, if input_shape = (3, 4, 5, 6, 7), axes = [0, 2, 4, 1, 3],
  // then transposed_shape = (3, 5, 7) + (4, 6)
  std::vector<int64_t> transposed_shape_(input_rank);
  for (int64_t i = 0; i < input_rank; ++i) {
    transposed_shape_[i] = input_shape[axes_[i]];
  }
  // Assign values.
  cnz_stride = stride_, cnz_transposed_shape = transposed_shape_, cnz_dims = axes_;
  return KERNEL_STATUS_OK;
}

uint32_t CountNonZeroCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kCountNonZeroInputNum, kCountNonZeroOutputNum),
                      "[%s] check input and output failed.", kCountNonZero);
  KERNEL_HANDLE_ERROR(CountNonZeroDimsCheckAndParse(ctx), "[%s] check & parse dims failed.", kCountNonZero);
  auto y_data_type = ctx.Output(kFirstOutputIndex)->GetDataType();
  KERNEL_CHECK_FALSE(y_data_type == DT_INT64, KERNEL_STATUS_PARAM_INVALID,
                     "[CountNonZero] Data type of output not supported, which is [%s].", DTypeStr(y_data_type).c_str());
  auto x_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (x_data_type) {
    case DT_INT8:
      return CountNonZeroComputeImpl<int8_t>(ctx);
      break;
    case DT_INT16:
      return CountNonZeroComputeImpl<int16_t>(ctx);
      break;
    case DT_INT32:
      return CountNonZeroComputeImpl<int32_t>(ctx);
      break;
    case DT_INT64:
      return CountNonZeroComputeImpl<int64_t>(ctx);
      break;
    case DT_UINT8:
      return CountNonZeroComputeImpl<uint8_t>(ctx);
      break;
    case DT_UINT16:
      return CountNonZeroComputeImpl<uint16_t>(ctx);
      break;
    case DT_UINT32:
      return CountNonZeroComputeImpl<uint32_t>(ctx);
      break;
    case DT_UINT64:
      return CountNonZeroComputeImpl<uint64_t>(ctx);
      break;
    case DT_FLOAT16:
      return CountNonZeroComputeImpl<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      return CountNonZeroComputeImpl<float>(ctx);
      break;
    case DT_DOUBLE:
      return CountNonZeroComputeImpl<double>(ctx);
      break;
    case DT_COMPLEX64:
      return CountNonZeroComputeImpl<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      return CountNonZeroComputeImpl<std::complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("[CountNonZero] kernel data type [%s] not support.", DTypeStr(x_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCountNonZero, CountNonZeroCpuKernel);
}  // namespace aicpu
