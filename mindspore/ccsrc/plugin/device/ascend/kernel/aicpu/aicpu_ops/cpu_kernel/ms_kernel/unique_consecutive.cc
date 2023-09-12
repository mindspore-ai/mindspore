/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#include "cpu_kernel/ms_kernel/unique_consecutive.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#define NoneN 1000
using namespace std;

namespace {
const size_t kInputNum = 1;
const char *const kUniqueConsecutive = "UniqueConsecutive";

int32_t MaybeWrapDim(int32_t dim, int32_t dim_post_expr) {
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

template <typename T>
class PositionIterator {
 public:
  PositionIterator(){};
  ~PositionIterator(){};
  PositionIterator(std::vector<T> stt, std::vector<T> sh) {
    if (stt.size() != sh.size()) {
      PositionIterator();
    } else {
      for (unsigned int i = 0; i < sh.size(); i++) {
        if (stt[i] >= sh[i]) {
          PositionIterator();
        }
      }
      pos_ = stt;
      shape_ = sh;
    }
  }
  PositionIterator operator++() {
    pos_[shape_.size() - static_cast<size_t>(1)] += 1;
    for (size_t i = shape_.size() - static_cast<size_t>(1); i > static_cast<size_t>(0); i--) {
      if (pos_[i] / shape_[i] != 0) {
        pos_[i - 1] += pos_[i] / shape_[i];
        pos_[i] = pos_[i] % shape_[i];
      }
    }
    return *this;
  }

  bool End() {
    if (pos_[0] != shape_[0]) {
      return false;
    }
    return true;
  }

  std::vector<T> GetPos() { return pos_; }

  std::vector<T> GetShape() { return shape_; }

 private:
  std::vector<T> pos_;
  std::vector<T> shape_;
};

template <typename T>
std::vector<T> ConstructStride(std::vector<T> t_shape) {
  std::vector<T> t_stride(t_shape.size(), 1);
  int initial = 1;
  for (size_t i = t_shape.size(); i > 0; i--) {
    t_stride[i - 1] = initial;
    initial = initial * static_cast<int>(t_shape[i - static_cast<size_t>(1)]);
  }
  return t_stride;
}

template <typename T>
T MulSum(std::vector<T> v1, std::vector<T> v2) {
  if (v1.size() != v2.size()) {
    return static_cast<T>(false);
  } else {
    T output = 0;
    for (unsigned int i = 0; i < v1.size(); i++) {
      output += v1[i] * v2[i];
    }
    return output;
  }
}
template <typename T1>
std::vector<std::vector<T1>> ReshapeInput(const std::vector<int64_t> &input_shape_, int32_t axis, T1 *x_dataptr) {
  int64_t dim0 = input_shape_[static_cast<size_t>(axis)];
  std::vector<int64_t> input_stride = ConstructStride<int64_t>(input_shape_);
  std::vector<int64_t> v_shape = input_shape_;
  v_shape.erase(v_shape.begin() + axis);
  std::vector<int64_t> v_start(v_shape.size(), 0);
  std::vector<int64_t> v_stride = input_stride;
  v_stride.erase(v_stride.begin() + axis);
  std::vector<std::vector<T1>> data_;
  for (int64_t i = 0; i < dim0; i++) {
    std::vector<T1> tmp_v1;
    for (PositionIterator<int64_t> mit(v_start, v_shape); !mit.End(); ++mit) {
      auto pos = mit.GetPos();
      tmp_v1.push_back(
        x_dataptr[static_cast<size_t>(MulSum<int64_t>(pos, v_stride) + i * input_stride[static_cast<size_t>(axis)])]);
    }
    data_.push_back(tmp_v1);
  }
  return data_;
}
}  // namespace

namespace aicpu {
void UniqueConsecutiveCpuKernel::DefaultSet(const CpuKernelContext &ctx) {
  // Get the inuput and output
  Tensor *output_y = ctx.Output(0);
  auto y_shape = ctx.Output(0)->GetTensorShape();
  // Set output y shape
  std::vector<int64_t> shape = {0};
  y_shape->SetDimSizes(shape);
  output_y->SetTensorShape(y_shape.get());
  // Set output idx and count shape
  Tensor *output1 = ctx.Output(1);
  output1->SetTensorShape(y_shape.get());
  Tensor *output2 = ctx.Output(2);
  output2->SetTensorShape(y_shape.get());
}

template <typename T1>
void UniqueConsecutiveCpuKernel::OutputYSet(const std::vector<int64_t> &y_shape_,
                                            const std::vector<int64_t> &input_shape_, int32_t axis, T1 *y_dataptr,
                                            const std::vector<std::vector<T1>> &out_data_) {
  std::vector<int64_t> y_stride = ConstructStride<int64_t>(y_shape_);
  std::vector<int64_t> y_v_shape = y_shape_;
  y_v_shape.erase(y_v_shape.begin() + axis);
  std::vector<int64_t> y_v_start(y_v_shape.size(), 0);
  std::vector<int64_t> y_v_stride = y_stride;
  y_v_stride.erase(y_v_stride.begin() + axis);
  std::vector<int64_t> v_shape = input_shape_;
  v_shape.erase(v_shape.begin() + axis);
  std::vector<int64_t> trans_stride = ConstructStride<int64_t>(v_shape);
  int64_t size0 = static_cast<int64_t>(out_data_.size());
  for (int64_t i = 0; i < size0; i++) {
    auto tmp_v = out_data_[static_cast<size_t>(i)];
    for (PositionIterator<int64_t> mit(y_v_start, y_v_shape); !mit.End(); ++mit) {
      auto pos = mit.GetPos();
      y_dataptr[static_cast<size_t>(MulSum<int64_t>(pos, y_v_stride) + i * y_stride[axis])] =
        tmp_v[static_cast<size_t>(MulSum<int64_t>(pos, trans_stride))];
    }
  }
}

template <typename T2>
void UniqueConsecutiveCpuKernel::SetOuputIdxandCount(const CpuKernelContext &ctx,
                                                     const std::vector<int64_t> &idx_shape_,
                                                     const std::vector<int64_t> &count_shape_, T2 *idx_dataptr,
                                                     T2 *count_dataptr) {
  std::vector<int64_t> shape = {0};
  // Output 1 -- idx
  auto output_1 = ctx.Output(1);
  auto output1_shape = output_1->GetTensorShape();
  if (return_idx_) {
    output1_shape->SetDimSizes(idx_shape_);
  } else {
    output1_shape->SetDimSizes(shape);
  }
  output_1->SetTensorShape(output1_shape.get());
  auto output_data_1 = reinterpret_cast<T2 *>(output_1->GetData());
  for (int i = 0; i < output1_shape->NumElements(); ++i) {
    *(output_data_1 + i) = *(idx_dataptr + i);
  }
  // Output 2 -- count
  auto output_2 = ctx.Output(2);
  auto output2_shape = output_2->GetTensorShape();
  if (return_counts_) {
    output2_shape->SetDimSizes(count_shape_);
  } else {
    output2_shape->SetDimSizes(shape);
  }
  output_2->SetTensorShape(output2_shape.get());
  auto output_data_2 = reinterpret_cast<T2 *>(output_2->GetData());
  int64_t data_num = output2_shape->NumElements();
  for (int i = 0; i < data_num; ++i) {
    *(output_data_2 + i) = *(count_dataptr + i);
  }
}

template <typename T2>
uint32_t UniqueConsecutiveCpuKernel::DtypeMapNone(const CpuKernelContext &ctx, DataType x_dtype) {
  switch (x_dtype) {
    case DT_COMPLEX128:
      return DoComputeNone<complex<double>, T2>(ctx);
    case DT_COMPLEX64:
      return DoComputeNone<complex<float>, T2>(ctx);
    case DT_DOUBLE:
      return DoComputeNone<double, T2>(ctx);
    case DT_FLOAT:
      return DoComputeNone<float, T2>(ctx);
    case DT_FLOAT16:
      return DoComputeNone<Eigen::half, T2>(ctx);
    case DT_INT8:
      return DoComputeNone<int8_t, T2>(ctx);
    case DT_INT16:
      return DoComputeNone<int16_t, T2>(ctx);
    case DT_INT32:
      return DoComputeNone<int32_t, T2>(ctx);
    case DT_INT64:
      return DoComputeNone<int64_t, T2>(ctx);
    case DT_UINT8:
      return DoComputeNone<uint8_t, T2>(ctx);
    case DT_UINT16:
      return DoComputeNone<uint16_t, T2>(ctx);
    case DT_UINT32:
      return DoComputeNone<uint32_t, T2>(ctx);
    case DT_UINT64:
      return DoComputeNone<uint64_t, T2>(ctx);
    default:
      KERNEL_LOG_ERROR("[UniqueConsecutive]: Input data type [%s] not support.", DTypeStr(x_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T2>
uint32_t UniqueConsecutiveCpuKernel::DtypeMapDim(const CpuKernelContext &ctx, int32_t tmp_axis, DataType x_dtype) {
  switch (x_dtype) {
    case DT_COMPLEX128:
      return DoComputeDim<complex<double>, T2>(ctx, tmp_axis);
    case DT_COMPLEX64:
      return DoComputeDim<complex<float>, T2>(ctx, tmp_axis);
    case DT_DOUBLE:
      return DoComputeDim<double, T2>(ctx, tmp_axis);
    case DT_FLOAT:
      return DoComputeDim<float, T2>(ctx, tmp_axis);
    case DT_FLOAT16:
      return DoComputeDim<Eigen::half, T2>(ctx, tmp_axis);
    case DT_INT8:
      return DoComputeDim<int8_t, T2>(ctx, tmp_axis);
    case DT_INT16:
      return DoComputeDim<int16_t, T2>(ctx, tmp_axis);
    case DT_INT32:
      return DoComputeDim<int32_t, T2>(ctx, tmp_axis);
    case DT_INT64:
      return DoComputeDim<int64_t, T2>(ctx, tmp_axis);
    case DT_UINT8:
      return DoComputeDim<uint8_t, T2>(ctx, tmp_axis);
    case DT_UINT16:
      return DoComputeDim<uint16_t, T2>(ctx, tmp_axis);
    case DT_UINT32:
      return DoComputeDim<uint32_t, T2>(ctx, tmp_axis);
    case DT_UINT64:
      return DoComputeDim<uint64_t, T2>(ctx, tmp_axis);
    default:
      KERNEL_LOG_ERROR("[UniqueConsecutive]: Input data type [%s] not support.", DTypeStr(x_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T1, typename T2>
uint32_t UniqueConsecutiveCpuKernel::DoComputeNone(const CpuKernelContext &ctx) {
  // Get the inuput and output y
  Tensor *input_x = ctx.Input(0);
  Tensor *output_y = ctx.Output(0);
  // Get some information of input and output y
  int64_t numel = input_x->NumElements();
  auto x_dataptr = reinterpret_cast<T1 *>(input_x->GetData());
  auto y_dataptr = reinterpret_cast<T1 *>(output_y->GetData());
  // Compute
  if (numel > 0) {
    T2 *idx_dataptr = new (std::nothrow) T2[numel];
    KERNEL_CHECK_NULLPTR(idx_dataptr, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
    T2 *count_dataptr = new (std::nothrow) T2[numel];
    if (count_dataptr == nullptr) {
      delete[] idx_dataptr;
      return KERNEL_STATUS_INNER_ERROR;
    }
    *y_dataptr = *x_dataptr;
    T1 *p = y_dataptr;
    T2 *q = count_dataptr;
    T2 last = 0;
    for (T2 i = 0; i < numel; i++) {
      if (x_dataptr[i] != *p) {
        *(++p) = x_dataptr[i];
        *(q++) = i - last;
        last = i;
      }
      idx_dataptr[i] = static_cast<T2>(p - y_dataptr);
    }
    *q = numel - last;
    auto x_shape = input_x->GetTensorShape();
    auto y_shape = output_y->GetTensorShape();

    if (x_shape->GetDims() == 0) {
      output_y->SetTensorShape(x_shape.get());
      SetOuputIdxandCount<T2>(ctx, x_shape->GetDimSizes(), x_shape->GetDimSizes(), idx_dataptr, count_dataptr);
    } else {
      std::vector<int64_t> shape;
      shape.push_back((p - y_dataptr) + 1);
      y_shape->SetDimSizes(shape);
      output_y->SetTensorShape(y_shape.get());
      SetOuputIdxandCount<T2>(ctx, x_shape->GetDimSizes(), shape, idx_dataptr, count_dataptr);
    }
    delete[] idx_dataptr;
    delete[] count_dataptr;
  } else {
    DefaultSet(ctx);
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t UniqueConsecutiveCpuKernel::DoComputeDim(const CpuKernelContext &ctx, const int32_t axis) {
  auto x_dataptr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto y_dataptr = reinterpret_cast<T1 *>(ctx.Output(0)->GetData());
  auto input_shape = ctx.Input(0)->GetTensorShape();
  std::vector<int64_t> input_shape_ = input_shape->GetDimSizes();
  std::vector<int64_t> y_shape_ = input_shape_;
  std::vector<int64_t> idx_shape_, count_shape_;
  auto num_zero_dims = std::count(input_shape_.begin(), input_shape_.end(), 0);
  int64_t dim0 = input_shape_[static_cast<size_t>(axis)];
  idx_shape_.push_back(dim0);
  if (dim0 == 0) {
    KERNEL_CHECK_FALSE((num_zero_dims == 1), KERNEL_STATUS_PARAM_INVALID,
                       "[UniqueConsecutive]: Number of zero sized dimensions "
                       "is more than one, so unique cannot be applied.");
    DefaultSet(ctx);
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((num_zero_dims == 0), KERNEL_STATUS_PARAM_INVALID,
                     "[UniqueConsecutive]: There are 0 sized dimensions, and "
                     "they aren't selected, so unique cannot be applied");
  if (input_shape_.size() != 1) {
    T2 *idx_dataptr = new (std::nothrow) T2[dim0];
    KERNEL_CHECK_NULLPTR(idx_dataptr, KERNEL_STATUS_INNER_ERROR, "apply memory failed.");
    T2 *count_dataptr = new (std::nothrow) T2[dim0];
    if (count_dataptr == nullptr) {
      delete[] idx_dataptr;
      return KERNEL_STATUS_INNER_ERROR;
    }
    std::vector<std::vector<T1>> data_ = ReshapeInput<T1>(input_shape_, axis, x_dataptr);
    std::vector<std::vector<T1>> out_data_;
    out_data_.push_back(data_[0]);
    auto p = data_[0];
    T2 *q = count_dataptr;
    T2 last = 0;
    for (size_t i = 0; i < static_cast<size_t>(dim0); i++) {
      if (!std::equal(data_[i].begin(), data_[i].end(), p.begin())) {
        p = data_[i];
        out_data_.push_back(data_[i]);
        *(q++) = static_cast<T2>(i) - last;
        last = static_cast<T2>(i);
      }
      idx_dataptr[i] = static_cast<T2>(static_cast<int32_t>(out_data_.size()) - 1);
    }
    *q = static_cast<T2>(dim0) - last;
    y_shape_[static_cast<size_t>(axis)] = static_cast<int64_t>(out_data_.size());
    OutputYSet(y_shape_, y_shape_, axis, y_dataptr, out_data_);
    count_shape_.push_back(out_data_.size());
    // Set output y shape
    Tensor *output_y = ctx.Output(0);
    auto y_shape = output_y->GetTensorShape();
    y_shape->SetDimSizes(y_shape_);
    output_y->SetTensorShape(y_shape.get());
    // Set output idx and count
    SetOuputIdxandCount<T2>(ctx, idx_shape_, count_shape_, idx_dataptr, count_dataptr);
    delete[] idx_dataptr;
    delete[] count_dataptr;
  } else {
    return DoComputeNone<T1, T2>(ctx);
  }
  return KERNEL_STATUS_OK;
}

uint32_t UniqueConsecutiveCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  auto x_dtype = ctx.Input(0)->GetDataType();
  auto idx_dtype = ctx.Output(1)->GetDataType();
  auto count_dtype = ctx.Output(2)->GetDataType();
  auto input_size = ctx.Input(0)->GetTensorShape()->GetDims();
  if (axis_ == NoneN) {
    switch (idx_dtype) {
      case DT_INT32:
        return DtypeMapNone<int32_t>(ctx, x_dtype);
      case DT_INT64:
        return DtypeMapNone<int64_t>(ctx, x_dtype);
      default:
        KERNEL_LOG_ERROR(
          "[UniqueConsecutive]: Output idx and count data type [%s] "
          "and [%s] not support.",
          DTypeStr(idx_dtype).c_str(), DTypeStr(count_dtype).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    int32_t tmp_axis = MaybeWrapDim(axis_, input_size);
    switch (idx_dtype) {
      case DT_INT32:
        return DtypeMapDim<int32_t>(ctx, tmp_axis, x_dtype);
      case DT_INT64:
        return DtypeMapDim<int64_t>(ctx, tmp_axis, x_dtype);
      default:
        KERNEL_LOG_ERROR(
          "[UniqueConsecutive]: Output idx and count data type [%s] "
          "and [%s] not support.",
          DTypeStr(idx_dtype).c_str(), DTypeStr(count_dtype).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
  }
}

uint32_t UniqueConsecutiveCpuKernel::ExtraParamCheck(CpuKernelContext &ctx) {
  // Check output y.
  Tensor *output_0 = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_0, KERNEL_STATUS_PARAM_INVALID, "Get [y] tensor failed.");
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get [y] data failed.");
  input_type_ = ctx.Input(0)->GetDataType();
  DataType output0_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input_type_ == output0_type), KERNEL_STATUS_PARAM_INVALID,
                     "[UniqueConsecutive]: The data type of output y [%s] need "
                     "be same with input [%s].",
                     DTypeStr(output0_type).c_str(), DTypeStr(input_type_).c_str());
  // Check output idx
  AttrValue *return_idx = ctx.GetAttr("return_idx");
  return_idx_ = (return_idx == nullptr) ? false : (return_idx->GetBool());
  if (return_idx_) {
    Tensor *output_1 = ctx.Output(1);
    KERNEL_CHECK_NULLPTR(output_1, KERNEL_STATUS_PARAM_INVALID, "Get [indices] tensor failed.");
    KERNEL_CHECK_NULLPTR(output_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get [indices] data failed.");
    idx_dtype_ = output_1->GetDataType();
  }
  // Check output counts
  AttrValue *return_counts = ctx.GetAttr("return_counts");
  return_counts_ = (return_counts == nullptr) ? false : (return_counts->GetBool());
  if (return_counts_) {
    Tensor *output_2 = ctx.Output(2);
    KERNEL_CHECK_NULLPTR(output_2, KERNEL_STATUS_PARAM_INVALID, "Get [counts] tensor failed.");
    KERNEL_CHECK_NULLPTR(output_2->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get [counts] data failed.");
    count_dtype_ = output_2->GetDataType();
  }
  // Check idx and counts datatype
  if (return_counts_ && return_idx_) {
    KERNEL_CHECK_FALSE((idx_dtype_ == count_dtype_), KERNEL_STATUS_PARAM_INVALID,
                       "[UniqueConsecutive]: Output idx and count data type "
                       "[%s] and [%s] is not identical.",
                       DTypeStr(idx_dtype_).c_str(), DTypeStr(count_dtype_).c_str())
  }
  return KERNEL_STATUS_OK;
}

uint32_t UniqueConsecutiveCpuKernel::Compute(CpuKernelContext &ctx) {
  // Check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kDynamicOutput),
                      "[UniqueConsecutive]: Check input and output number failed.");
  KERNEL_HANDLE_ERROR(ExtraParamCheck(ctx), "[UniqueConsecutive]: check params failed.");
  // Get the attr
  AttrValue *axis = ctx.GetAttr("axis");
  axis_ = (axis == nullptr) ? NoneN : static_cast<int32_t>(axis->GetInt());
  // Get the inuput and output
  Tensor *input_x = ctx.Input(0);
  auto input_size = input_x->GetTensorShape()->GetDims();
  // Check the axis
  if (input_size == 0 && axis_ != NoneN) {
    KERNEL_LOG_ERROR(
      "[UniqueConsecutive]: axis specified as %d but tensor has no "
      "dimensions.",
      axis_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t min_d = std::min(-input_size, input_size - 1);
  int32_t max_d = std::max(-input_size, input_size - 1);
  if ((axis_ < min_d || axis_ > max_d) && axis_ != NoneN) {
    KERNEL_LOG_ERROR(
      "[UniqueConsecutive]: Axis out of range (expected to be in range of "
      "[%d, %d]).",
      min_d, max_d);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // DoCompute

  return DoCompute(ctx);
}
REGISTER_CPU_KERNEL(kUniqueConsecutive, UniqueConsecutiveCpuKernel);
}  // namespace aicpu
