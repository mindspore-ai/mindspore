/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "mirror_pad.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/equal_util.h"
#include "utils/kernel_util.h"
namespace {
constexpr uint32_t kMirrotPadInputNum = 2;
constexpr uint32_t kMirrotPadOutputNum = 1;
const char *kMirrorPad = "MirrorPad";
constexpr int kMinDims = 0;
constexpr int kMaxDims = 5;
constexpr int kTwo = 2;
std::vector<std::string> attr_names;
std::vector<int64_t> input_dim_shape;
std::vector<int64_t> output_dim_shape;
std::vector<std::pair<int64_t, int64_t>> padding_;
std::vector<uint64_t> input_strides_;
std::vector<uint64_t> output_strides_;
int64_t input_num_elements;
int64_t output_num_elements;
int32_t dims_;
int64_t offset_;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t MirrorPadCpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  // check params
  attr_names.emplace_back("mode");
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMirrotPadInputNum, kMirrotPadOutputNum, attr_names),
                      "[%s] check params failed.", kMirrorPad);
  // get Attr mode
  AttrValue *mode_ptr = ctx.GetAttr("mode");
  auto mode = mode_ptr->GetString();
  KERNEL_CHECK_FALSE((mode == "SYMMETRIC" || mode == "REFLECT"), KERNEL_STATUS_PARAM_INVALID,
                     "Attr mode must be either REFLECT or SYMMETRIC, but got attr mode[%s]", mode);
  if (mode == "SYMMETRIC") {
    offset_ = 0;
  } else if (mode == "REFLECT") {
    offset_ = 1;
  }
  // get input x
  Tensor *x_ptr = ctx.Input(0);
  data_type_ = x_ptr->GetDataType();
  auto x_shape_ptr = x_ptr->GetTensorShape();
  auto dims = x_shape_ptr->GetDims();
  dims_ = x_shape_ptr->GetDims();
  KERNEL_CHECK_FALSE((kMinDims <= dims && dims <= kMaxDims), KERNEL_STATUS_PARAM_INVALID,
                     "inputs rank not in [%lld, %lld]: %lld", kMinDims, kMaxDims, dims);
  // get input paddings
  Tensor *paddings_ptr = ctx.Input(1);
  auto paddings_shape_ptr = paddings_ptr->GetTensorShape();
  KERNEL_CHECK_FALSE((paddings_ptr->GetDataType() == DT_INT32 || paddings_ptr->GetDataType() == DT_INT64),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim data type must be DT_INT32 or DT_INT64, "
                     "but got data type[%s]",
                     DTypeStr(paddings_ptr->GetDataType()).c_str());
  KERNEL_CHECK_FALSE(IsMatrix(paddings_shape_ptr->GetDimSizes()) && paddings_shape_ptr->GetDimSize(1),
                     KERNEL_STATUS_PARAM_INVALID, "paddings must be a matrix with 2 columns: [%lld] ",
                     paddings_shape_ptr->GetDimSizes());
  KERNEL_CHECK_FALSE(dims == paddings_shape_ptr->GetDimSize(0), KERNEL_STATUS_PARAM_INVALID,
                     "The first dimension of paddings must be the rank of inputs [%lld] , "
                     "[%lld]",
                     x_shape_ptr->GetDimSizes(), paddings_shape_ptr->GetDimSizes());
  // Compute the shape of the output tensor, and allocate it.
  auto size_pads_data = reinterpret_cast<T *>(paddings_ptr->GetData());
  input_num_elements = 1;
  output_num_elements = 1;
  for (int d = 0; d < dims_; ++d) {
    int64_t before = *(size_pads_data + d * 2);
    int64_t after = *(size_pads_data + d * 2 + 1);
    padding_.push_back(std::make_pair(before, after));
    KERNEL_CHECK_FALSE(before >= 0 && after >= 0, KERNEL_STATUS_PARAM_INVALID,
                       "paddings must be non-negative: [%lld]  [%lld]", before, after);
    if (offset_ == 0) {
      KERNEL_CHECK_FALSE(before <= x_shape_ptr->GetDimSize(d) && after <= x_shape_ptr->GetDimSize(d),
                         KERNEL_STATUS_PARAM_INVALID,
                         "paddings must be no greater "
                         "than the dimension size: [%lld] , [%lld]  greater than [%lld] ",
                         before, after, x_shape_ptr->GetDimSize(d));
    } else if (offset_ == 1) {
      KERNEL_CHECK_FALSE(before < x_shape_ptr->GetDimSize(d) && after < x_shape_ptr->GetDimSize(d),
                         KERNEL_STATUS_PARAM_INVALID,
                         "paddings must be no greater "
                         "than the dimension size: [%lld] , [%lld]  not less than [%lld] ",
                         before, after, x_shape_ptr->GetDimSize(d));
    }
    input_dim_shape.push_back(x_shape_ptr->GetDimSize(d));
    int64_t dimi = after + x_shape_ptr->GetDimSize(d) + before;
    input_num_elements *= x_shape_ptr->GetDimSize(d);
    output_num_elements *= dimi;
    output_dim_shape.push_back(dimi);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MirrorPadCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input_data_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (output_num_elements == ctx.Input(0)->NumElements() || dims_ == 0) {
    uint64_t copy_size = ctx.Input(0)->GetDataSize();
    auto mem_ret = memcpy_s(output_data, copy_size, input_data_ptr, copy_size);
    KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "Memcpy size[%zu] from input value to output failed.", copy_size);
  } else {
    KERNEL_CHECK_FALSE((MirrorPadCompute<T>(input_data_ptr, output_data) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "MirrorPadCompute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MirrorPadCpuKernel::MirrorPadCompute(T *input_data_ptr, T *output_data_ptr) {
  input_strides_.resize(dims_);
  output_strides_.resize(dims_);
  input_strides_[dims_ - 1] = 1;
  output_strides_[dims_ - 1] = 1;
  for (int i = dims_ - 1; i > 0; --i) {
    input_strides_[i - 1] = input_strides_[i] * input_dim_shape[i];
    output_strides_[i - 1] = output_strides_[i] * output_dim_shape[i];
  }
  std::vector<std::pair<int64_t, int64_t>> index;
  index.resize(dims_);
  index[dims_ - 1] = std::make_pair(output_strides_[dims_ - 1] * padding_[dims_ - 1].first,
                                    output_strides_[dims_ - 1] * padding_[dims_ - 1].second);
  for (int i = dims_ - 1; i > 0; --i) {
    index[i - 1].first = index[i].first + output_strides_[i - 1] * padding_[i - 1].first;
    index[i - 1].second = index[i].second + output_strides_[i - 1] * padding_[i - 1].second;
  }
  if (dims_ == 1) {
    memcpy_s(output_data_ptr, padding_[0].first * sizeof(T), input_data_ptr + offset_, padding_[0].first * sizeof(T));
    memcpy_s(output_data_ptr + padding_[0].first + input_num_elements, padding_[0].second * sizeof(T),
             input_data_ptr + input_num_elements - padding_[0].second - offset_, padding_[0].second * sizeof(T));
    memcpy_s(output_data_ptr + padding_[0].first, input_num_elements * sizeof(T), input_data_ptr,
             input_num_elements * sizeof(T));
    std::reverse(output_data_ptr, output_data_ptr + padding_[0].first);
    std::reverse(output_data_ptr + padding_[0].first + input_num_elements,
                 output_data_ptr + padding_[0].first + input_num_elements + padding_[0].second);
    return KERNEL_STATUS_OK;
  }

  std::vector<int64_t> pos;
  std::vector<int64_t> output_pos, tmp_pos;
  pos.resize(dims_ - 1, 0);
  int64_t output_index = index[0].first;
  int64_t inx = 0, copy_size = sizeof(T) * input_dim_shape[dims_ - 1];
  while (inx < input_num_elements) {
    memcpy_s(output_data_ptr + output_index, copy_size, input_data_ptr + inx, copy_size);
    output_pos.push_back(output_index);
    pos[dims_ - kTwo] += 1;
    int64_t dep = dims_ - 1;
    for (int64_t i = dims_ - 2; i >= 0; --i) {
      if (i > 0 && pos[i] >= input_dim_shape[i]) {
        pos[i] -= input_dim_shape[i];
        pos[i - 1] += 1;
        dep = i;
      } else {
        break;
      }
    }
    output_index += index[dep].first + index[dep].second + input_dim_shape[dims_ - 1];
    inx += input_dim_shape[dims_ - 1];
  }
  for (int64_t i = dims_ - 1; i >= 0; --i) {
    int64_t block_size = output_strides_[i], count = 0;
    copy_size = block_size * sizeof(T);
    for (auto item : output_pos) {
      T *base_output_ptr1 = output_data_ptr + item;
      for (int64_t cnt = 1; cnt <= padding_[i].first; ++cnt) {
        memcpy_s(base_output_ptr1 - cnt * block_size, copy_size, base_output_ptr1 + (cnt - 1 + offset_) * block_size,
                 copy_size);
      }
      T *base_output_ptr2 = output_data_ptr + item + input_dim_shape[i] * block_size;
      for (int64_t cnt = 1; cnt <= padding_[i].second; ++cnt) {
        memcpy_s(base_output_ptr2 + (cnt - 1) * block_size, copy_size, base_output_ptr2 - (cnt + offset_) * block_size,
                 copy_size);
      }
      if (i > 0 && count % input_dim_shape[i - 1] == 0) {
        tmp_pos.push_back(item - padding_[i].first * block_size);
      }
      ++count;
    }
    output_pos.clear();
    for (auto item : tmp_pos) {
      output_pos.push_back(item);
    }
    tmp_pos.clear();
  }
  return KERNEL_STATUS_OK;
}

uint32_t MirrorPadCpuKernel::Compute(CpuKernelContext &ctx) {
  auto padding_type_ = ctx.Input(1)->GetDataType();
  if (padding_type_ == DT_INT32) {
    KERNEL_CHECK_FALSE((CheckAndInitParams<int32_t>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "CheckAndInitParams failed.");
  } else {
    KERNEL_CHECK_FALSE((CheckAndInitParams<int64_t>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "CheckAndInitParams failed.");
  }
  switch (data_type_) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_BOOL:
      return DoCompute<bool>(ctx);
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype[%s]", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kMirrorPad, MirrorPadCpuKernel);
}  // namespace aicpu