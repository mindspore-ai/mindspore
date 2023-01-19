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
#include "split.h"
#include "utils/kernel_util.h"

namespace {
const char *kSplit = "Split";
constexpr uint32_t kSplitInputNum = 2;
std::vector<std::string> attr_names;
}  // namespace

namespace aicpu {
uint32_t SplitCpuKernel::CheckAndInitParams(CpuKernelContext &ctx) {
  // check params
  AttrValue *num_split_ptr = ctx.GetAttr("num_split");
  num_split_ = num_split_ptr->GetInt();
  uint32_t kSplitOutputNum = num_split_ptr->GetInt();
  attr_names.emplace_back("num_split");
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kSplitInputNum, kSplitOutputNum, attr_names), "[%s] check params failed.",
                      kSplit);
  KERNEL_CHECK_FALSE((num_split_ >= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Attr num_split must >= 1, but got attr num_split[%lld]", num_split_);
  Tensor *split_dim_ptr = ctx.Input(0);
  auto split_dim_shape_ptr = split_dim_ptr->GetTensorShape();
  KERNEL_CHECK_FALSE((split_dim_shape_ptr->GetDims() == 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim should be a scalar integer, but got rank[%lld]", split_dim_shape_ptr->GetDims());
  KERNEL_CHECK_FALSE((split_dim_ptr->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "Input split_dim data type must be DT_INT32, but got data type[%s]",
                     DTypeStr(split_dim_ptr->GetDataType()).c_str());
  auto split_dim_data_ptr = split_dim_ptr->GetData();
  KERNEL_CHECK_NULLPTR(split_dim_data_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input split_dim data failed.");
  split_dim_ = *(reinterpret_cast<int32_t *>(split_dim_data_ptr));
  Tensor *value_ptr = ctx.Input(1);
  value_data_ptr_ = value_ptr->GetData();
  auto value_shape_ptr = value_ptr->GetTensorShape();
  int64_t value_dim = value_shape_ptr->GetDims();
  if (split_dim_ < 0) {
    split_dim_ += value_dim;
  }
  KERNEL_CHECK_FALSE(value_dim > split_dim_, KERNEL_STATUS_PARAM_INVALID,
                     "Dim of Input value must greater than split_dim, value dim is [%d], split_dim is [%d].", value_dim,
                     num_split_);
  value_shape_vec_ = value_shape_ptr->GetDimSizes();
  data_type_ = value_ptr->GetDataType();
  value_num_ = value_ptr->NumElements();
  KERNEL_CHECK_FALSE((value_shape_ptr->GetDimSize(split_dim_) % num_split_ == 0), KERNEL_STATUS_PARAM_INVALID,
                     "Number of ways to split should evenly divide the split "
                     "dimension, but got split_dim [%d] (size = [%lld]) and num_split is [%lld]",
                     split_dim_, value_shape_ptr->GetDimSize(split_dim_), num_split_);
  output_ptr_vec_.resize(num_split_);
  for (int64_t i = 0; i < num_split_; i++) {
    Tensor *output_ptr = ctx.Output(i);
    auto output_data_ptr = output_ptr->GetData();
    output_ptr_vec_[i] = output_data_ptr;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitCpuKernel::DoCompute(CpuKernelContext &ctx) {
  T *input_data_ptr = static_cast<T *>(value_data_ptr_);
  std::vector<T *> output_data_vec;
  output_data_vec.resize(num_split_);
  for (int64_t i = 0; i < num_split_; i++) {
    output_data_vec[i] = reinterpret_cast<T *>(output_ptr_vec_[i]);
  }

  if (num_split_ == 1) {
    KERNEL_CHECK_FALSE((SplitWithOneOutput<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitWithOneOutput failed.");
    return KERNEL_STATUS_OK;
  }
  if (split_dim_ == 0) {
    KERNEL_CHECK_FALSE((SplitWithDimZero<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitWithDimZero failed.");
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((SplitCompute<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "Split Compute failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitCpuKernel::SplitWithOneOutput(T *input_data_ptr, std::vector<T *> output_data_vec) {
  int64_t copy_size = value_num_ * sizeof(T);
  auto mem_ret = memcpy_s(output_data_vec[0], copy_size, input_data_ptr, copy_size);
  KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                     "Memcpy size[%zu] from input value to output[0] failed.", copy_size);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitCpuKernel::SplitWithDimZero(T *input_data_ptr, std::vector<T *> output_data_vec) {
  int64_t copy_num = value_num_ / value_shape_vec_[0];
  T *input_copy_ptr = input_data_ptr;
  const int64_t split_dim_output_size = value_shape_vec_[0] / num_split_;
  for (int32_t i = 0; i < num_split_; i++) {
    int64_t copy_size_per = copy_num * split_dim_output_size;
    int64_t copy_size = copy_size_per * sizeof(T);
    auto mem_ret = memcpy_s(output_data_vec[i], copy_size, input_copy_ptr, copy_size);
    KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                       "Memcpy size[%zu] from input value to output[%d] failed.", copy_size, i);
    input_copy_ptr += copy_size_per;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitCpuKernel::SplitCompute(T *input_data_ptr, std::vector<T *> output_data_vec) {
  int64_t prefix = 1;
  for (int32_t i = 0; i < split_dim_; ++i) {
    prefix *= value_shape_vec_[i];
  }
  int64_t midfix = value_shape_vec_[split_dim_];
  int64_t subfix = 1;
  for (size_t i = split_dim_ + 1; i < value_shape_vec_.size(); i++) {
    subfix *= value_shape_vec_[i];
  }
  const int64_t split_dim_output_size = midfix / num_split_;
  int64_t offset = 0;
  for (int64_t i = 0; i < num_split_; ++i) {
    T *output_data_ptr = output_data_vec[i];
    T *input_copy_ptr = input_data_ptr + offset;
    int64_t copy_num = subfix * split_dim_output_size;
    int64_t copy_size = copy_num * sizeof(T);
    for (int64_t j = 0; j < prefix; j++) {
      auto mem_ret = memcpy_s(output_data_ptr, copy_size, input_copy_ptr, copy_size);
      KERNEL_CHECK_FALSE((mem_ret == EOK), KERNEL_STATUS_PARAM_INVALID,
                         "Memcpy size[%zu] from input value to output[%d] failed.", copy_size, i);
      input_copy_ptr += (subfix * midfix);
      output_data_ptr += copy_num;
    }
    offset += copy_num;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SplitCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "CheckAndInitParams failed.");
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
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported datatype[%s]", DTypeStr(data_type_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kSplit, SplitCpuKernel);
}  // namespace aicpu
