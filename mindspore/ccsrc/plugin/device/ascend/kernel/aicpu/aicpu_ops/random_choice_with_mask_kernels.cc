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
#include "./random_choice_with_mask_kernels.h"
#include <random>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"
#include "common/distinct_uniform_int_distribution.h"
#include "common/tensor.h"

namespace aicpu {
static void ParseOutputCoordinate(std::vector<int64_t> dims_, int64_t output_length, int64_t input_dim_size,
                                  int64_t input_total_count, const int *tmp_output, int *output) {
  int it = 0;
  int column = LongToInt(input_total_count / dims_[0]);
  for (int64_t i = 0; i < output_length; i++) {
    int tmp_output_number = tmp_output[LongToSize(i)];
    int tmp_column = column;
    for (int64_t j = 0; j < input_dim_size; j++) {
      if (j == input_dim_size - 1) {
        output[it++] = tmp_output_number;
        continue;
      }
      output[it++] = tmp_output_number / column;
      tmp_output_number = tmp_output_number % column;
      tmp_column = tmp_column / LongToInt(dims_[LongToSize(j) + 1]);
    }
  }
}

static void GetOutputLength(bool *padding_flag, int64_t *output_length, int64_t *output_non_zero_length, int64_t count,
                            int64_t non_zero_num) {
  if (count == 0) {
    *padding_flag = false;
    *output_length = non_zero_num;
    *output_non_zero_length = non_zero_num;
  } else if (count > 0 && count <= non_zero_num) {
    *padding_flag = false;
    *output_length = count;
    *output_non_zero_length = count;
  } else if (count > non_zero_num) {
    *padding_flag = true;
    *output_length = count;
    *output_non_zero_length = non_zero_num;
  } else {
    AICPU_LOGI("input count must greater or equal to 0 but instead is %d", count);
  }
}

static bool GetInputTotalCount(const std::vector<int64_t> &dims_, int64_t *input_total_count,
                               const int64_t &input_dim_size) {
  const int64_t max_inpu_dim = 5;
  if (input_dim_size < 1 || input_dim_size > max_inpu_dim) {
    AICPU_LOGE(
      "input dim size is %d, it must greater or equal to 1 channels "
      "and less than or equal to 5 channels!",
      input_dim_size);
    return false;
  }
  for (int64_t i = 0; i < input_dim_size; i++) {
    *input_total_count *= dims_[LongToSize(i)];
  }
  if (*input_total_count <= 0) {
    AICPU_LOGE("input_total_count is %d, please check setting.", *input_total_count);
    return false;
  }
  return true;
}

static void UpdateOutput(const std::vector<int64_t> &dims_, const int64_t &non_zero_num, const int64_t &count_,
                         const int64_t &output_length, const int *mask_dim, int32_t *output_coordinate, bool *mask) {
  const int64_t dim_size = static_cast<int64_t>(dims_.size());
  for (int64_t i = non_zero_num * dim_size; i < count_ * dim_size; i++) {
    output_coordinate[i] = 0;
  }
  for (int64_t i = 0; i < output_length; i++) {
    mask[i] = static_cast<bool>(mask_dim[i]);
  }
  for (int64_t i = non_zero_num; i < count_; i++) {
    mask[i] = false;
  }
}

static bool GenerateRandomMask(const int64_t &output_length, const int64_t &non_zero_num,
                               const int64_t &output_non_zero_length, int **input_dim, int **tmp_output,
                               int **mask_dim) {
  *tmp_output = static_cast<int *>(malloc(LongToSize(output_length) * sizeof(int)));
  if (*tmp_output == nullptr) {
    AICPU_LOGE("malloc memory failed!");
    free(*input_dim);
    return false;
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  aicpu::distinct_uniform_int_distribution<> dis(0, non_zero_num - 1);
  *mask_dim = static_cast<int *>(malloc(LongToSize(output_length) * sizeof(int)));
  if (*mask_dim == nullptr) {
    AICPU_LOGE("malloc memory failed!");
    free(*input_dim);
    free(*tmp_output);
    return false;
  }
  if (memset_s(*mask_dim, output_length, 0x00, output_length) != EOK) {
    AICPU_LOGE("memset_s to mask_dim failed!");
    free(*input_dim);
    free(*tmp_output);
    free(*mask_dim);
    return false;
  }
  if (memset_s(*tmp_output, output_length, 0x00, output_length) != EOK) {
    AICPU_LOGE("memset_s to tmp_output failed!");
    free(*input_dim);
    free(*tmp_output);
    free(*mask_dim);
    return false;
  }

  if (output_non_zero_length > output_length) {
    AICPU_LOGE("output_non_zero_length size is too long!");
    free(*input_dim);
    free(*tmp_output);
    free(*mask_dim);
    return false;
  }
  for (int32_t i = 0; i < output_non_zero_length; i++) {
    int32_t mean = dis.exec(&gen);
    *((*tmp_output) + i) = *((*input_dim) + mean);
    *((*mask_dim) + i) = 1;
  }
  return true;
}

uint32_t RandomChoiceWithMaskKernel::DoCompute() {
  auto *input = reinterpret_cast<bool *>(io_addrs_[0]);
  auto *output_coordinate = reinterpret_cast<int32_t *>(io_addrs_[1]);
  auto *mask = reinterpret_cast<bool *>(io_addrs_[2]);
  int64_t input_dim_size = static_cast<int64_t>(dims_.size());
  int64_t non_zero_num = 0;
  int64_t input_total_count = 1;

  bool ret = GetInputTotalCount(dims_, &input_total_count, input_dim_size);
  if (!ret) {
    AICPU_LOGE("Get input total count failed!");
    return kAicpuKernelStateInternalError;
  }

  int *input_dim = reinterpret_cast<int *>(malloc(input_total_count * sizeof(int)));
  if (input_dim == nullptr) {
    AICPU_LOGE("Malloc memory failed!");
    return kAicpuKernelStateInternalError;
  }
  for (int64_t i = 0; i < input_total_count; i++) {
    if (input[i] != 0) {
      input_dim[LongToSize(non_zero_num)] = LongToInt(i);
      non_zero_num++;
    }
  }
  bool padding_flag = false;
  int64_t output_length = 0;
  int64_t output_non_zero_length = 0;
  GetOutputLength(&padding_flag, &output_length, &output_non_zero_length, count_, non_zero_num);

  int *tmp_output = nullptr;
  int *mask_dim = nullptr;
  ret = GenerateRandomMask(output_length, non_zero_num, output_non_zero_length, &input_dim, &tmp_output, &mask_dim);
  if (!ret) {
    AICPU_LOGE("Generate random mask failed!");
    return kAicpuKernelStateInternalError;
  }

  if (padding_flag) {
    size_t index = 0;
    size_t u_non_zero_num = LongToSize(non_zero_num);
    for (int64_t i = output_length - 1; i > non_zero_num; --i) {
      tmp_output[u_non_zero_num + index] = 0;
      mask_dim[u_non_zero_num + index] = 0;
      ++index;
    }
  }

  size_t copy_output_length = 0;
  if (output_length * input_dim_size >= INT_MAX || output_length * input_dim_size < 0) {
    AICPU_LOGE("Output size exceed INT_MAX");
    free(input_dim);
    free(tmp_output);
    free(mask_dim);
    return kAicpuKernelStateInternalError;
  }
  copy_output_length = LongToSize(output_length * input_dim_size);
  int *output = reinterpret_cast<int *>(malloc(copy_output_length * sizeof(int)));
  if (output == nullptr) {
    AICPU_LOGE("malloc memory failed!");
    free(input_dim);
    free(tmp_output);
    free(mask_dim);
    return kAicpuKernelStateInternalError;
  }
  if (memset_s(output, copy_output_length, 0x00, copy_output_length) != EOK) {
    AICPU_LOGE("memset_s memory failed!");
    free(input_dim);
    free(mask_dim);
    free(tmp_output);
    free(output);
    return kAicpuKernelStateInternalError;
  }
  ParseOutputCoordinate(dims_, output_length, input_dim_size, input_total_count, tmp_output, output);

  int32_t actual_output_length = LongToInt(count_ * input_dim_size);
  int32_t new_output_length = std::min(actual_output_length, SizeToInt(copy_output_length));
  if (INT_MAX / SizeToInt(sizeof(int32_t)) < new_output_length) {
    AICPU_LOGE("The output length is out of range!");
    free(input_dim);
    free(mask_dim);
    free(tmp_output);
    free(output);
    return kAicpuKernelStateInternalError;
  }
  size_t copy_output_bytes = IntToSize(new_output_length) * sizeof(int32_t);
  if (memcpy_s(output_coordinate, copy_output_bytes, output, copy_output_bytes) != EOK) {
    AICPU_LOGE("memcpy_s memory failed!");
    free(input_dim);
    free(mask_dim);
    free(tmp_output);
    free(output);
    return kAicpuKernelStateInternalError;
  }
  UpdateOutput(dims_, non_zero_num, count_, output_length, mask_dim, output_coordinate, mask);
  AICPU_LOGI("no zero num is %d, output_length is %d ", non_zero_num, output_length);
  UpdateOutputShapeValue(non_zero_num, output_length);
  free(input_dim);
  free(mask_dim);
  free(tmp_output);
  free(output);
  return kAicpuKernelStateSucess;
}

void RandomChoiceWithMaskKernel::UpdateOutputShapeValue(int64_t non_zero_num, int64_t output_length) {
  if (unknow_shape_) {
    output_shape_and_type_[0]->dims[0] = non_zero_num;
    output_shape_and_type_[1]->dims[0] = output_length;
  }
}
uint32_t RandomChoiceWithMaskKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> nodedef_map = node_def_.attrs();
  aicpuops::AttrValue random_choice_count_attrs = nodedef_map["count"];
  count_ = random_choice_count_attrs.i();
  AICPU_LOGI("This op attr count is %d", count_);

  if ((count_ == 0) && (!unknow_shape_)) {
    AICPU_LOGE("This op attr count is 0, but the shapetype is %d", unknow_shape_);
    return kAicpuKernelStateInvalid;
  }

  size_t inputs_size = node_def_.inputs_size();
  for (size_t i = 0; i < inputs_size; i++) {
    aicpuops::Tensor input_tensor = node_def_.inputs(i);
    aicpuops::TensorShape input_shape = input_tensor.tensor_shape();
    for (int j = 0; j < input_shape.dim_size(); j++) {
      dims_.push_back(input_shape.dim(j).size());
    }
  }

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t RandomChoiceWithMask(void *param) {
  aicpu::RandomChoiceWithMaskKernel random_choice_with_mask_kernel;
  return random_choice_with_mask_kernel.Compute(param);
}
}
