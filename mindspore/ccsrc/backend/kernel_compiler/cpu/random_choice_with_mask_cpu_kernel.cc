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

#include "backend/kernel_compiler/cpu/random_choice_with_mask_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#define BLOCKSIZE 256
#define MAX_DIMENSION 5

namespace mindspore {
namespace kernel {

void ParseOutputCoordinate(std::vector<int64_t> dims, int32_t output_length, int32_t input_dim_size,
                           int32_t input_total_count, const int *tmp_output, int *output) {
  int it = 0;
  int column = input_total_count / dims[0];
  for (int i = 0; i < output_length; i++) {
    int32_t tmp_output_number = tmp_output[i];
    int tmp_column = column;
    for (int j = 0; j < input_dim_size; j++) {
      if (j == input_dim_size - 1) {
        output[it++] = tmp_output_number;
        continue;
      }
      output[it++] = tmp_output_number / column;
      tmp_output_number = tmp_output_number % column;
      tmp_column = tmp_column / dims[j + 1];
    }
  }
}

void GetOutputLength(bool *padding_flag, int32_t *output_length, int32_t *output_non_zero_length, int32_t count,
                     int32_t non_zero_num) {
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
    MS_LOG(EXCEPTION) << "Input count must be greater than or equal to 0, but is " << count;
  }
}

void GetInputTotalCount(const std::vector<int64_t> &dims_, int32_t *input_total_count, const int32_t &input_dim_size) {
  for (int32_t i = 0; i < input_dim_size; i++) {
    *input_total_count *= dims_[i];
  }
}

void UpdateOutput(const std::vector<int64_t> &dims_, const int32_t &non_zero_num, const int32_t &count_,
                  const int32_t &output_length, const int *mask_dim, int32_t *output_coordinate, bool *mask) {
  for (int32_t i = non_zero_num * dims_.size(); i < static_cast<int32_t>(count_ * dims_.size()); i++) {
    output_coordinate[i] = 0;
  }
  for (int32_t i = 0; i < output_length; i++) {
    mask[i] = static_cast<bool>(mask_dim[i]);
  }
  for (int32_t i = non_zero_num; i < count_; i++) {
    mask[i] = false;
  }
}

void RandomChoiceWithMaskCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(ERROR) << "Input num is " << input_num << ", but RandomChoiceWithMask needs 1 input.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(ERROR) << "Output num is " << output_num << ", but RandomChoiceWithMask needs 2 outputs.";
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_shape_size_ = input_shape.size();
  if (input_shape_size_ < 1 || input_shape_size_ > MAX_DIMENSION) {
    MS_LOG(ERROR) << "Input is " << input_shape_size_
                  << "-D, but RandomChoiceWithMask supports only 1-D to 5-D inputs.";
  }

  seed_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed"));
  seed2_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed2"));
  count_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "count"));

  MS_LOG(INFO) << "This op attr count is " << count_;

  for (size_t i = 0; i < input_num; i++) {
    auto input_i_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
    for (size_t j = 0; j < input_i_shape.size(); j++) {
      dims_.emplace_back(input_i_shape[j]);
    }
  }
}

bool RandomChoiceWithMaskCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *output_coordinate = reinterpret_cast<int32_t *>(outputs[0]->addr);
  auto *mask = reinterpret_cast<bool *>(outputs[1]->addr);
  int32_t input_dim_size = dims_.size();
  int32_t non_zero_num = 0;
  int32_t input_total_count = 1;

  if (input_dim_size < 1 || input_dim_size > 5) {
    MS_LOG(EXCEPTION) << "Input dim size is " << input_dim_size << ", which is not supported.";
  }

  int seedc = seed2_ != 0 ? seed2_ : (seed_ != 0 ? seed_ : generator_());
  GetInputTotalCount(dims_, &input_total_count, input_dim_size);
  int *input_dim = new (std::nothrow) int[input_total_count];
  if (input_dim == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc memory failed!";
    return false;
  }
  for (int32_t i = 0; i < input_total_count; i++) {
    if (input[i] != 0) {
      input_dim[non_zero_num] = i;
      non_zero_num++;
    }
  }

  bool padding_flag = false;
  int32_t output_length = 0;
  int32_t output_non_zero_length = 0;
  GetOutputLength(&padding_flag, &output_length, &output_non_zero_length, count_, non_zero_num);
  int *tmp_output = new (std::nothrow) int[output_length];
  if (tmp_output == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc memory failed!";
    delete[] input_dim;
    return false;
  }

  std::mt19937 gen(seedc);
  std::uniform_int_distribution<> dis(0, non_zero_num - 1);
  int *mask_dim = new (std::nothrow) int[output_length];
  if (mask_dim == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc memory failed!";
    delete[] input_dim;
    delete[] tmp_output;
    return false;
  }
  (void)memset_s(mask_dim, output_length, 0X00, output_length);
  (void)memset_s(tmp_output, output_length, 0X00, output_length);

  for (int32_t i = 0; i < output_non_zero_length; i++) {
    int32_t mean = dis(gen);
    tmp_output[i] = input_dim[mean];
    mask_dim[i] = 1;
  }
  if (padding_flag) {
    int32_t index = 0;
    for (int32_t i = output_length - 1; i > non_zero_num; i--) {
      tmp_output[non_zero_num + index] = 0;
      mask_dim[non_zero_num + index] = 0;
      index++;
    }
  }

  int32_t copy_output_length = 0;
  if (output_length * input_dim_size >= INT_MAX || output_length * input_dim_size < 0) {
    MS_LOG(EXCEPTION) << "Output size exceed INT_MAX";
    delete[] input_dim;
    delete[] tmp_output;
    delete[] mask_dim;
    return false;
  }

  copy_output_length = output_length * input_dim_size;
  int *output = new (std::nothrow) int[copy_output_length];
  if (output == nullptr) {
    MS_LOG(EXCEPTION) << "Malloc memory failed!";
    delete[] input_dim;
    delete[] tmp_output;
    delete[] mask_dim;
    return false;
  }
  (void)memset_s(output, copy_output_length, 0X00, copy_output_length);
  ParseOutputCoordinate(dims_, output_length, input_dim_size, input_total_count, tmp_output, output);

  int32_t actual_output_length = count_ * dims_.size();
  copy_output_length = std::min(actual_output_length, copy_output_length);
  int32_t copy_output_bytes = 0;
  if (INT_MAX / static_cast<int>(sizeof(int32_t)) < copy_output_length) {
    MS_LOG(EXCEPTION) << "The output length is out of range!";
    delete[] input_dim;
    delete[] mask_dim;
    delete[] tmp_output;
    delete[] output;
    return false;
  }

  copy_output_bytes = copy_output_length * sizeof(int32_t);
  memcpy_s(output_coordinate, copy_output_bytes, output, copy_output_bytes);
  UpdateOutput(dims_, non_zero_num, count_, output_length, mask_dim, output_coordinate, mask);
  delete[] input_dim;
  delete[] mask_dim;
  delete[] tmp_output;
  delete[] output;

  return true;
}

}  // namespace kernel
}  // namespace mindspore
