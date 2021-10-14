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

namespace mindspore {
namespace kernel {
void ParseOutputCoordinate(std::vector<int32_t> dims, int32_t output_length, int32_t input_dim_size,
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
      tmp_column = tmp_column / dims[IntToSize(j + 1)];
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

void GetInputTotalCount(const std::vector<int32_t> &dims_, int32_t *input_total_count, const int32_t &input_dim_size) {
  for (size_t i = 0; i < IntToSize(input_dim_size); i++) {
    *input_total_count *= dims_[i];
  }
}

void UpdateOutput(const std::vector<int32_t> &dims_, const int32_t &non_zero_num, const int32_t &count_,
                  const int32_t &output_length, const int *mask_dim, int32_t *output_coordinate, bool *mask) {
  for (int32_t i = non_zero_num * SizeToInt(dims_.size()); i < count_ * SizeToInt(dims_.size()); i++) {
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
  if (input_num != INPUT_NUM) {
    MS_LOG(ERROR) << "Input num is " << input_num << ", but RandomChoiceWithMask needs 1 input.";
  }

  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != OUTPUT_NUM) {
    MS_LOG(ERROR) << "Output num is " << output_num << ", but RandomChoiceWithMask needs 2 outputs.";
  }

  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_shape_size_ = input_shape.size();
  if (input_shape_size_ < 1 || input_shape_size_ > MAX_DIMENSION) {
    MS_LOG(ERROR) << "Input is " << input_shape_size_
                  << "-D, but RandomChoiceWithMask supports only 1-D to 5-D inputs.";
  }

  seed_ = static_cast<size_t>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed"));
  seed2_ = static_cast<size_t>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed2"));
  count_ = static_cast<int>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "count"));

  MS_LOG(INFO) << "This op attr count is " << count_;

  for (size_t i = 0; i < input_num; i++) {
    auto input_i_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
    for (size_t j = 0; j < input_i_shape.size(); j++) {
      (void)dims_.emplace_back(input_i_shape[j]);
    }
  }
  input_dim_size = SizeToInt(dims_.size());
  if (input_dim_size < 1 || input_dim_size > MAX_INPUT_DIMS) {
    MS_LOG(EXCEPTION) << "Input dim size is " << input_dim_size << ", which is not supported.";
  }
}

void RandomChoiceWithMaskCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);

  GetInputTotalCount(dims_, &input_total_count, input_dim_size);
  int temp_output_length = count_ > 0 ? count_ : input_total_count;

  workspace_size_list_.push_back(IntToSize(input_total_count) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * IntToSize(input_dim_size) * sizeof(int));
}

bool RandomChoiceWithMaskCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *input_dim = reinterpret_cast<int *>(workspace[0]->addr);
  auto *tmp_output = reinterpret_cast<int *>(workspace[1]->addr);
  auto *mask_dim = reinterpret_cast<int *>(workspace[2]->addr);
  auto *output = reinterpret_cast<int *>(workspace[3]->addr);
  auto *output_coordinate = reinterpret_cast<int32_t *>(outputs[0]->addr);
  auto *mask = reinterpret_cast<bool *>(outputs[1]->addr);

  size_t seedc = seed2_ != 0 ? seed2_ : (seed_ != 0 ? seed_ : generator_());
  for (int32_t i = 0; i < input_total_count; i++) {
    if (input[i] != 0) {
      input_dim[non_zero_num] = i;
      non_zero_num++;
    }
  }

  GetOutputLength(&padding_flag, &output_length, &output_non_zero_length, count_, non_zero_num);
  (void)memset_s(mask_dim, IntToSize(output_length), 0X00, IntToSize(output_length));
  (void)memset_s(tmp_output, IntToSize(output_length), 0X00, IntToSize(output_length));

  std::vector<int32_t> all_nums(non_zero_num);
  std::iota(begin(all_nums), end(all_nums), 0);
  shuffle(all_nums.begin(), all_nums.end(), std::default_random_engine(seedc));

  for (int32_t i = 0; i < output_non_zero_length; i++) {
    int32_t mean = all_nums[i];
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
  }

  copy_output_length = output_length * input_dim_size;
  (void)memset_s(output, IntToSize(copy_output_length), 0X00, IntToSize(copy_output_length));
  ParseOutputCoordinate(dims_, output_length, input_dim_size, input_total_count, tmp_output, output);

  int32_t actual_output_length = count_ * SizeToInt(dims_.size());
  copy_output_length = std::min(actual_output_length, copy_output_length);
  if (INT_MAX / static_cast<int>(sizeof(int32_t)) < copy_output_length) {
    MS_LOG(EXCEPTION) << "The output length is out of range!";
  }

  size_t copy_output_bytes = IntToSize(copy_output_length) * sizeof(int32_t);
  auto ret = memcpy_s(output_coordinate, outputs[0]->size, output, copy_output_bytes);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s failed, ret = " << ret;
  }
  UpdateOutput(dims_, non_zero_num, count_, output_length, mask_dim, output_coordinate, mask);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
