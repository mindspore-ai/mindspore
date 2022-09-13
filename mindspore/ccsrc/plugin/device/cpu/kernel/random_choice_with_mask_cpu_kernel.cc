/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/random_choice_with_mask_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/random_choice_with_mask.h"

namespace mindspore {
namespace kernel {
constexpr char kKernelName[] = "RandomChoiceWithMask";

void ParseOutputCoordinate(std::vector<int32_t> dims_, int32_t output_length_, int32_t input_dim_size_,
                           int32_t input_total_count_, const int *tmp_output, int *output) {
  int it = 0;
  int column = input_total_count_ / dims_[0];
  for (int i = 0; i < output_length_; i++) {
    int32_t tmp_output_number = tmp_output[i];
    int tmp_column = column;
    for (int j = 0; j < input_dim_size_; j++) {
      if (j == input_dim_size_ - 1) {
        output[it++] = tmp_output_number;
        continue;
      }
      output[it++] = tmp_output_number / column;
      tmp_output_number = tmp_output_number % column;
      tmp_column = tmp_column / dims_[IntToSize(j + 1)];
    }
  }
}

void GetOutputLength(bool *padding_flag_, int32_t *output_length_, int32_t *output_non_zero_length_, int32_t count_,
                     int32_t non_zero_num_) {
  if (count_ == 0) {
    *padding_flag_ = false;
    *output_length_ = non_zero_num_;
    *output_non_zero_length_ = non_zero_num_;
  } else if (count_ > 0 && count_ <= non_zero_num_) {
    *padding_flag_ = false;
    *output_length_ = count_;
    *output_non_zero_length_ = count_;
  } else if (count_ > non_zero_num_) {
    *padding_flag_ = true;
    *output_length_ = count_;
    *output_non_zero_length_ = non_zero_num_;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kKernelName << "', the 'count' must be greater than or equal to 0, but got "
                      << count_;
  }
}

void GetInputTotalCount(const std::vector<int32_t> &dims_, int32_t *input_total_count_,
                        const int32_t &input_dim_size_) {
  for (size_t i = 0; i < IntToSize(input_dim_size_); i++) {
    *input_total_count_ *= dims_[i];
  }
}

void UpdateOutput(const std::vector<int32_t> &dims_, const int32_t &non_zero_num_, const int32_t &count_,
                  const int32_t &output_length_, const int *mask_dim_, int32_t *output_coordinate_, bool *mask_) {
  for (int32_t i = non_zero_num_ * SizeToInt(dims_.size()); i < count_ * SizeToInt(dims_.size()); i++) {
    output_coordinate_[i] = 0;
  }
  for (int32_t i = 0; i < output_length_; i++) {
    mask_[i] = static_cast<bool>(mask_dim_[i]);
  }
  for (int32_t i = non_zero_num_; i < count_; i++) {
    mask_[i] = false;
  }
}

bool RandomChoiceWithMaskCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  auto random_choice_with_mask_ptr = std::dynamic_pointer_cast<ops::RandomChoiceWithMask>(base_operator);
  seed = random_choice_with_mask_ptr->get_seed();
  seed2 = random_choice_with_mask_ptr->get_seed2();
  count = random_choice_with_mask_ptr->get_count();
  return true;
}

int RandomChoiceWithMaskCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  for (size_t i = 0; i < x_shape.size(); i++) {
    (void)dims.emplace_back(LongToInt(x_shape[i]));
  }
  input_dim_size = SizeToInt(dims.size());
  GetInputTotalCount(dims, &input_total_count, input_dim_size);
  int temp_output_length = count > 0 ? count : input_total_count;

  workspace_size_list_.push_back(IntToSize(input_total_count) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * sizeof(int));
  workspace_size_list_.push_back(IntToSize(temp_output_length) * IntToSize(input_dim_size) * sizeof(int));
  return KRET_OK;
}

bool RandomChoiceWithMaskCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &workspace,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *input_dim = reinterpret_cast<int *>(workspace[0]->addr);
  auto *tmp_output = reinterpret_cast<int *>(workspace[1]->addr);
  auto *mask_dim = reinterpret_cast<int *>(workspace[2]->addr);
  auto *output = reinterpret_cast<int *>(workspace[3]->addr);
  auto *output_coordinate = reinterpret_cast<int32_t *>(outputs[0]->addr);
  auto *mask = reinterpret_cast<bool *>(outputs[1]->addr);

  size_t seedc = seed2 != 0 ? seed2 : (seed != 0 ? seed : generator_());
  int32_t non_zero_num = 0;
  for (int32_t i = 0; i < input_total_count; i++) {
    if (input[i] != 0) {
      input_dim[non_zero_num] = i;
      non_zero_num++;
    }
  }

  bool padding_flag = false;
  int32_t output_length = 0;
  int32_t output_non_zero_length = 0;
  GetOutputLength(&padding_flag, &output_length, &output_non_zero_length, count, non_zero_num);
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

  if (output_length * input_dim_size >= INT_MAX || output_length * input_dim_size < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output size exceed INT_MAX.";
  }

  int32_t copy_output_length = output_length * input_dim_size;
  (void)memset_s(output, IntToSize(copy_output_length), 0X00, IntToSize(copy_output_length));
  ParseOutputCoordinate(dims, output_length, input_dim_size, input_total_count, tmp_output, output);

  int32_t actual_output_length = count * SizeToInt(dims.size());
  copy_output_length = std::min(actual_output_length, copy_output_length);
  if (INT_MAX / static_cast<int>(sizeof(int32_t)) < copy_output_length) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', output length is out of range.";
  }

  size_t copy_output_bytes = IntToSize(copy_output_length) * sizeof(int32_t);
  auto ret = memcpy_s(output_coordinate, outputs[0]->size, output, copy_output_bytes);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s failed. Error no: " << ret;
  }
  UpdateOutput(dims, non_zero_num, count, output_length, mask_dim, output_coordinate, mask);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomChoiceWithMask, RandomChoiceWithMaskCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
