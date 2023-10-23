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

#include "plugin/device/gpu/kernel/debug/assert_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
bool AssertGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  summarize_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("summarize")));
  return true;
}

int AssertGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  auto inputs_size = inputs.size();
  if (inputs_size < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be greater than or equal to 1, but got "
                  << inputs_size;
    return KRET_RESIZE_FAILED;
  }

  auto input_data_size = inputs_size - 1;
  summarizes_.resize(input_data_size);
  types_.resize(input_data_size);
  for (size_t i = 1; i < inputs_size; i++) {
    auto input_type_id = inputs[i]->dtype_id();
    types_[i - 1] = static_cast<int>(input_type_id);
    auto element = inputs[i]->size() / abstract::TypeIdSize(input_type_id);
    summarizes_[i - 1] = static_cast<int>(std::min(static_cast<size_t>(summarize_), element));
  }
  input_addrs_.resize(input_data_size);
  workspace_size_list_.clear();
  workspace_size_list_.emplace_back(sizeof(void *) * input_data_size);
  workspace_size_list_.emplace_back(sizeof(int) * input_data_size);
  workspace_size_list_.emplace_back(sizeof(int) * input_data_size);

  return KRET_OK;
}

MS_REG_GPU_KERNEL(Assert, AssertGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
