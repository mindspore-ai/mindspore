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

#include "plugin/device/cpu/hal/hardware/cpu_somas.h"
#include <string>
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace cpu {
constexpr size_t ALONE = 1;

bool CPUSomas::Initialize() { return true; }

std::string CPUSomas::GetDeviceName() const { return "CPU"; }

size_t CPUSomas::GetAlignSize(size_t original_size) const {
  constexpr size_t alignment = 512;
  size_t aligned_size = (original_size > 0) ? ((original_size + alignment - 1) / alignment) * alignment : 0;
  return aligned_size;
}

bool CPUSomas::GetDependExecOrderFlag(const session::KernelGraph &graph) const { return false; }

bool CPUSomas::InitDevSpecControlTensors(const session::KernelGraph &graph) { return true; }

bool CPUSomas::DevSpecNodeProcess(const session::KernelGraph &graph) { return true; }

void CPUSomas::CommunicationTensorProcess(const std::vector<somas::SomasTensorPtr> &tensors) const {
  if (tensors.size() != ALONE) {
    size_t all_communication_size = 0;
    for (auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->aligned_size_ = tensor->GetOriginalSize();
      MS_EXCEPTION_IF_CHECK_FAIL(tensor->aligned_size_ != 0, "The size of communication tensor is zero, tensor id: " +
                                                               std::to_string(tensor->GetId()));
      all_communication_size += tensor->aligned_size_;
    }
    auto aligned_communication_size = GetAlignSize(all_communication_size);
    auto need_aligned = aligned_communication_size - all_communication_size;
    MS_EXCEPTION_IF_NULL(tensors[tensors.size() - 1]);
    tensors[tensors.size() - 1]->aligned_size_ += need_aligned;
  }
}

bool CPUSomas::NeedContiguous(const std::vector<size_t> &inputs) const { return inputs.size() > ALONE; }
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
