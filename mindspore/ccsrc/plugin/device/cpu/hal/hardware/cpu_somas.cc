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
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
