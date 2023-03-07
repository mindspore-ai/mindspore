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

#ifndef MINDSPORE_CCSRC_plugin_DEVICE_CPU_HAL_HARDWARE_CPU_SOMAS_H__
#define MINDSPORE_CCSRC_plugin_DEVICE_CPU_HAL_HARDWARE_CPU_SOMAS_H__

#include <string>
#include <vector>
#include "backend/common/somas/somas.h"
#include "include/backend/device_type.h"

namespace mindspore {
namespace device {
namespace cpu {
using KernelGraph = session::KernelGraph;
class CPUSomas : public somas::Somas {
 private:
  bool Initialize() override;
  string GetDeviceName() const override;
  size_t GetAlignSize(size_t original_size) const override;
  void CommunicationTensorProcess(const std::vector<somas::SomasTensorPtr> &tensors) const override;
  bool NeedContiguous(const std::vector<size_t> &inputs) const override;

  bool GetDependExecOrderFlag(const session::KernelGraph &graph) const override;
  bool InitDevSpecControlTensors(const session::KernelGraph &graph) override;
  bool DevSpecNodeProcess(const session::KernelGraph &graph) override;
};
REG_SOMAS(CPU, DeviceType::kCPU, CPUSomas)
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_plugin_DEVICE_CPU_HAL_HARDWARE_CPU_SOMAS_H__
