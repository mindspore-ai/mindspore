/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_GRAPH_KERNEL_PACKET_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_GRAPH_KERNEL_PACKET_KERNEL_MOD_H_

#include <vector>
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
class OPS_ASCEND_API KernelPacketAscendKernelMod : public KernelPacketKernelMod {
 public:
  using KernelPacketKernelMod::KernelPacketKernelMod;
  ~KernelPacketAscendKernelMod() = default;

 protected:
  bool CopyHostToDevice(void *dst, const void *src, size_t size, void *stream) override;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_GRAPH_KERNEL_PACKET_KERNEL_MOD_H_
