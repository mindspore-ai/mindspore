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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_SEND_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_SEND_KERNEL_H_

#include <vector>
#include "plugin/device/cpu/kernel/rpc/rpc_kernel.h"

namespace mindspore {
namespace kernel {
constexpr char kRpcDynamicShapeData[] = "RPC_DYNAMIC_SHAPE_DATA";
// RpcSendKernel send data to another process across network communication.
class RpcSendKernelMod : public RpcKernelMod {
 public:
  RpcSendKernelMod() = default;
  ~RpcSendKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override {
    return true;
  }

  void Init(const CNodePtr &kernel_node) override;
  void InitKernel(const CNodePtr &kernel_node) override { return; }

  std::vector<KernelAttr> GetOpSupport() override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_SEND_KERNEL_H_
