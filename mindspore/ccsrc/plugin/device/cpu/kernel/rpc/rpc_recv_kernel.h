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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_RECV_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_RECV_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/rpc/rpc_kernel.h"

namespace mindspore {
namespace kernel {
// RpcRecvKernel receives data from another process across network communication. It can not be launched until remote
// data is received and inputs are ready.
class RpcRecvKernelMod : public RpcKernelMod {
 public:
  RpcRecvKernelMod() : recv_monad_(false) {}
  ~RpcRecvKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  void set_real_data_offset(const std::vector<size_t> &real_data_offset) { real_data_offset_ = real_data_offset; }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  // Whether this RpcRecv node receives a monda data.
  bool recv_monad_;

  // When this kernel receives dynamic shape data, the data must be parsed by offset set by RecvActor.
  std::vector<size_t> real_data_offset_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_RECV_KERNEL_H_
