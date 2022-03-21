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

#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/rpc/rpc_kernel.h"

namespace mindspore {
namespace kernel {
// RpcRecvKernel receives data from another process across network communication. It should not be launched until remote
// data is received and inputs are ready.
class RpcRecvKernelMod : public RpcKernelMod {
 public:
  RpcRecvKernelMod() : recv_monad_(false) {}
  ~RpcRecvKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    if (!kernel_func_) {
      MS_LOG(EXCEPTION) << "Kernel func pointer is not initialized!";
    }
    return kernel_func_(this, inputs, workspace, outputs);
  }

  void InitKernel(const CNodePtr &kernel_node) override {
    auto input0 = common::AnfAlgo::GetInputNode(kernel_node, 0);
    // If the input is a monad, no need to launch recv kernel.
    if (HasAbstractUMonad(input0) || HasAbstractIOMonad(input0)) {
      recv_monad_ = true;
    }

    auto kernel_attr = GetKernelAttrFromNode(kernel_node);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(EXCEPTION) << "RpcRecv does not support this kernel data type: " << kernel_attr;
    }
    kernel_func_ = func_list_[index].second;
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs) {
    if (recv_monad_) {
      MS_LOG(DEBUG) << "RpcRecv has a monad as input, no need to launch it.";
      return true;
    }

    MS_EXCEPTION_IF_NULL(remote_input_);
    T *recv_data = GetDeviceAddress<T>(outputs, 0);
    int ret = memcpy_s(recv_data, outputs[0]->size, remote_input_->Body().data(), remote_input_->Body().size());
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s for recv output failed, ret code: " << ret;
    }

    return true;
  }
  using RpcRecvFunc =
    std::function<bool(RpcRecvKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, RpcRecvFunc>> func_list_;
  RpcRecvFunc kernel_func_;

  // Whether this RpcRecv node receives a monda data.
  bool recv_monad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_RECV_KERNEL_H_
