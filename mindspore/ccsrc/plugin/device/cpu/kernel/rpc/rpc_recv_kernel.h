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
  RpcRecvKernelMod() : recv_monad_(false), is_dynamic_shape_(false) {}
  ~RpcRecvKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override {
    if (recv_monad_) {
      MS_LOG(DEBUG) << "RpcRecv has a monad as input, no need to launch it.";
      return true;
    }

    MS_EXCEPTION_IF_NULL(remote_input_);
    // If the string body is not empty, it means we need to copy data from 'body' instead of raw pointer 'data'.
    bool use_string_msg = !remote_input_->Body().empty();
    auto data_ptr = use_string_msg ? (remote_input_->Body().data()) : (static_cast<char *>(remote_input_->data));
    size_t data_size = use_string_msg ? (remote_input_->Body().size()) : (remote_input_->size);

    if (is_dynamic_shape_) {
      if (real_data_offset_.empty()) {
        MS_LOG(EXCEPTION) << "Dynamic shape data must have data offsets to copy from source message.";
      }
      for (size_t i = 0; i < inputs.size(); i++) {
        MS_EXCEPTION_IF_NULL(inputs[i]->addr);
        int ret = memcpy_s(inputs[i]->addr, inputs[i]->size, data_ptr + real_data_offset_[i], inputs[i]->size);
        if (ret != 0) {
          MS_LOG(EXCEPTION) << "memcpy_s for recv output " << i << " failed, ret code: " << ret;
        }
      }
    } else {
      size_t offset = 0;
      for (size_t i = 0; i < inputs.size(); i++) {
        MS_EXCEPTION_IF_NULL(inputs[i]->addr);
        int ret = memcpy_s(inputs[i]->addr, inputs[i]->size, data_ptr + offset, inputs[i]->size);
        if (ret != 0) {
          MS_LOG(EXCEPTION) << "memcpy_s for recv output failed, ret code: " << ret;
        }

        offset += inputs[i]->size;
        // Maybe the size of data from remote is smaller than inputs size, need to break in advance to avoid illegal
        // memory access. For example, the 'umonad' inputs of RpcRecvKernel is not sent from remote.
        // This should be fixed in graph optimizing step.
        if (offset == data_size) {
          break;
        }
      }
    }

    // Pay attention that the remote_input_ is a pointer of MessageBase which is allocated as heap memory by rpc module.
    // We need to delete it after launching kernel.
    delete remote_input_;
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  void set_real_data_offset(const std::vector<size_t> &real_data_offset) { real_data_offset_ = real_data_offset; }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  // Whether this RpcRecv node receives a monda data.
  bool recv_monad_;

  // When this kernel receives dynamic shape data, the data must be parsed by offset set by RecvActor.
  std::vector<size_t> real_data_offset_;

  // Whether this kernel receives dynamic shape data.
  bool is_dynamic_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RPC_RPC_RECV_KERNEL_H_
