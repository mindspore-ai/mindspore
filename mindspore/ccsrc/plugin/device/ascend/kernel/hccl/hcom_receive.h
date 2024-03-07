/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_RECEIVE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_RECEIVE_H_

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "plugin/device/ascend/kernel/hccl/hccl_kernel.h"
#include "include/backend/distributed/rpc/rpc_server_base.h"
#include "distributed/cluster/actor_route_table_proxy.h"

namespace mindspore {
namespace kernel {
class HcomReceiveKernel : public HcclKernel {
 public:
  HcomReceiveKernel() = default;
  ~HcomReceiveKernel() override;

  /* Inherit from kernelmod */
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }

 private:
  std::vector<int64_t> real_shape_;
  bool is_dynamic_shape_ = false;
  bool get_shape_attr_flag_ = false;
  std::unique_ptr<mindspore::distributed::rpc::RPCServerBase> server_ = nullptr;
  MessageBase *HandleMessage(MessageBase *const msg);
  int ReceiveShapeForDynamic();
  std::mutex mtx;
  std::condition_variable cv;
};

MS_HCCL_REG_KERNEL(Receive, HcomReceiveKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_RECEIVE_H_
