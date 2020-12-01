/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_SEND_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_SEND_H
#include <memory>
#include <vector>
#include "backend/kernel_compiler/rts/rt_kernel.h"
#include "backend/kernel_compiler/rts/rt_kernel_info.h"

namespace mindspore {
namespace kernel {
class SendKernel : public RtKernel {
 public:
  SendKernel();
  ~SendKernel() override;
  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;

 private:
  uint32_t event_id_;
};

MS_REG_RTKERNEL(streamsend, SendKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_SEND_H
