/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RES_LIMIT_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RES_LIMIT_H
#include <memory>
#include <vector>
#include <map>
#include "plugin/ascend/kernel_executor/rts/rt_kernel.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mindspore {
namespace kernel {
struct ResLimitInfo {
  aclrtDevResLimitType type;
  uint32_t core_num;
  uint32_t stream_id;
};

class ResLimitKernel : public RtKernel {
 public:
  ResLimitKernel() = default;
  ~ResLimitKernel() override;
  bool Init(const AnfNodePtr &anf_node) override;
  int Resize(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override;
  bool Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &, void *stream_ptr) override;

 private:
  std::vector<ResLimitInfo> res_limit_infos_;
  bool is_dyn_graph_ = false;
  bool is_exec_resize_ = false;
};

MS_REG_RTKERNEL(reslimit, ResLimitKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_RES_LIMIT_H
