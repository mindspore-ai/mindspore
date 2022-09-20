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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_GETNEXT_DYNAMIC_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_GETNEXT_DYNAMIC_H

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/ascend/kernel/rts/rt_kernel.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_info.h"

namespace mindspore {
namespace kernel {
class GetNextDynamic : public RtKernel {
 public:
  GetNextDynamic();
  ~GetNextDynamic() override;

  bool Init(const AnfNodePtr &anf_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &others = std::map<uint32_t, tensor::TensorPtr>()) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, uint32_t) override {
    std::vector<TaskInfoPtr> res;
    return res;
  }
};

class GetNextDynamicDesc : public RtKerDesc {
 public:
  GetNextDynamicDesc();
  ~GetNextDynamicDesc() override;
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> GetKernelInfo(const CNodePtr &kernel_node) override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RTS_GETNEXT_DYNAMIC_H
