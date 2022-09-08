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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"

namespace mindspore {
namespace kernel {
class AclKernelMod : public AscendKernelMod {
 public:
  AclKernelMod() = default;
  explicit AclKernelMod(const AnfNodePtr &anf_node_ptr) : AscendKernelMod(anf_node_ptr) {}
  ~AclKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, uint32_t) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  void SetNodeName(const std::string &node_name) { node_name_ = node_name; }

 protected:
  void SyncData() override;

 private:
  std::string node_name_;
};

using AclKernelModPtr = std::shared_ptr<AclKernelMod>;
using AclKernelModPtrList = std::vector<AclKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
