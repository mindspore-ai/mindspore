/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"

namespace mindspore {
namespace kernel {
class TbeKernelMod : public AscendKernelMod {
 public:
  explicit TbeKernelMod(KernelPackPtr kernel_pack) : kernel_pack_(std::move(kernel_pack)) {}
  TbeKernelMod(KernelPackPtr kernel_pack, const AnfNodePtr &anf_node_ptr)
      : AscendKernelMod(anf_node_ptr), kernel_pack_(std::move(kernel_pack)) {}
  ~TbeKernelMod() override = default;

  void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;
  std::vector<size_t> GenParameters() override;
  AddressPtr GetOverflowAddress();
  void GetRealIOAddress(const AnfNodePtr &cnode, const std::vector<AddressPtr> &inputs,
                        const std::vector<AddressPtr> &outputs, std::vector<AddressPtr> *real_inputs,
                        std::vector<AddressPtr> *real_outputs) const;

 protected:
  KernelPackPtr kernel_pack_;
};

using TbeKernelModPtr = std::shared_ptr<TbeKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_
