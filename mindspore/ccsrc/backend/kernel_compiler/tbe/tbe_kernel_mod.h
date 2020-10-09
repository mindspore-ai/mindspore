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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "backend/kernel_compiler/ascend_kernel_mod.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"

namespace mindspore {
namespace kernel {
class TbeKernelMod : public AscendKernelMod {
 public:
  explicit TbeKernelMod(KernelPackPtr kernel_pack) : kernel_pack_(std::move(kernel_pack)) {}
  ~TbeKernelMod() override = default;

  void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;
  device::DynamicKernelPtr GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) override;
  std::vector<size_t> GenParameters() override;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

using TbeKernelModPtr = std::shared_ptr<TbeKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TBE_TBE_KERNEL_MOD_H_
