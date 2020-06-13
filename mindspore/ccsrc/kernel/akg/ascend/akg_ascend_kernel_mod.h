/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include "kernel/ascend_kernel_mod.h"
#include "kernel/tbe/tbe_utils.h"

namespace mindspore {
namespace kernel {
class AkgKernelMod : public AscendKernelMod {
 public:
  explicit AkgKernelMod(const KernelPackPtr &kernel_pack);
  ~AkgKernelMod() final {}

  void SetInputSizeList(const std::vector<size_t> &size_list);
  void SetOutputSizeList(const std::vector<size_t> &size_list);
  void SetWorkspaceSizeList(const std::vector<size_t> &size_list);
  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

using AkgKernelModPtr = std::shared_ptr<AkgKernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
