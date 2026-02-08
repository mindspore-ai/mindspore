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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include "kernel/ascend/akg/kernel_mod_impl/akg_utils.h"

using AkgKernelManager = mindspore::kernel::akg::KernelManager;
using AkgKernelManagerPtr = std::shared_ptr<AkgKernelManager>;

namespace mindspore {
namespace kernel {
class AkgKernelMod : public KernelMod {
 public:
  explicit AkgKernelMod(const KernelPackPtr &kernel_pack, const AnfNodePtr &anf_node_ptr);
  ~AkgKernelMod() final {}
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  std::vector<size_t> GenParameters() override;
  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in akg kernel module.";
  }

  static AkgKernelManagerPtr kernel_manager_;

 private:
  KernelPackPtr kernel_pack_;
  std::vector<std::vector<size_t>> args_remap_;
};

using AkgKernelModPtr = std::shared_ptr<AkgKernelMod>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_MOD_H_
