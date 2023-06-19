/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_WKV_GRAD_BISHENG_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_WKV_GRAD_BISHENG_KERNEL_H

#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "plugin/device/ascend/kernel/bisheng/bisheng_kernel_mod.h"

namespace mindspore {
namespace kernel {
class WKVGradBishengKernel : public BiShengKernelMod {
  KernelFunc(WKVGradBishengKernel);

 public:
  WKVGradBishengKernel() = default;
  ~WKVGradBishengKernel() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::string GetOpName() override { return bisheng_name_; };
  TilingFunc GetTilingFunc() override { return tiling_func_; };
  std::vector<WorkSpaceFunc> GetWorkspaceFunc() override { return workspace_func_list_; }
  std::string GetBinary() { return "libwkv"; }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs, void *stream);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_WKV_GRAD_BISHENG_KERNEL_H
