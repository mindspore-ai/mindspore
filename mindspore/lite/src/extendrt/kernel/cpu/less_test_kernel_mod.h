/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_LESS_TEST_KERNEL_MOD_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_LESS_TEST_KERNEL_MOD_H_

#include <vector>
#include <string>

#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "kernel/common_utils.h"

namespace mindspore::kernel {
class LessTestKernelMod : public CpuKernelMod {
 public:
  LessTestKernelMod() = default;
  ~LessTestKernelMod() override = default;

  explicit LessTestKernelMod(const std::string name) { kernel_name_ = name; }

  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr);

  virtual bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs);
  std::vector<KernelAttr> GetOpSupport() override { return {}; }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_LESS_TEST_KERNEL_MOD_H_
