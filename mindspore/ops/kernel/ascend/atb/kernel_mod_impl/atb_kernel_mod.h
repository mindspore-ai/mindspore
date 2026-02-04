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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_MOD_ATB_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_MOD_ATB_KERNEL_MOD_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/ascend/atb/kernel_mod_impl/atb_adapter.h"
#include "acl/acl.h"
#include "atb/atb_infer.h"

namespace mindspore::kernel {
class ATBKernelMod : public KernelMod {
 public:
  explicit ATBKernelMod(std::string &&kernel_name) : op_name_(std::move(kernel_name)) {}
  ~ATBKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  virtual void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) = 0;
  virtual bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) = 0;

  void UpdateWorkspace(uint64_t workspace_size);
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in ATB."; }

 protected:
  std::string op_name_;
  atb::Operation *op_;
  device::ascend::ParamSetter param_setter_;
  uint64_t hash_id_{0};
};

#define MS_ATB_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_KERNEL_FACTORY_REG(ATBKernelMod, NAME, DERIVE_CLASS)
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_MOD_ATB_KERNEL_MOD_H_
