/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_DYNAMIC_AICPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_DYNAMIC_AICPU_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_ext_info_handle.h"
namespace mindspore {
namespace kernel {
class DynamicAicpuOpKernelMod : public AicpuOpKernelMod {
 public:
  DynamicAicpuOpKernelMod() : unknow_type_(device::ascend::UnknowShapeOpType::DEPEND_IN_SHAPE) {}
  explicit DynamicAicpuOpKernelMod(const AnfNodePtr &anf_node_ptr);
  ~DynamicAicpuOpKernelMod() noexcept override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  void SyncData() override;

 private:
  void AllocateExtInfoDeviceAddr(const CNodePtr &cnode);
  void UpdateOutputShapeFromExtInfo(const CNodePtr &cnode);

  std::shared_ptr<device::ascend::AicpuExtInfoHandler> ext_info_handler_ = nullptr;
  size_t ext_info_size_ = 0;
  device::ascend::UnknowShapeOpType unknow_type_;
  void *stream_ = nullptr;
};

using DynamicAicpuOpKernelModPtr = std::shared_ptr<DynamicAicpuOpKernelMod>;
using DynamicAicputOpKernelModPtrList = std::vector<DynamicAicpuOpKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AICPU_DYNAMIC_AICPU_KERNEL_MOD_H_
