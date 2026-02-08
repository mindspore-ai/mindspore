/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#include "ops/base_operator.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "include/utils/anfalgo.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
using TensorParams = device::ascend::TensorParams;

class OPS_ASCEND_API AclKernelMod : public KernelMod {
 public:
  // =========================================New interface==========================================================
  AclKernelMod() {}
  ~AclKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  // The input addresses that are not used in the kernel launch.
  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override;

  // Some kernels, e.g., Send/Print, output is use less.
  std::vector<size_t> GetUseLessOutputIdx() const override;

  void SetDeviceInfo(const std::vector<std::string> &input_device_formats,
                     const std::vector<std::string> &output_device_formats,
                     const std::vector<TypeId> &input_device_types, const std::vector<TypeId> &output_device_types);

  bool IsNeedUpdateOutputShapeAndSize() override {
    MS_EXCEPTION_IF_NULL(converter_);
    return converter_->is_need_retrieve_output_shape();
  }

  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  void PackageInput(const size_t idx, const std::string &format, ShapeVector *shape);
  void PackageOutput(const size_t idx, const ShapeVector &shape);
  void RefreshAclConverter(const std::vector<KernelTensor *> &inputs);
  virtual void SetValueDependArgs(const std::set<int64_t> &indices);
  void SetDynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }
  std::string GetFormatFromInput(const std::vector<KernelTensor *> &inputs);

  const std::set<int64_t> &GetValueDependArgs() const { return value_depend_args_; }
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in ACL."; }

 protected:
  std::string DebugString() const;
  void GetInputInfo(const std::vector<KernelTensor *> &inputs);
  int GetOutputInfo(const std::vector<KernelTensor *> &outputs);

  bool is_dynamic_{true};

  std::vector<std::string> ms_attr_str_;
  device::ascend::AclConverterPtr converter_;
  std::vector<TensorParams> output_params_;

  std::vector<TensorParams> input_params_;
  // record indices of value depend arguments
  std::set<int64_t> value_depend_args_;
};

using AclKernelModPtr = std::shared_ptr<AclKernelMod>;
using AclKernelModPtrList = std::vector<AclKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
