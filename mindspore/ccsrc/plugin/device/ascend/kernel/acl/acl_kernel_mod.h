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
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "runtime/pynative/op_runtime_info.h"
#include "transform/acl_ir/acl_convert.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
using TensorParams = transform::TensorParams;

class AclKernelMod : public KernelMod {
 public:
  // =========================================New interface==========================================================
  AclKernelMod() {}
  ~AclKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
              const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;

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
  void SetNeedConvertHostTensor(const bool convert_flag) { need_convert_host_tensor_ = convert_flag; }
  void CreateAclConverter();
  void SetValueDependArgs(const std::set<int64_t> &indices) { value_depend_args_ = indices; }
  std::string GetFormatFromInput(const std::vector<KernelTensor *> &inputs);

  const std::set<int64_t> &GetValueDependArgs() const { return value_depend_args_; }
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in ACL."; }

  // =======================Old interface, will deleted after all kernel modified used new interface=================
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    MS_LOG(EXCEPTION) << "This interface is discarded.";
  }

 protected:
  std::string DebugString() const;
  void GetInputInfo(const std::vector<KernelTensor *> &inputs);
  int GetOutputInfo(const std::vector<KernelTensor *> &outputs);

 private:
  std::vector<TensorParams> input_params_;
  std::vector<TensorParams> output_params_;
  // record indices of value depend arguments
  std::set<int64_t> value_depend_args_;
  // inputs of operator
  const std::vector<KernelTensor *> *inputs_ = nullptr;

  std::vector<std::string> ms_attr_str_;
  transform::AclConverterPtr converter_;

  bool need_convert_host_tensor_{false};
  transform::AclInputToHost input_to_host_array_;
};

using AclKernelModPtr = std::shared_ptr<AclKernelMod>;
using AclKernelModPtrList = std::vector<AclKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
