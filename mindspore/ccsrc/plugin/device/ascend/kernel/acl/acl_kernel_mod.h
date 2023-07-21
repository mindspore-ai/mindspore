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
#include <string>
#include "ops/base_operator.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
#include "runtime/pynative/op_runtime_info.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
using TensorParams = transform::TensorParams;

class AclKernelMod : public AscendKernelMod {
 public:
  AclKernelMod() {
    if (converter_ == nullptr) {
      converter_ = std::make_shared<transform::AclConverter>();
    }
  }
  explicit AclKernelMod(const AnfNodePtr &anf_node_ptr) : AscendKernelMod(anf_node_ptr) {
    if (converter_ == nullptr) {
      converter_ = std::make_shared<transform::AclConverter>();
    }
  }
  ~AclKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, uint32_t) override;

  int Resize(
    const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  void SetDeviceInfo(const std::vector<std::string> &input_device_formats,
                     const std::vector<std::string> &output_device_formats,
                     const std::vector<TypeId> &input_device_types, const std::vector<TypeId> &output_device_types);
  bool IsNeedRetrieveOutputShape() override;

  void PackageInput(const size_t idx, const std::string &format, ShapeVector *shape);
  void PackageOutput(const size_t idx, const ShapeVector &shape);
  void SetPrimitive(const PrimitivePtr &primitive);
  void SetNeedConvertHostTensor(const bool convert_flag) { need_convert_host_tensor_ = convert_flag; }
  void CreateAclConverter();

 protected:
  std::string DebugString() const;
  void SyncOutputShape() override;
  void GetInputInfo(const std::vector<KernelTensorPtr> &inputs);
  int GetOutputInfo(const std::vector<KernelTensorPtr> &outputs);

 private:
  std::vector<TensorParams> input_params_;
  std::vector<TensorParams> output_params_;
  transform::AclInputToHost input_to_host_array_;

  PrimitivePtr primitive_ptr_;
  std::string kernel_name_;

  std::vector<std::string> ms_attr_str_;
  transform::AclConverterPtr converter_;

  bool need_convert_host_tensor_{false};
};

using AclKernelModPtr = std::shared_ptr<AclKernelMod>;
using AclKernelModPtrList = std::vector<AclKernelModPtr>;
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACL_ACL_KERNEL_MOD_H_
