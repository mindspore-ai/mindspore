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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_BUILD_INFO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_BUILD_INFO_H_
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "kernel/oplib/op_info_keys.h"
#include "ir/dtype.h"
#include "ir/kernel_info_dev.h"
#include "kernel/kernel.h"

#ifdef OPAQUE
#undef OPAQUE
#endif

namespace mindspore {
namespace kernel {
enum KernelObjectType : int {
  UNKNOWN_TYPE = 0,
  TENSOR,
  SCALAR,
  TUPLE,
  TUPLE_UNFOLD,
};

enum OpType : int {
  UNKNOWN_OP_TYPE = 0,
  DYNAMIC,
  SKIP,
};

std::string KernelObjectTypeLabel(const KernelObjectType &obj_type);
std::string KernelTypeLabel(const KernelType &kernel_type);
std::string OpTypeLabel(const OpType &op_type);

class BACKEND_EXPORT KernelBuildInfo {
 public:
  class KernelBuildInfoBuilder;

  KernelBuildInfo() = default;

  ~KernelBuildInfo() = default;

  KernelType kernel_type() const { return kernel_type_; }

  OpType op_type() const { return op_type_; }

  void set_kernel_type(KernelType kernel_type) { kernel_type_ = kernel_type; }

  void set_op_type(OpType op_type) { op_type_ = op_type; }

  std::string GetInputFormat(size_t input_index) const;

  std::string GetOutputFormat(size_t output_index) const;

  TypeId GetInputDeviceType(size_t input_index) const;

  TypeId GetOutputDeviceType(size_t output_index) const;

  KernelObjectType GetInputKernelObjectType(size_t input_index) const;

  KernelObjectType GetOutputKernelObjectType(size_t output_index) const;

  std::string GetInputReshapeType(size_t input_index) const;

  bool IsInputDefaultPadding() const;

  bool IsOutputDefaultPadding() const;

  std::string GetOutputReshapeType(size_t output_index) const;

  const std::string &GetOriginDataFormat() const;

  const std::vector<std::string> &GetAllInputFormats() const;

  const std::vector<std::string> &GetAllOutputFormats() const;

  const std::vector<TypeId> &GetAllInputDeviceTypes() const;

  const std::vector<TypeId> &GetAllOutputDeviceTypes() const;

  const std::vector<KernelObjectType> &GetAllInputKernelObjectTypes() const;

  const std::vector<KernelObjectType> &GetAllOutputKernelObjectTypes() const;

  const std::vector<KernelObjectType> &GetAllOutputElementsKernelObjectTypes() const;

  void SetOpType(const OpType &op_type);

  void SetOutputsKernelObjectType(const std::vector<KernelObjectType> &outputs_kernel_object_type);

  void SetInputsKernelObjectType(const std::vector<KernelObjectType> &inputs_kernel_object_type);

  void SetOutputElementsKernelObjectType(const std::vector<KernelObjectType> &output_elements_kernel_object_type);

  std::vector<std::string> GetAllOutputReshapeType() const;

  std::vector<std::string> GetAllInputReshapeType() const;

  std::string core_type() const { return core_type_; }

  void SetOutputFormat(const std::string &format, size_t index);

  void SetInputFormat(const std::string &format, size_t index);

  void SetOutputDeviceType(const TypeId &output_device_type, size_t index);

  void SetInputsFormat(const std::vector<std::string> &inputs_format);

  void SetOutputsFormat(const std::vector<std::string> &outputs_format);

  void SetInputsDeviceType(const std::vector<TypeId> &inputs_device_type);

  void SetOutputsDeviceType(const std::vector<TypeId> &outputs_device_type);

  OpPattern op_pattern() const { return op_pattern_; }

  std::vector<nlohmann::json> output_data_desc() const { return output_data_desc_; }

  std::string fusion_type() const { return fusion_type_; }

  bool valid() const { return valid_; }
  void set_valid(bool valid) { valid_ = valid; }

  Processor processor() const { return processor_; }
  void set_processor(Processor processor) { processor_ = processor; }

  size_t GetInputNum() const;

  size_t GetOutputNum() const;

  size_t GetOutputNumWithoutMonad() const;

  std::string ToString() const;

  bool IsSimilarityKernelBuildInfo(const KernelBuildInfo &other) const;

  bool operator==(const KernelBuildInfo &other) const;

  bool operator!=(const KernelBuildInfo &other) const;

  static auto constexpr kInvalidFormat = "InvalidFormat";

 private:
  KernelType kernel_type_{UNKNOWN_KERNEL_TYPE};
  OpType op_type_{UNKNOWN_OP_TYPE};
  std::string origin_data_format_{kOpFormat_DEFAULT};
  std::string core_type_;
  std::vector<std::string> inputs_format_;
  OpPattern op_pattern_{kCommonPattern};
  std::vector<std::string> outputs_format_;
  std::vector<std::string> input_reshape_type_;
  std::vector<std::string> output_reshape_type_;
  std::vector<TypeId> inputs_device_type_;
  std::vector<TypeId> outputs_device_type_;
  std::vector<KernelObjectType> inputs_kernel_object_type_;
  std::vector<KernelObjectType> outputs_kernel_object_type_;
  // Indicates kernel object types of elements in TupleUnfold.
  // Only valid when output kernel object type is TupleUnfold.
  std::vector<KernelObjectType> output_elements_kernel_object_type_;
  std::vector<nlohmann::json> output_data_desc_;
  std::string fusion_type_{kernel::kPatternOpaque};
  Processor processor_{UNKNOWN};
  // Indicates whether buildinfo is valid, the invalid buildinfo needs to select kernel again.
  bool valid_{true};
};
using KernelBuildInfoPtr = std::shared_ptr<KernelBuildInfo>;

class BACKEND_EXPORT KernelBuildInfo::KernelBuildInfoBuilder {
 public:
  KernelBuildInfoBuilder() { kernel_build_info_ = std::make_shared<KernelBuildInfo>(); }

  explicit KernelBuildInfoBuilder(const KernelBuildInfoPtr &kernel_build_info)
      : kernel_build_info_(std::make_shared<KernelBuildInfo>()) {
    SetKernelType(kernel_build_info->kernel_type());
    SetOpType(kernel_build_info->op_type());
    SetFusionType(kernel_build_info->fusion_type());
    SetProcessor(kernel_build_info->processor());
    SetOpPattern(kernel_build_info->op_pattern());
    SetCoreType(kernel_build_info->core_type());
    SetOutputDataDesc(kernel_build_info->output_data_desc());
    for (size_t index = 0; index < kernel_build_info->GetInputNum(); ++index) {
      (void)kernel_build_info_->inputs_device_type_.emplace_back(kernel_build_info->GetInputDeviceType(index));
      (void)kernel_build_info_->inputs_format_.emplace_back(kernel_build_info->GetInputFormat(index));
      (void)kernel_build_info_->input_reshape_type_.emplace_back(kernel_build_info->GetInputReshapeType(index));
    }
    kernel_build_info_->inputs_kernel_object_type_ = kernel_build_info->GetAllInputKernelObjectTypes();

    for (size_t index = 0; index < kernel_build_info->GetOutputNum(); ++index) {
      (void)kernel_build_info_->outputs_device_type_.emplace_back(kernel_build_info->GetOutputDeviceType(index));
      (void)kernel_build_info_->outputs_format_.emplace_back(kernel_build_info->GetOutputFormat(index));
      (void)kernel_build_info_->output_reshape_type_.emplace_back(kernel_build_info->GetOutputReshapeType(index));
    }
    kernel_build_info_->outputs_kernel_object_type_ = kernel_build_info->GetAllOutputKernelObjectTypes();
  }

  ~KernelBuildInfoBuilder() = default;

  void SetKernelType(const KernelType &kernel_type);

  void SetOpType(const OpType &op_type);

  void SetOriginDataFormat(const std::string &origin_data_format);

  void SetInputsFormat(const std::vector<std::string> &inputs_format);

  void SetOutputsFormat(const std::vector<std::string> &outputs_format);

  void SetInputsDeviceType(const std::vector<TypeId> &inputs_device_type);

  void SetOutputsDeviceType(const std::vector<TypeId> &outputs_device_type);

  void SetInputsReshapeType(const std::vector<std::string> &input_reshape_type);

  void SetOutputsReshapeType(const std::vector<std::string> &output_reshape_type);

  void SetInputsKernelObjectType(const std::vector<KernelObjectType> &input_kernel_object_type);

  void SetOutputsKernelObjectType(const std::vector<KernelObjectType> &output_kernel_object_type);

  void SetCoreType(const std::string &core_type);

  void SetFusionType(const std::string &fusion_type);
  // save prebuild result
  void SetOutputDataDesc(const std::vector<nlohmann::json> &data_desc);

  void SetProcessor(Processor processor);

  void SetOpPattern(OpPattern pattern);

  void SetInputFormat(const std::string &format, size_t index);

  void SetOutputFormat(const std::string &format, size_t index);

  void SetInputReshapeType(const std::string &input_reshape_type, size_t index);

  void SetOutputReshapeType(const std::string &output_reshape_type, size_t index);

  void SetInputDeviceType(const TypeId &input_device_type, size_t index);

  void SetOutputDeviceType(const TypeId &output_device_type, size_t index);

  void SetValid(bool valid);

  std::shared_ptr<KernelBuildInfo> Build();

 private:
  std::shared_ptr<KernelBuildInfo> kernel_build_info_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_BUILD_INFO_H_
