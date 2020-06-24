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

#ifndef MINDSPORE_CCSRC_KERNEL_KERNEL_BUILD_INFO_H_
#define MINDSPORE_CCSRC_KERNEL_KERNEL_BUILD_INFO_H_
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ir/dtype.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
class KernelBuildInfo {
 public:
  class KernelBuildInfoBuilder;

  KernelBuildInfo() {
    kernel_type_ = TBE_KERNEL;
    fusion_type_ = OPAQUE;
    processor_ = AICORE;
    op_pattern_ = kCommonPattern;
    input_reshape_type_ = {};
    output_reshape_type_ = {};
    inputs_format_ = {};
    outputs_format_ = {};
    inputs_device_type_ = {};
    outputs_device_type_ = {};
  }

  ~KernelBuildInfo() = default;

  KernelType kernel_type() const { return kernel_type_; }

  std::string GetInputFormat(size_t input_index) const;

  std::string GetOutputFormat(size_t output_index) const;

  TypeId GetInputDeviceType(size_t input_index) const;

  TypeId GetOutputDeviceType(size_t output_index) const;

  std::vector<Axis> GetInputReshapeType(size_t input_index) const;

  bool IsInputDefaultPadding() const;

  bool IsOutputDefaultPadding() const;

  std::vector<Axis> GetOutputReshapeType(size_t input_index) const;

  std::vector<std::string> GetAllInputFormats() const;

  std::vector<std::string> GetAllOutputFormats() const;

  std::vector<TypeId> GetAllInputDeviceTypes() const;

  std::vector<TypeId> GetAllOutputDeviceTypes() const;

  OpPattern op_pattern() const { return op_pattern_; }

  FusionType fusion_type() const { return fusion_type_; }

  Processor processor() const { return processor_; }

  size_t GetInputNum() const;

  size_t GetOutputNum() const;

  std::string ToString() const;

  bool operator==(const KernelBuildInfo &other) const;

 public:
  static auto constexpr kInvalidFormat = "InvalidFormat";

 private:
  KernelType kernel_type_;
  std::vector<std::string> inputs_format_;
  OpPattern op_pattern_;
  std::vector<std::string> outputs_format_;
  std::vector<std::vector<Axis>> input_reshape_type_;
  std::vector<std::vector<Axis>> output_reshape_type_;
  std::vector<TypeId> inputs_device_type_;
  std::vector<TypeId> outputs_device_type_;
  FusionType fusion_type_;
  Processor processor_;
};
using KernelBuildInfoPtr = std::shared_ptr<KernelBuildInfo>;

class KernelBuildInfo::KernelBuildInfoBuilder {
 public:
  KernelBuildInfoBuilder() { kernel_build_info_ = std::make_shared<KernelBuildInfo>(); }

  explicit KernelBuildInfoBuilder(std::shared_ptr<KernelBuildInfo> kernel_build_info)
      : kernel_build_info_(std::move(kernel_build_info)) {}

  ~KernelBuildInfoBuilder() = default;

  void SetKernelType(const KernelType &kernel_type);

  void SetInputsFormat(const std::vector<std::string> &inputs_format);

  void SetOutputsFormat(const std::vector<std::string> &outputs_format);

  void SetInputsDeviceType(const std::vector<TypeId> &inputs_device_type);

  void SetOutputsDeviceType(const std::vector<TypeId> &outputs_device_type);

  void SetInputReshapeType(const std::vector<std::vector<Axis>> &input_reshape_type);

  void SetOutputReshapeType(const std::vector<std::vector<Axis>> &output_reshape_type);

  void SetFusionType(FusionType fusion_type);

  void SetProcessor(Processor processor);

  void SetOpPattern(OpPattern pattern);

  void SetInputFormat(const std::string &format, size_t index);

  void SetOutputFormat(const std::string &format, size_t index);

  std::shared_ptr<KernelBuildInfo> Build();

 private:
  std::shared_ptr<KernelBuildInfo> kernel_build_info_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_KERNEL_KERNEL_BUILD_INFO_H_
