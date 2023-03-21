/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "include/backend/device_address.h"
#include "include/backend/kernel_info.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::runtime {
class AclRuntimeInfo {
 public:
  AclRuntimeInfo() : is_dynamic_input_size_(true), is_dynamic_output_size_(true), use_(false) {}
  void SetUse(bool flag) { use_ = flag; }
  void SetIsDynamicInputSize(bool flag) {
    CheckInUse();
    is_dynamic_input_size_ = flag;
  }
  void SetIsDynamicOutputSize(bool flag) {
    CheckInUse();
    is_dynamic_output_size_ = flag;
  }
  void SetInputNames(std::vector<std::string> input_names) {
    CheckInUse();
    input_names_ = std::move(input_names);
  }
  void SetOutputNames(std::vector<std::string> output_names) {
    CheckInUse();
    output_names_ = std::move(output_names);
  }

  bool use() const { return use_; }
  bool is_dynamic_input_size() {
    CheckInUse();
    return is_dynamic_input_size_;
  }
  bool is_dynamic_output_size() {
    CheckInUse();
    return is_dynamic_output_size_;
  }
  const std::vector<std::string> &input_names() {
    if (is_dynamic_input_size()) {
      MS_LOG(EXCEPTION) << "This node has dynamic_input_size, should not get AclRuntimeInfo.";
    }
    return input_names_;
  }
  const std::vector<std::string> &output_names() {
    if (is_dynamic_output_size()) {
      MS_LOG(EXCEPTION) << "This node has dynamic_output_size, should not get AclRuntimeInfo.";
    }
    return output_names_;
  }

 private:
  void CheckInUse() {
    if (!use()) {
      MS_LOG(EXCEPTION) << "AclRuntimeInfo is not in use.";
    }
  }
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool is_dynamic_input_size_;
  bool is_dynamic_output_size_;
  bool use_;
};
using AclRuntimeInfoPtr = std::shared_ptr<AclRuntimeInfo>;

class BACKEND_EXPORT OpRuntimeInfo {
 public:
  OpRuntimeInfo(std::vector<std::string> output_format, std::vector<TypeId> output_type,
                std::vector<size_t> output_tensor_size, std::vector<ShapeVector> output_infer_shape,
                std::vector<ShapeVector> output_device_shape, device::KernelInfo *kernel_info,
                std::vector<std::pair<device::KernelInfo *, size_t>> input_kernel_infos)
      : acl_runtime_info_(std::make_shared<AclRuntimeInfo>()),
        output_format_(std::move(output_format)),
        output_type_(std::move(output_type)),
        output_tensor_size_(std::move(output_tensor_size)),
        output_infer_shape_(std::move(output_infer_shape)),
        output_device_shape_(std::move(output_device_shape)),
        kernel_info_(kernel_info),
        input_kernel_infos_(std::move(input_kernel_infos)) {}
  ~OpRuntimeInfo() = default;

  // Key for user data.
  constexpr static char key[] = "OpRuntimeInfo";

  std::string output_format(size_t index) const;
  TypeId output_type(size_t index) const;
  size_t output_tensor_size(size_t index) const;
  const ShapeVector &output_infer_shape(size_t index) const;
  const ShapeVector &output_device_shape(size_t index) const;
  void SetOutputTensorSize(size_t index, size_t tensor_size);
  void SetOutputInferShape(size_t index, const ShapeVector &shape);
  void SetOutputDeviceShape(size_t index, const ShapeVector &shape);
  device::DeviceAddressPtr GetOutputDeviceAddress(size_t index) const;
  device::DeviceAddressPtr GetWorkspaceDeviceAddress(size_t index) const;
  device::DeviceAddressPtr GetInputDeviceAddress(size_t index) const;
  size_t GetInputSize() const;
  size_t GetOutputSize() const;
  size_t GetWorkspaceSize() const;
  void Resize(const AnfNodePtr &node);

  static void CacheGraphOpRuntimeInfo(const KernelGraphPtr &graph);
  // for acl
  AclRuntimeInfoPtr acl_runtime_info_;

 private:
  std::vector<std::string> output_format_;
  std::vector<TypeId> output_type_;
  std::vector<size_t> output_tensor_size_;
  std::vector<ShapeVector> output_infer_shape_;
  std::vector<ShapeVector> output_device_shape_;
  device::KernelInfo *kernel_info_;
  std::vector<std::pair<device::KernelInfo *, size_t>> input_kernel_infos_;
};
using OpRuntimeInfoPtr = std::shared_ptr<OpRuntimeInfo>;
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_RUN_OP_OP_RUNTIME_INFO_H_
