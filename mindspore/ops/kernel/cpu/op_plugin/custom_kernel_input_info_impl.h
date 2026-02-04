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

#ifndef MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_
#define MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "base/bfloat16.h"
#include "base/float16.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "kernel/cpu/op_plugin/custom_kernel_input_info.h"

namespace mindspore::kernel {
namespace op_plugin {
class KernelInputInfoImpl : public KernelInputInfo {
 public:
  KernelInputInfoImpl() = default;
  virtual ~KernelInputInfoImpl() = default;
  KernelInputInfoImpl(KernelInputInfoImpl &&) = default;
  KernelInputInfoImpl &operator=(KernelInputInfoImpl &&) = default;
  void SetKernelInput(const std::vector<kernel::KernelTensor *> &inputs) { inputs_ = inputs; }
  size_t GetInputSize() { return inputs_.size(); }
  bool IsScalarInput(size_t idx) final { return inputs_[idx]->type_id() != TypeId::kObjectTypeTensorType; }

  bool GetBoolInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<bool>(); }

  // Note: op plugin expects GetIntInput/GetFloatInput to provide implicit casts for scalar values
  // (see comments in ms_op_plugin/op_plugin/utils/op_utils.cc).
  int64_t GetIntInput(size_t idx) {
    const auto dtype = inputs_[idx]->dtype_id();
    switch (dtype) {
      case TypeId::kNumberTypeInt8:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<int8_t>());
      case TypeId::kNumberTypeInt16:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<int16_t>());
      case TypeId::kNumberTypeInt32:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<int32_t>());
      case TypeId::kNumberTypeInt64:
      case TypeId::kNumberTypeInt:
        return inputs_[idx]->GetValueWithCheck<int64_t>();
      case TypeId::kNumberTypeUInt8:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<uint8_t>());
      case TypeId::kNumberTypeUInt16:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<uint16_t>());
      case TypeId::kNumberTypeUInt32:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<uint32_t>());
      case TypeId::kNumberTypeUInt64:
      case TypeId::kNumberTypeUInt:
        return static_cast<int64_t>(inputs_[idx]->GetValueWithCheck<uint64_t>());
      default:
        // Fallback to int64 (will throw with a helpful message if size/type mismatch).
        return inputs_[idx]->GetValueWithCheck<int64_t>();
    }
  }

  double GetFloatInput(size_t idx) {
    const auto dtype = inputs_[idx]->dtype_id();
    switch (dtype) {
      case TypeId::kNumberTypeFloat16:
        return static_cast<double>(inputs_[idx]->GetValueWithCheck<float16>());
      case TypeId::kNumberTypeBFloat16:
        return static_cast<double>(inputs_[idx]->GetValueWithCheck<bfloat16>());
      case TypeId::kNumberTypeFloat32:
      case TypeId::kNumberTypeFloat:
        return static_cast<double>(inputs_[idx]->GetValueWithCheck<float>());
      case TypeId::kNumberTypeFloat64:
      case TypeId::kNumberTypeDouble:
        return inputs_[idx]->GetValueWithCheck<double>();
      default:
        // Fallback to double (will throw with a helpful message if size/type mismatch).
        return inputs_[idx]->GetValueWithCheck<double>();
    }
  }

  std::string GetStrInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<std::string>(); }

  std::vector<int64_t> GetIntVecInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<std::vector<int64_t>>(); }

  std::vector<double> GetFloatVecInput(size_t idx) {
    const auto dtype = inputs_[idx]->dtype_id();
    if (dtype == TypeId::kNumberTypeFloat64 || dtype == TypeId::kNumberTypeDouble) {
      return inputs_[idx]->GetValueWithCheck<std::vector<double>>();
    }
    const auto values = inputs_[idx]->GetValueWithCheck<std::vector<float>>();
    return std::vector<double>(values.begin(), values.end());
  }

  std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) {
    return inputs_[idx]->GetValueWithCheck<std::vector<std::vector<int64_t>>>();
  }

  std::vector<std::vector<double>> GetFloat2DVecInput(size_t idx) {
    const auto dtype = inputs_[idx]->dtype_id();
    if (dtype == TypeId::kNumberTypeFloat64 || dtype == TypeId::kNumberTypeDouble) {
      return inputs_[idx]->GetValueWithCheck<std::vector<std::vector<double>>>();
    }
    const auto values = inputs_[idx]->GetValueWithCheck<std::vector<std::vector<float>>>();
    std::vector<std::vector<double>> result;
    result.reserve(values.size());
    for (const auto &row : values) {
      result.emplace_back(row.begin(), row.end());
    }
    return result;
  }

  int GetInputTypeId(size_t idx) { return static_cast<int>(inputs_[idx]->dtype_id()); }

  std::optional<OpPluginTensorStorageInfo> GetInputTensorLayout(size_t idx) {
    if (inputs_[idx]->type_id() != TypeId::kObjectTypeTensorType) {
      return std::nullopt;
    }
    const auto &input = inputs_[idx];
    if (input->tensor_storage_info() == nullptr) {
      return std::nullopt;
    }
    const auto &strides = input->tensor_storage_info()->strides;
    const auto &storage_offset = input->tensor_storage_info()->storage_offset;
    return std::make_optional<OpPluginTensorStorageInfo>(OpPluginTensorStorageInfo{strides, storage_offset});
  }

  OpPluginOutputShapeInfo *GetOutputShapeInfo() override { return output_shape_info_.get(); }
  void SetOutputShapeInfo(std::unique_ptr<OpPluginOutputShapeInfo> info) override {
    output_shape_info_ = std::move(info);
  }

 private:
  std::vector<kernel::KernelTensor *> inputs_;
  std::unique_ptr<OpPluginOutputShapeInfo> output_shape_info_;
};
}  // namespace op_plugin
}  // namespace mindspore::kernel
#endif  // MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_
