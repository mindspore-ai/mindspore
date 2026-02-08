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

#ifndef MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_H_
#define MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_H_

#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace mindspore::kernel {
namespace op_plugin {
struct OpPluginTensorStorageInfo {
  std::vector<int64_t> strides;
  size_t storage_offset;
};

struct OpPluginOutputShapeInfo {
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<bool> shape_calculated;
};

// KernelInputInfo is an interface class.
// There is also a copy of the same code in the ms_op_plugin repository.
// Both sides should be consistent and neither side's code should be modified separately.
class KernelInputInfo {
 public:
  KernelInputInfo() = default;
  virtual ~KernelInputInfo() = default;
  KernelInputInfo(KernelInputInfo &&) = default;
  KernelInputInfo &operator=(KernelInputInfo &&) = default;
  virtual bool IsScalarInput(size_t idx) = 0;

  template <typename T>
  inline T GetKernelInput(size_t) const {
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  virtual size_t GetInputSize() = 0;

  virtual bool GetBoolInput(size_t idx) = 0;
  virtual int64_t GetIntInput(size_t idx) = 0;
  virtual double GetFloatInput(size_t idx) = 0;
  virtual std::string GetStrInput(size_t idx) = 0;

  virtual std::vector<int64_t> GetIntVecInput(size_t idx) = 0;
  virtual std::vector<double> GetFloatVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<double>> GetFloat2DVecInput(size_t idx) = 0;
  virtual int GetInputTypeId(size_t idx) = 0;
  virtual std::optional<OpPluginTensorStorageInfo> GetInputTensorLayout(size_t idx) = 0;

  virtual OpPluginOutputShapeInfo *GetOutputShapeInfo() = 0;
  virtual void SetOutputShapeInfo(std::unique_ptr<OpPluginOutputShapeInfo>) = 0;

  std::vector<size_t> workspace_;
};
}  // namespace op_plugin
}  // namespace mindspore::kernel
#endif  // MINDSPORE_OPS_KERNEL_CPU_OP_PLUGIN_CUSTOM_KERNEL_INPUT_INFO_H_
