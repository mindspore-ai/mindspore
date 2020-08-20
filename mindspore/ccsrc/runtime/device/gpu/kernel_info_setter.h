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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_KERNEL_INFO_SETTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_KERNEL_INFO_SETTER_H_

#include <utility>
#include <string>
#include <vector>
#include <map>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "utils/utils.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace device {
namespace gpu {
// map<opName, (inputFormatPosition, outputFormatPosition)>, used for getting the insert position of format transform.
static std::map<std::string, std::pair<std::vector<size_t>, std::vector<size_t>>> kKernelFormatPositionMap = {
  {prim::kPrimConv2D->name(), {{0, 1}, {0}}},
  {prim::kPrimConv2DBackpropInput->name(), {{0, 1}, {0}}},
  {prim::kPrimConv2DBackpropFilter->name(), {{0, 1}, {0}}},
  {prim::kPrimRelu->name(), {{0}, {0}}},
  {prim::kPrimReluGrad->name(), {{0, 1}, {0}}},
  {prim::kPrimMaxPool->name(), {{0}, {0}}},
  {prim::kPrimMaxPoolGrad->name(), {{0, 1, 2}, {0}}},
  {kAvgPoolOpName, {{0}, {0}}},
  {kAvgPoolGradGpuOpName, {{0, 1, 2}, {0}}},
  {kTensorAddOpName, {{0, 1}, {0}}},
  {kFusedBatchNormEx, {{0}, {0}}},
  {kFusedBatchNormExWithActivation, {{0}, {0}}},
  {kFusedBatchNormExWithAddAndActivation, {{0, 5}, {0}}},
  {kFusedBatchNormGradEx, {{0, 1}, {0}}},
  {kFusedBatchNormGradExWithActivation, {{0, 1, 7}, {0}}},
  {kFusedBatchNormGradExWithAddAndActivation, {{0, 1, 7}, {0, 3}}},
};

void SetKernelInfo(const CNodePtr &apply_kernel_ptr);

class KernelAttr {
 public:
  using DataType = std::pair<TypeId, std::string>;
  KernelAttr() : all_same_(false) {}
  ~KernelAttr() = default;

  KernelAttr &AddInputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT) {
    input_type_.emplace_back(ms_type, format);
    return *this;
  }

  KernelAttr &AddOutputAttr(const TypeId &ms_type, const std::string &format = kOpFormat_DEFAULT) {
    output_type_.emplace_back(ms_type, format);
    return *this;
  }

  KernelAttr &AddAllSameAttr(const bool &all_same) {
    all_same_ = all_same;
    return *this;
  }

  const DataType &GetInputAttr(const size_t index) const { return input_type_[index]; }
  const DataType &GetOutputAttr(const size_t index) const { return output_type_[index]; }
  const bool &GetAllSame() const { return all_same_; }

  size_t GetInputSize() const { return input_type_.size(); }
  size_t GetOutputSize() const { return output_type_.size(); }

 private:
  std::vector<DataType> input_type_;
  std::vector<DataType> output_type_;
  bool all_same_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_KERNEL_INFO_SETTER_H_
