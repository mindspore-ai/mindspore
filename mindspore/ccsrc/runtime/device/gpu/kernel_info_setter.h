/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include <map>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "utils/utils.h"
#include "frontend/operator/ops.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace gpu {
const size_t kAllPositions = SIZE_MAX;

// Map<opName, (inputFormatPosition, outputFormatPosition)>, used for getting the inserted position of format transform.
// If the inserted position is kAllPositions, then insert all the positions, because the input or output numbers of
// this op are variable.
static std::map<std::string, std::pair<std::vector<size_t>, std::vector<size_t>>> kKernelFormatPositionMap = {
  // Format sensitive.
  {prim::kPrimConv2D->name(), {{0, 1}, {0}}},
  {prim::kPrimConv2DBackpropInput->name(), {{0, 1}, {0}}},
  {prim::kPrimConv2DBackpropFilter->name(), {{0, 1}, {0}}},
  {prim::kPrimMaxPool->name(), {{0}, {0}}},
  {prim::kPrimMaxPoolGrad->name(), {{0, 1, 2}, {0}}},
  {kAvgPoolOpName, {{0}, {0}}},
  {kAvgPoolGradOpName, {{0, 1, 2}, {0}}},
  {kBatchNorm, {{0}, {0}}},
  {kBatchNormWithActivation, {{0}, {0}}},
  {kBatchNormWithAddAndActivation, {{0, 5}, {0}}},
  {kBatchNormGradOpName, {{0, 1}, {0}}},
  {kBatchNormGradWithActivation, {{0, 1, 7}, {0}}},
  {kBatchNormGradWithAddAndActivation, {{0, 1, 7}, {0, 3}}},
  {kBiasAddOpName, {{0}, {0}}},
  {prim::kPrimBiasAddGrad->name(), {{0}, {}}},
  // Format insensitive.
  {prim::kPrimRelu->name(), {{0}, {0}}},
  {prim::kPrimReluGrad->name(), {{0, 1}, {0}}},
  {prim::kPrimRelu6->name(), {{0}, {0}}},
  {prim::kPrimRelu6Grad->name(), {{0, 1}, {0}}},
  {kSliceOpName, {{0}, {0}}},
  {kTensorAddOpName, {{0, 1}, {0}}},
  {prim::kPrimConcat->name(), {{kAllPositions}, {0}}},
  {prim::kPrimAddN->name(), {{kAllPositions}, {0}}},
};

void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);

class FormatTransformChecker {
 public:
  void CheckSupportFormatTransform(const std::shared_ptr<session::KernelGraph> &kernel_graph);
  bool format_transform() const { return format_transform_; }

  static FormatTransformChecker &GetInstance() {
    static FormatTransformChecker instance;
    return instance;
  }

 private:
  FormatTransformChecker() = default;
  ~FormatTransformChecker() = default;
  FormatTransformChecker(const FormatTransformChecker &);
  FormatTransformChecker &operator=(const FormatTransformChecker &);

  bool format_transform_{true};
};

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
