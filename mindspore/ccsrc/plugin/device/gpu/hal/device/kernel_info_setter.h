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
#include "include/common/utils/utils.h"
#include "kernel/kernel.h"
#include "kernel/kernel_build_info.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/kernel_graph.h"
#include "kernel/common_utils.h"
#include "ops/core_ops.h"
#include "utils/ms_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace gpu {
const size_t kAllPositions = SIZE_MAX;
const size_t kFormatTransformDimension = 4;

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
  {kBatchNormOpName, {{0}, {0}}},
  {kBatchNormWithActivation, {{0}, {0}}},
  {kBatchNormWithAddAndActivation, {{0, 5}, {0}}},
  {kBatchNormGradOpName, {{0, 1}, {0}}},
  {kBatchNormGradWithActivation, {{0, 1, 7}, {0}}},
  {kBatchNormGradWithAddAndActivation, {{0, 1, 7}, {0, 3}}},
  {kBiasAddOpName, {{0}, {0}}},
  {prim::kPrimBiasAddGrad->name(), {{0}, {}}},
  // Format insensitive.
  {prim::kPrimReLU->name(), {{0}, {0}}},
  {prim::kPrimReluGrad->name(), {{0, 1}, {0}}},
  {prim::kPrimReLU6->name(), {{0}, {0}}},
  {prim::kPrimReLU6Grad->name(), {{0, 1}, {0}}},
  {kSliceOpName, {{0}, {0}}},
  {kSliceGradOpName, {{0, 1}, {0}}},
  {kTensorAddOpName, {{0, 1}, {0}}},
  {prim::kPrimConcat->name(), {{kAllPositions}, {0}}},
  {prim::kPrimAddN->name(), {{kAllPositions}, {0}}},
  {prim::kPrimSplit->name(), {{0}, {kAllPositions}}},
};

std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &kernel_node,
                                                           KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);

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

class GPU_EXPORT GPUGraphKernelInfo : public GraphKernelInfo {
 public:
  GPUGraphKernelInfo() = default;
  virtual ~GPUGraphKernelInfo() = default;
  void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) override;
};

REG_GRAPH_KERNEL_INFO(kGPUDevice, GPUGraphKernelInfo);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_KERNEL_INFO_SETTER_H_
