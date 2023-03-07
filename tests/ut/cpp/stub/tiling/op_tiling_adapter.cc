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

#include <algorithm>
#include "plugin/device/ascend/kernel/tbe/tiling/op_tiling_adapter.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_build.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/ge_types_convert.h"
#include "include/common/utils/utils.h"
#include "external/graph/tensor.h"
#include "external/register/op_tiling_registry.h"
#include "graph/utils/graph_utils.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"

namespace mindspore {
namespace device {
namespace tiling {
constexpr auto COMPILE_INFO_KEY = "compile_info_key";
constexpr auto COMPILE_INFO_JSON = "compile_info_json";
constexpr auto ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
constexpr auto ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
constexpr auto ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";
constexpr auto CONSTANTOP = "Constant";
constexpr auto ATTR_NAME_WEIGHTS = "value";
constexpr auto PARAM_DYNAMIC = "dynamic";

std::string OpTilingCalculateAdapter::GetRealOpType(const std::string &op_type) const {
  static const std::map<std::string, std::string> kOpTypeMap = {
    {"SparseApplyFtrl", "SparseApplyFtrlD"},
    {"SparseApplyProximalAdagrad", "SparseApplyProximalAdagradD"},
    {"SparseGatherV2", "Gather"},
    {"Pad", "PadD"},
    {"Split", "SplitD"},
    {"Concat", "ConcatD"},
    {"Softmax", "SoftmaxV2"},
    {"DropoutDoMask", "DropOutDoMask"},
    {"IOU", "Iou"},
    {"DynamicBroadcastTo", "BroadcastTo"},
    {"DynamicResizeNearestNeighbor", "ResizeNearestNeighborV2"},
    {"ParallelResizeBilinear", "SyncResizeBilinearV2"},
    {"ParallelResizeBilinearGrad", "SyncResizeBilinearV2Grad"},
    {"CeLU", "CeluV2"},
  };
  auto iter = kOpTypeMap.find(op_type);
  if (iter == kOpTypeMap.end()) {
    return op_type;
  }
  return iter->second;
}

std::string OpTilingCalculateAdapter::GetOutputName(const CNodePtr &node, size_t index) { return ""; }

std::string OpTilingCalculateAdapter::GetInputName(const CNodePtr &node, size_t index) { return ""; }

void OpTilingCalculateAdapter::ConvertInputShapeAndType(const CNodePtr &node, ge::OpDescPtr *op_desc) {}

void OpTilingCalculateAdapter::ConvertOutputShapeAndType(const CNodePtr &node, ge::OpDescPtr *op_desc) {}

void OpTilingCalculateAdapter::ConvertCompileInfo(const CNodePtr &node, ge::OpDescPtr *op_desc) {}

ge::NodePtr OpTilingCalculateAdapter::NewConstantOp(const CNodePtr &node, const std::string &name,
                                                    const tensor::TensorPtr &tensor_data, ge::ComputeGraphPtr *ge_graph,
                                                    size_t index) const {
  ge::NodePtr constand_op;
  return constand_op;
}

std::vector<std::tuple<std::size_t, ge::NodePtr>> OpTilingCalculateAdapter::ConvertDepends(
  const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map, ge::OpDescPtr *op_desc,
  ge::ComputeGraphPtr *ge_graph) {
  std::vector<std::tuple<std::size_t, ge::NodePtr>> constant_ops;
  return constant_ops;
}

void OpTilingCalculateAdapter::AddEdge(const ge::NodePtr &ge_node,
                                       const std::vector<std::tuple<std::size_t, ge::NodePtr>> &constant_ops) {}

void OpTilingCalculateAdapter::InitOpIoName(const CNodePtr &node) {}

ge::Operator OpTilingCalculateAdapter::GeNodeToGeOperatorAdapter(const ::ge::NodePtr &ge_node) const {
  ge::Operator op;
  return op;
}

ge::NodePtr OpTilingCalculateAdapter::AnfNodeToGeNodeAdapter(
  const CNodePtr &node, ge::ComputeGraphPtr *ge_graph, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
  const std::string &op_compile_info) {
  ge::NodePtr ge_node;
  return ge_node;
}

ge::Operator OpTilingCalculateAdapter::AnfNodeToGeOperatorAdapter(
  const CNodePtr &node, ::ge::ComputeGraphPtr *ge_graph, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
  const std::string &op_compile_info) {
  ge::Operator op;
  return op;
}

void OpTilingCalculateAdapter::UpdateWorkspace(const ::ge::NodePtr &ge_node,
                                               const std::vector<int64_t> &workspace_size_list) {}
}  // namespace tiling
}  // namespace device
}  // namespace mindspore
