/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/conv1d_weight_expanding_pass.h"
#include <memory>
#include <algorithm>
#include <vector>

namespace mindspore::opt {
namespace {
constexpr size_t kTripleNum = 3;
constexpr size_t kConvWeightIndex = 2;
}  // namespace
lite::STATUS Conv1DWeightExpandingPass::ExpandFilterShape(const ParamValueLitePtr &tensor) {
  if (tensor == nullptr) {
    return lite::RET_NULL_PTR;
  }
  auto shape = tensor->tensor_shape();
  std::vector<int> new_shape(shape);
  switch (tensor->format()) {
    case schema::Format_NCHW:
    case schema::Format_KCHW:
      new_shape.insert(new_shape.begin() + 2, 1);
      break;
    case schema::Format_NHWC:
    case schema::Format_KHWC:
      new_shape.insert(new_shape.begin() + 1, 1);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported format.";
      return RET_ERROR;
  }
  tensor->set_tensor_shape(new_shape);
  return RET_OK;
}

bool Conv1DWeightExpandingPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConv2D) && !CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      continue;
    }

    auto conv_cnode = node->cast<CNodePtr>();
    MS_ASSERT(conv_cnode->inputs().size() > kConvWeightIndex);
    auto weight_node = conv_cnode->input(kConvWeightIndex);
    MS_ASSERT(weight_node != nullptr);
    auto weight_value = GetLiteParamValue(weight_node);
    if (weight_value == nullptr) {
      MS_LOG(ERROR) << "weight node must be param value.";
      return false;
    }
    // expand weight tensor to 4 dimensions.
    if (weight_value->tensor_shape().size() == kTripleNum) {
      auto status = ExpandFilterShape(weight_value);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Expand filter shape failed.";
        return false;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::opt
