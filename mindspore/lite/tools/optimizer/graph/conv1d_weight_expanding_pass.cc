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
#include "nnacl/op_base.h"

namespace mindspore::opt {
namespace {
constexpr size_t kTripleNum = 3;
constexpr size_t kConvWeightIndex = 2;
}  // namespace
lite::STATUS Conv1DWeightExpandingPass::ExpandFilterShape(const AnfNodePtr &weight_node, const schema::Format &format) {
  MS_ASSERT(weight_node != nullptr);
  auto weight_tensor = GetTensorInfo(weight_node);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "weight node must be param value.";
    return lite::RET_ERROR;
  }
  auto shape = weight_tensor->shape();
  if (shape.size() != kTripleNum) {
    return lite::RET_OK;
  }
  std::vector<int64_t> new_shape(shape);
  switch (format) {
    case schema::Format_NCHW:
    case schema::Format_KCHW:
      // expand the 'w' dimension.
      new_shape.insert(new_shape.begin() + 2, 1);
      break;
    case schema::Format_NHWC:
    case schema::Format_KHWC:
      // expand the 'w' dimension.
      new_shape.insert(new_shape.begin() + 1, 1);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported format.";
      return RET_ERROR;
  }
  weight_tensor->set_shape(new_shape);
  if (!utils::isa<ParameterPtr>(weight_node)) {
    return lite::RET_OK;
  }
  auto weight_param = weight_node->cast<ParameterPtr>();
  MS_ASSERT(weight_param != nullptr);
  auto type = weight_tensor->data_type();
  weight_param->set_abstract(std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), new_shape));
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

    auto prim = GetValueNode<PrimitivePtr>(conv_cnode->input(0));
    MS_CHECK_TRUE_MSG(prim != nullptr, false, "GetValueNode failed");
    schema::Format schema_format = schema::Format::Format_KCHW;
    if (prim->GetAttr(ops::kFormat) != nullptr) {
      schema_format = static_cast<schema::Format>(GetValue<int64_t>(prim->GetAttr(ops::kFormat)));
    }
    // expand weight tensor to 4 dimensions.
    auto status = ExpandFilterShape(weight_node, schema_format);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Expand filter shape failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
