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

#include "tools/optimizer/format/conv_weight_format.h"
#include <vector>
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
}  // namespace
STATUS ConvWeightFormatBase::ConvWeightFormatTrans(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (ConvWeightFormatTrans(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (ConvWeightFormatTrans(sub_func_graph) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConv2DFusion) &&
        !CheckPrimitiveType(node, opt::kPrimConv2DBackpropInputFusion) &&
        !CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      continue;
    }
    MS_ASSERT(cnode->inputs().size() > kConvWeightIndex);
    auto weight_node = cnode->input(kConvWeightIndex);
    MS_ASSERT(weight_node != nullptr);
    if (utils::isa<CNodePtr>(weight_node)) {
      if (lite::HandleWeightConst(graph, cnode, weight_node->cast<CNodePtr>(), src_format_, dst_format_) !=
          lite::RET_OK) {
        MS_LOG(ERROR) << "handle cnode weight failed.";
        return RET_ERROR;
      }
      continue;
    }
    if (TransferConvWeight(weight_node) != lite::RET_OK) {
      MS_LOG(ERROR) << "transfer weight format failed.";
      return lite::RET_ERROR;
    }
    if (utils::isa<Parameter>(weight_node)) {
      if (lite::HandleWeightSharing(graph, dst_format_, weight_node->cast<ParameterPtr>(), src_format_, dst_format_) !=
          lite::RET_OK) {
        MS_LOG(ERROR) << "handle weight-sharing failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS ConvWeightFormatBase::TransferConvWeight(const AnfNodePtr &weight_node) {
  MS_ASSERT(weight_node != nullptr);
  auto weight_value = GetTensorInfo(weight_node);
  if (weight_value == nullptr) {
    MS_LOG(ERROR) << "weight node must const value";
    return lite::RET_ERROR;
  }
  auto status = TransFilterFormat(weight_value, src_format_, dst_format_);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "trans conv weight failed.";
    return lite::RET_ERROR;
  }
  auto type_id = static_cast<TypeId>(weight_value->data_type());
  auto shape = weight_value->shape();
  std::vector<int64_t> shape_vector(shape.begin(), shape.end());
  auto abstract = lite::CreateTensorAbstract(shape_vector, type_id);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return lite::RET_ERROR;
  }
  weight_node->set_abstract(abstract);
  return lite::RET_OK;
}

bool ConvWeightFormatBase::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  if (src_format_ == dst_format_) {
    return true;
  }
  auto manager = Manage(graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto status = ConvWeightFormatTrans(graph);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Conv2D weight FormatTrans failed: " << status;
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
