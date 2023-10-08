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
#include "tools/converter/parser/onnx/onnx_pad_adjust.h"
#include <string>
#include <vector>
#include <memory>
#include "mindspore/core/ops/lite_ops.h"
#include "ops/reshape.h"
#include "ops/primitive_c.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr uint32_t kTripleNum = 3;
constexpr uint32_t kQuadraNum = 4;

CNodePtr NewReshapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &shape) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return nullptr;
  }
  auto prim_c = reshape_prim->GetPrim();
  prim_c->set_attr("shape", MakeValue(shape));
  ValueNodePtr value_node = NewValueNode(prim_c);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "create valuenode return nullptr");
  auto new_parameter =
    opt::BuildIntVecParameterNode(func_graph, shape, input_node->fullname_with_scope() + "_reshape/shape");
  MS_CHECK_TRUE_MSG(new_parameter != nullptr, nullptr, "create parameter return nullptr");
  new_parameter->set_name(input_node->fullname_with_scope() + "_reshape/shape");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node, new_parameter};
  auto reshape = func_graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(reshape != nullptr, nullptr, "create cnode return nullptr");
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_reshape");

  std::vector<int64_t> new_shape;
  std::transform(shape.begin(), shape.end(), std::back_inserter(new_shape),
                 [](int x) { return static_cast<int64_t>(x); });
  auto abstract = lite::CreateTensorAbstract(new_shape, kNumberTypeInt32);
  MS_CHECK_TRUE_MSG(abstract != nullptr, nullptr, "Create tensor abstract for reshape failed.");
  reshape->set_abstract(abstract);
  return reshape;
}

bool AdjstVariablePadding(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) {
  MS_CHECK_TRUE_RET(func_graph != nullptr && input_node != nullptr, false);
  // reshape the padding of pad operator to 2 x i.
  std::vector<int> shape_pre = {2, -1};
  auto reshape_pre = NewReshapeOpNode(func_graph, input_node, shape_pre);
  if (reshape_pre == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return false;
  }
  std::vector<int> perm = {1, 0};
  auto transpose =
    opt::GenTransposeNode(func_graph, reshape_pre, perm, reshape_pre->fullname_with_scope() + "_transpose");
  if (transpose == nullptr) {
    MS_LOG(ERROR) << "create transpose failed.";
    return false;
  }
  // reshape the padding of pad operator to -1.
  std::vector<int> shape_pos = {-1};
  auto reshape_pos = NewReshapeOpNode(func_graph, transpose, shape_pos);
  if (reshape_pos == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return false;
  }

  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  if (!manager->Replace(input_node, reshape_pos)) {
    MS_LOG(ERROR) << "Replace node failed.";
    return false;
  }
  return true;
}

bool AdjstConstPadding(const CNodePtr &cnode, const AnfNodePtr &input_node) {
  MS_CHECK_TRUE_RET(cnode != nullptr && input_node != nullptr, false);
  auto func_graph = cnode->func_graph();
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto tensor_info = opt::GetTensorInfo(input_node);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "get tensor info from parameter failed.";
    return false;
  }
  // the data type has been unified into int32.
  int *data = reinterpret_cast<int *>(tensor_info->data_c());
  MS_CHECK_TRUE_RET(tensor_info->data_type() == kNumberTypeInt32 && data != nullptr, false);
  auto data_size = tensor_info->DataSize();
  std::vector<int> padding(data_size);
  for (size_t i = 0; i < data_size / DIMENSION_2D; i++) {
    padding.at(i * DIMENSION_2D) = *(data + i);
    padding.at(i * DIMENSION_2D + 1) = *(data + data_size / DIMENSION_2D + i);
  }
  std::vector<std::vector<int32_t>> padding2d(DIMENSION_2D);
  for (size_t i = 0; i < data_size / DIMENSION_2D; i++) {
    padding2d[Index0].push_back(padding[i]);
    padding2d[Index1].push_back(padding[i + data_size / DIMENSION_2D]);
  }
  auto param_node = opt::BuildIntVec2DParameterNode(func_graph, padding2d, cnode->fullname_with_scope() + "_pads");
  cnode->set_input(Index2, param_node);
  return true;
}
}  // namespace

bool OnnxPadAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimPadFusion) ||
        (cnode->inputs().size() != kTripleNum && cnode->inputs().size() != kQuadraNum)) {
      continue;
    }
    // get the second input node whose output is the padding parameter of pad.
    // and adjust padding: begin1, begin2, begin3... end1, end2, end3... to begin1, end1, begin2, end2, begin3, end3...
    auto input_node = cnode->input(2);
    MS_CHECK_TRUE_RET(input_node != nullptr, false);
    auto param_input = input_node->cast<ParameterPtr>();
    if (param_input != nullptr && param_input->has_default()) {
      if (!AdjstConstPadding(cnode, input_node)) {
        MS_LOG(ERROR) << "Adjust paddings for node: " << cnode->fullname_with_scope() << " failed.";
        return false;
      } else {
        continue;
      }
    }
    if (!AdjstVariablePadding(func_graph, input_node)) {
      MS_LOG(ERROR) << "Adjust paddings for node: " << cnode->fullname_with_scope() << " failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
