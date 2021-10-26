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
#include "ops/reshape.h"
#include "ops/transpose.h"
#include "ops/primitive_c.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr uint32_t kTripleNum = 3;
constexpr uint32_t kQuadraNum = 4;

ParameterPtr CreateNewParameter(const FuncGraphPtr &func_graph, const std::vector<int> &data) {
  MS_ASSERT(func_graph != nullptr);
  auto parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(parameter != nullptr, nullptr);
  ShapeVector shape_vector;
  shape_vector.push_back(static_cast<int64_t>(data.size()));
  if (INT_MUL_OVERFLOW_THRESHOLD(data.size(), sizeof(int), SIZE_MAX)) {
    MS_LOG(ERROR) << "data_size overflow";
    return nullptr;
  }
  size_t size = data.size() * sizeof(int);
  auto tensor_info = lite::CreateTensorInfo(data.data(), size, shape_vector, kNumberTypeInt32);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return nullptr;
  }

  auto status = lite::InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return parameter;
}

CNodePtr NewReshapeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &shape) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input_node != nullptr);
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "create reshape failed.";
    return nullptr;
  }
  reshape_prim->set_attr("shape", MakeValue(shape));
  ValueNodePtr value_node = NewValueNode(reshape_prim);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "create valuenode return nullptr");
  auto new_parameter = CreateNewParameter(func_graph, shape);
  MS_CHECK_TRUE_MSG(new_parameter != nullptr, nullptr, "create parameter return nullptr");
  new_parameter->set_name(input_node->fullname_with_scope() + "_reshape/shape");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node, new_parameter};
  auto reshape = func_graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(reshape != nullptr, nullptr, "create cnode return nullptr");
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_reshape");
  return reshape;
}

CNodePtr NewTransposeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                            const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input_node != nullptr);
  auto transpose_prim = std::make_shared<ops::Transpose>();
  if (transpose_prim == nullptr) {
    MS_LOG(ERROR) << "create transpose failed.";
    return nullptr;
  }
  transpose_prim->set_attr("perm", MakeValue(perm));
  ValueNodePtr value_node = NewValueNode(transpose_prim);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "create valuenode return nullptr");
  auto new_parameter = CreateNewParameter(func_graph, perm);
  MS_CHECK_TRUE_MSG(new_parameter != nullptr, nullptr, "create parameter return nullptr");
  new_parameter->set_name(input_node->fullname_with_scope() + "_transpose/perm");
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node, new_parameter};
  auto reshape = func_graph->NewCNode(op_inputs);
  MS_CHECK_TRUE_MSG(reshape != nullptr, nullptr, "create cnode return nullptr");
  reshape->set_fullname_with_scope(input_node->fullname_with_scope() + "_transpose");
  return reshape;
}
}  // namespace

bool OnnxPadAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!opt::CheckPrimitiveType(cnode, prim::kPrimPadFusion) ||
        (cnode->inputs().size() != kTripleNum && cnode->inputs().size() != kQuadraNum)) {
      continue;
    }
    // get the second input node whose output is the padding parameter of pad.
    auto input_node = cnode->input(2);
    if (!input_node->isa<CNode>()) {
      continue;
    }
    // reshape the padding of pad operator to 2 x i.
    std::vector<int> shape_pre = {2, -1};
    auto reshape_pre = NewReshapeOpNode(func_graph, input_node, shape_pre);
    if (reshape_pre == nullptr) {
      MS_LOG(ERROR) << "create reshape failed.";
      return false;
    }
    std::vector<int> perm = {1, 0};
    auto transpose = NewTransposeOpNode(func_graph, reshape_pre, perm);
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
    manager->Replace(input_node, reshape_pos);
  }
  return true;
}
}  // namespace mindspore::lite
