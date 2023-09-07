/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_custom_op_adjust.h"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <memory>
#include "mindspore/core/ops/random_ops.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/op_name.h"
#include "tools/converter/ops/ops_def.h"
#include "ops/transpose.h"
#include "ops/shape.h"
#include "ops/add.h"
#include "ops/mul.h"
#include "ops/reverse_v2.h"
#include "ops/uniform_real.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/node_util.h"
#include "tools/lite_exporter/fetch_content.h"
#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore::lite {
namespace {

CNodePtr NewCNode(const CNodePtr &cnode, const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs,
                  const abstract::AbstractBasePtr &abstract, const std::string &name) {
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, funcGraph cannot be nullptr";
    return nullptr;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, FuncGraph manager cannot be nullptr";
    return nullptr;
  }
  auto new_node = func_graph->NewCNode(primitive, inputs);
  if (new_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create node " << name << " for node " << cnode->fullname_with_scope();
    return nullptr;
  }
  new_node->set_fullname_with_scope(name);
  for (size_t i = 0; i < inputs.size(); i++) {
    manager->SetEdge(new_node, i + 1, inputs[i]);
  }
  new_node->set_abstract(abstract);
  return new_node;
}

STATUS AdjustRot90(const FuncGraphPtr &func_graph, const CNodePtr &cnode, bool *need_update_manager) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto reversev2_op = std::make_shared<ops::ReverseV2>();
  MS_CHECK_TRUE_RET(reversev2_op != nullptr, RET_ERROR);
  auto inputs = cnode->inputs();
  MS_CHECK_TRUE_RET(inputs.size() == 3, RET_ERROR);
  auto primitive_c = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(primitive_c != nullptr, RET_ERROR);
  auto kernel_ptr = std::dynamic_pointer_cast<Rot90>(primitive_c);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast Rot90 failed!";
    return false;
  }
  auto axis_value = primitive_c->GetAttr(ops::kAxis);
  MS_CHECK_TRUE_RET(axis_value != nullptr, -1);
  reversev2_op->set_axis(GetValue<std::vector<int64_t>>(axis_value));
  auto reversev2_cnode = NewCNode(cnode, reversev2_op->GetPrim(), {inputs[1]}, cnode->abstract()->Clone(),
                                  cnode->fullname_with_scope() + "_reversev2_with_axis");
  MS_CHECK_TRUE_RET(reversev2_cnode != nullptr, RET_ERROR);
  auto transpose_op = std::make_shared<ops::Transpose>();
  MS_CHECK_TRUE_RET(transpose_op != nullptr, RET_ERROR);
  auto transpose_op_cnode = NewCNode(cnode, transpose_op->GetPrim(), {reversev2_cnode, inputs[2]},
                                     cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_transpose");
  MS_CHECK_TRUE_RET(transpose_op_cnode != nullptr, RET_ERROR);
  auto manager = Manage(func_graph, true);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  auto node_users = manager->node_users()[cnode];
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, transpose_op_cnode);
  }
  *need_update_manager = true;
  return RET_OK;
}

STATUS AdjustRandomUniformLike(const FuncGraphPtr &func_graph, const CNodePtr &cnode, bool *need_update_manager) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto &inputs = cnode->inputs();
  MS_CHECK_TRUE_RET(inputs.size() == 2, RET_ERROR);
  auto primitive_c = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(primitive_c != nullptr, RET_ERROR);
  auto kernel_ptr = std::dynamic_pointer_cast<RandomUniformLike>(primitive_c);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Cast RandomUniformLike failed!";
    return false;
  }
  auto shape_op = std::make_shared<ops::Shape>();
  MS_CHECK_TRUE_RET(shape_op != nullptr, RET_ERROR);
  auto shape_op_cnode = NewCNode(cnode, shape_op->GetPrim(), {inputs[1]}, cnode->abstract()->Clone(),
                                 cnode->fullname_with_scope() + "_shape");
  MS_CHECK_TRUE_RET(shape_op_cnode != nullptr, RET_ERROR);

  auto uniform_op = std::make_shared<ops::UniformReal>();
  MS_CHECK_TRUE_RET(uniform_op != nullptr, RET_ERROR);
  auto uniform_op_cnode = NewCNode(cnode, uniform_op->GetPrim(), {shape_op_cnode}, cnode->abstract()->Clone(),
                                   cnode->fullname_with_scope() + "_uniform");
  MS_CHECK_TRUE_RET(uniform_op_cnode != nullptr, RET_ERROR);
  if (kernel_ptr->HasAttr(ops::kSeed)) {
    uniform_op->set_seed(GetValue<float>(kernel_ptr->GetAttr(ops::kSeed)));
  }
  if (!kernel_ptr->HasAttr("high") && !kernel_ptr->HasAttr("low")) {
    auto manager = Manage(func_graph, true);
    MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
    auto node_users = manager->node_users()[cnode];
    for (auto &node_user : node_users) {
      manager->SetEdge(node_user.first, node_user.second, uniform_op_cnode);
    }
    *need_update_manager = true;
    return RET_OK;
  }
  float high_val = 1.0f;
  float low_val = 0.0f;
  if (kernel_ptr->HasAttr("high")) {
    high_val = GetValue<float>(kernel_ptr->GetAttr("high"));
  }
  if (kernel_ptr->HasAttr("low")) {
    low_val = GetValue<float>(kernel_ptr->GetAttr("low"));
  }
  auto distance = high_val - low_val;
  ValueNodePtr distance_value_node = NewValueNode(distance);
  MS_CHECK_TRUE_RET(distance_value_node != nullptr, RET_ERROR);

  auto mul_op = std::make_shared<ops::Mul>();
  MS_CHECK_TRUE_RET(mul_op != nullptr, RET_ERROR);
  auto mul_op_cnode = NewCNode(cnode, mul_op->GetPrim(), {uniform_op_cnode, distance_value_node},
                               cnode->abstract()->Clone(), cnode->fullname_with_scope() + "_mul");
  MS_CHECK_TRUE_RET(mul_op_cnode != nullptr, RET_ERROR);

  auto min_val = low_val;
  ValueNodePtr min_var_value_node = NewValueNode(min_val);
  MS_CHECK_TRUE_RET(min_var_value_node != nullptr, RET_ERROR);

  auto add_op = std::make_shared<ops::Add>();
  MS_CHECK_TRUE_RET(add_op != nullptr, RET_ERROR);
  auto add_op_cnode = NewCNode(cnode, add_op->GetPrim(), {mul_op_cnode, min_var_value_node}, cnode->abstract()->Clone(),
                               cnode->fullname_with_scope() + "_add");
  MS_CHECK_TRUE_RET(add_op_cnode != nullptr, RET_ERROR);

  auto manager = Manage(func_graph, true);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  auto node_users = manager->node_users()[cnode];
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, add_op_cnode);
  }
  *need_update_manager = true;
  return RET_OK;
}

STATUS AdjustOp(const FuncGraphPtr &func_graph, const CNodePtr cnode, const AnfNodePtr node,
                bool *need_update_manager) {
  int status = RET_OK;
  std::string primitive_name = "";
  status = opt::GetPrimitiveType(node, &primitive_name);
  if (primitive_name == kNameRot90) {
    status = AdjustRot90(func_graph, cnode, need_update_manager);
  } else if (primitive_name == kNameRandomUniformLike) {
    status = AdjustRandomUniformLike(func_graph, cnode, need_update_manager);
  }
  return status;
}
}  // namespace

bool OnnxCustomOpAdjust::Adjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  bool need_update_manager = false;
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "node is not cnode.";
      continue;
    }
    status = AdjustOp(func_graph, cnode, node, &need_update_manager);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust op split pass is failed.";
      return false;
    }
  }
  if (need_update_manager) {
    mindspore::opt::UpdateManager(func_graph);
  }
  return true;
}
}  // namespace mindspore::lite
