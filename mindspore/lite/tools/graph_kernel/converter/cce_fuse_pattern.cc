/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/cce_fuse_pattern.h"

#include <memory>
#include <vector>
#include <algorithm>
#include "tools/graph_kernel/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/nn_ops.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph_cloner.h"
#include "ir/anf.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/ir/pattern_matcher.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {

void AddReshapeTransposeFusion::InitValidShapes() {
  // cce only support the following shapes
  valid_shape_ = {
    {"Add", {{2, 4096, 320}, {2, 1024, 640}, {2, 256, 1280}}},
    {"Reshape", {{2, 64, 64, 320}, {2, 32, 32, 640}, {2, 16, 16, 1280}}},
    {"Transpose", {{2, 320, 64, 64}, {2, 640, 32, 32}, {2, 1280, 16, 16}}},
  };
}

const BaseRef AddReshapeTransposeFusion::DefinePattern() const {
  auto x = std::make_shared<Var>();
  auto y = std::make_shared<Var>();
  auto add = VectorRef({add_, x, y});
  auto reshape_param = std::make_shared<Var>();
  auto reshape = VectorRef({reshape_, add, reshape_param});
  auto trans_param = std::make_shared<Var>();
  auto trans = VectorRef({trans_, reshape, trans_param});
  return trans;
}

const bool AddReshapeTransposeFusion::IsValidShape(AnfNodePtr const &node) const {
  if (!node->isa<CNode>()) {
    MS_LOG(ERROR) << "not cnode ";
    return false;
  }
  auto cnode = utils::cast<CNodePtr>(node);
  auto shape_ptr = cnode->Shape();
  if (shape_ptr == nullptr) {
    return false;
  }
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  auto op_shape = shape->shape();
  bool is_dynamic_rank = IsDynamicRank(op_shape);
  bool is_dynamic_shape = IsDynamicShape(op_shape);
  if (is_dynamic_rank || is_dynamic_shape) {
    return false;
  }
  auto prim_node = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim_node == nullptr) {
    MS_LOG(ERROR) << "prim_node is nullptr ";
    return false;
  }
  auto prim_name = prim_node->name();
  if (valid_shape_.find(prim_name) == valid_shape_.end()) {
    MS_LOG(INFO) << "not found op " << prim_name;
    return false;
  }
  auto valid_op_shape = valid_shape_.at(prim_name);
  if (valid_op_shape.find(op_shape) == valid_op_shape.end()) {
    MS_LOG(INFO) << "not found shape " << op_shape;
    return false;
  }
  return true;
}

const AnfNodePtr AddReshapeTransposeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  if (!node->isa<CNode>()) {
    return node;
  }
  auto add_node = opt::GetAnfNodeByVar(equiv, add_);
  if (add_node == nullptr) {
    MS_LOG(ERROR) << "add_node match failed";
    return node;
  }
  auto reshape_node = opt::GetAnfNodeByVar(equiv, reshape_);
  if (reshape_node == nullptr) {
    MS_LOG(ERROR) << "reshape_node match failed";
    return node;
  }
  auto trans_node = opt::GetAnfNodeByVar(equiv, trans_);
  if (trans_node == nullptr) {
    MS_LOG(ERROR) << "trans_node match failed";
    return node;
  }
  if (!IsValidShape(add_node) || !IsValidShape(reshape_node) || !IsValidShape(trans_node)) {
    graphkernel::GkUtils::SetCceOpNotFusion(add_node);
    graphkernel::GkUtils::SetCceOpNotFusion(reshape_node);
    graphkernel::GkUtils::SetCceOpNotFusion(trans_node);
    return node;
  }
  return node;
}
}  // namespace mindspore::graphkernel
