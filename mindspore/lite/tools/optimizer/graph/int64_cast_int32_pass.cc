/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/int64_cast_int32_pass.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "ops/cast.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "src/common/utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNotEqualMinIndex = 3;
}  // namespace

bool Int64CastInt32Pass::NotEqualInputsCheck(const CNodePtr &cnode) {
  MS_ASSERT(cnode->size() == kNotEqualMinIndex);
  auto abstract0 = GetCNodeInputAbstract(cnode, kInputIndexOne);
  if (abstract0 == nullptr) {
    MS_LOG(ERROR) << "Abstract of CNode is nullptr";
    return false;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract0)) {
    MS_LOG(DEBUG) << "abstract is not AbstractTensor";
    return false;
  }
  auto abstract1 = GetCNodeInputAbstract(cnode, kInputIndexTwo);
  if (abstract1 == nullptr) {
    MS_LOG(ERROR) << "Abstract of CNode is nullptr";
    return false;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract1)) {
    MS_LOG(DEBUG) << "abstract is not AbstractTensor";
    return false;
  }
  auto abstract0_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract0);
  MS_ASSERT(abstract0_tensor != nullptr && abstract0_tensor->shape() != nullptr);
  auto type0_ptr = abstract0_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type0_ptr != nullptr, false, "type_ptr is nullptr");
  auto abstract1_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract1);
  MS_ASSERT(abstract1_tensor != nullptr && abstract1_tensor->shape() != nullptr);
  auto type1_ptr = abstract1_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type1_ptr != nullptr, false, "type_ptr is nullptr");
  if (type0_ptr->type_id() == type1_ptr->type_id()) {
    return true;
  }
  return false;
}

bool Int64CastInt32Pass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  bool change_flag = false;
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    MS_ASSERT(node != nullptr);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimCast) || CheckPrimitiveType(node, prim::kPrimSplit) ||
        CheckPrimitiveType(node, prim::kPrimGather)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    auto inputs_size = cnode->size();
    if (CheckPrimitiveType(node, prim::kPrimNotEqual)) {
      if (NotEqualInputsCheck(cnode)) {
        continue;
      }
    }
    if (!IsRealCNodeKernel(cnode)) {
      continue;
    }

    for (size_t index = kInputIndexOne; index < inputs_size; index++) {
      auto abstract = GetCNodeInputAbstract(cnode, index);
      if (abstract == nullptr) {
        MS_LOG(DEBUG) << "Cnode " << cnode->fullname_with_scope() << " input " << index << " abstract is nullptr";
        continue;
      }
      if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
        MS_LOG(DEBUG) << "abstract is not AbstractTensor";
        continue;
      }
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
      MS_ASSERT(abstract_tensor != nullptr && abstract_tensor->shape() != nullptr);
      auto type_ptr = abstract_tensor->element()->GetTypeTrack();
      MS_CHECK_TRUE_MSG(type_ptr != nullptr, change_flag, "type_ptr is nullptr");
      if (type_ptr->type_id() == mindspore::kNumberTypeInt64) {
        auto new_cast = std::make_shared<mindspore::ops::Cast>();
        MS_CHECK_TRUE_MSG(new_cast != nullptr, change_flag, "new_cast is nullptr");
        auto new_cast_c = new_cast->GetPrim();
        MS_CHECK_TRUE_MSG(new_cast_c != nullptr, change_flag, "new_cast_c is nullptr");
        ValueNodePtr value_node = NewValueNode(new_cast_c);
        MS_CHECK_TRUE_MSG(value_node != nullptr, change_flag, "NewValueNode Failed");

        auto param_node = opt::BuildIntValueParameterNode(
          graph, static_cast<int32_t>(kNumberTypeInt32),
          cnode->fullname_with_scope() + "_input" + std::to_string(index) + "_cast_type");

        auto cast_cnode = graph->NewCNode({value_node});
        MS_CHECK_TRUE_MSG(cast_cnode != nullptr, change_flag, "new_cnode is nullptr");
        cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_input" + std::to_string(index) +
                                            "_pre_cast");
        cast_cnode->set_abstract(abstract->Clone());
        auto cast_abstract = cast_cnode->abstract();
        MS_ASSERT(cast_abstract != nullptr);
        cast_abstract->set_value(std::make_shared<ValueAny>());

        auto manager = Manage(graph);
        auto input_node = cnode->input(index);
        (void)manager->Replace(input_node, cast_cnode);
        manager->AddEdge(cast_cnode, input_node);
        manager->AddEdge(cast_cnode, param_node);

        // auto inputs = cnode->inputs();
        // inputs[index] = cast_cnode;
        // cnode->set_inputs(inputs);
        change_flag = true;
      }
    }
    if (change_flag) {
      return change_flag;
    }
  }
  return change_flag;
}
}  // namespace mindspore::opt
