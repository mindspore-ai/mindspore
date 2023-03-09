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
#include "backend/common/pass/inplace_assign_for_custom_op.h"

#include <memory>
#include <vector>
#include <string>
#include <regex>
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
constexpr auto kCustomOutput = 0;
constexpr auto kCustomInput = 1;
constexpr auto kCustomAttrInplaceAssignOutput = "inplace_assign_output";

// Used to find Custom op outputs' inplace assign index
std::vector<std::vector<int64_t>> GetHybridInplaceIndex(const CNodePtr &cnode) {
  if (common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrFuncType) != kCustomTypeHybrid) {
    return {};
  }

  if (!common::AnfAlgo::HasNodeAttr(kCustomAttrInplaceAssignOutput, cnode)) {
    return {};
  }
  auto inplace_index_str = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kCustomAttrInplaceAssignOutput);
  std::regex delimiters(" ");
  std::vector<std::string> index(
    std::sregex_token_iterator(inplace_index_str.begin(), inplace_index_str.end(), delimiters, -1),
    std::sregex_token_iterator());
  std::vector<std::vector<int64_t>> inplace_index;
  std::vector<int64_t> tmp;
  for (size_t i = 0; i < index.size(); i++) {
    tmp.push_back(std::stol(index[i]));
    if ((i & 1) != 0) {
      inplace_index.push_back(tmp);
      tmp.clear();
    }
  }
  return inplace_index;
}

CNodePtr InplaceAssign(const FuncGraphPtr &func_graph, const AnfNodePtr &src, const CNodePtr &dst) {
  // Insert Depend
  AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), src, dst};
  auto depend_cnode = func_graph->NewCNode(depend_inputs);
  depend_cnode->set_abstract(src->abstract());

  return depend_cnode;
}

CNodePtr InplaceAssignAfterCustom(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto inplace_info = GetHybridInplaceIndex(cnode);
  if (inplace_info.size() != 1) {
    return nullptr;
  }
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  if (auto i = LongToSize(inplace_info[0][kCustomInput]); i < input_size) {
    return InplaceAssign(func_graph, cnode->input(i + 1), cnode);
  } else {
    return nullptr;
  }
}

CNodePtr InplaceAssignAfterTupleGetItem(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto input_node = cnode->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(input_node);
  auto real_input = dyn_cast<CNode>(input_node);
  if (real_input == nullptr) {
    return nullptr;
  }
  auto value_ptr = GetValueNode(cnode->input(kInputNodeOutputIndexInTupleGetItem));
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto gt_idx = GetValue<int64_t>(value_ptr);
  if (IsPrimitiveCNode(real_input, prim::kPrimCustom)) {
    auto inplace_info = GetHybridInplaceIndex(real_input);
    for (auto index : inplace_info) {
      if (index[kCustomOutput] == gt_idx && index[kCustomInput] >= 0) {
        auto custom_input_size = common::AnfAlgo::GetInputTensorNum(real_input);
        if (auto i = LongToSize(index[kCustomInput]); i < custom_input_size) {
          return InplaceAssign(func_graph, real_input->input(i + 1), cnode);
        }
      }
    }
  }
  return nullptr;
}

std::vector<std::string> InplaceAssignForCustomOp::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimCustom->name());
  return ret;
}

const AnfNodePtr InplaceAssignForCustomOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }

  if (IsPrimitiveCNode(cnode, prim::kPrimCustom) && visited_.find(cnode) == visited_.end()) {
    (void)visited_.insert(cnode);
    return InplaceAssignAfterCustom(func_graph, cnode);
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) && visited_.find(cnode) == visited_.end()) {
    (void)visited_.insert(cnode);
    return InplaceAssignAfterTupleGetItem(func_graph, cnode);
  }

  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
