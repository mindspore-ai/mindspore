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
#include <memory>
#include <vector>
#include <string>
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/array_ops.h"
#include "ops/lite_ops.h"
#include "ops/tuple_get_item.h"
#include "ops/make_tuple.h"
#include "tools/optimizer/graph/concat_op_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "mindspore/core/abstract/ops/primitive_infer_map.h"
#include "mindspore/core/utils/anf_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "extendrt/utils/func_graph_utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {

#if !defined(_WIN32) && !defined(_WIN64)
CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(SizeToLong(output_idx));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(output_idx));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  CNodePtr tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  tuple_getitem->set_scope(node->scope());
  auto abs = node->abstract()->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_i = abs->elements()[output_idx];
  MS_EXCEPTION_IF_NULL(abs_i);
  tuple_getitem->set_abstract(abs_i);
  return tuple_getitem;
}

bool IsTupleHasDynamicSequence(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if (sequence_abs->dynamic_len() || sequence_abs->dynamic_len_element_abs() != nullptr) {
    return true;
  }
  if (std::any_of(sequence_abs->elements().begin(), sequence_abs->elements().end(),
                  [](const abstract::AbstractBasePtr &abs) { return IsTupleHasDynamicSequence(abs); })) {
    return true;
  }
  return false;
}

size_t GetOutputElementNum(const AnfNodePtr &node) {
  if (node->abstract() != nullptr && IsTupleHasDynamicSequence(node->abstract())) {
    return common::AnfAlgo::GetOutputNumByAbstract(node->abstract());
  }
  return AnfUtils::GetOutputTensorNum(node);
}

int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                         std::vector<AnfNodePtr> *plant_inputs) {
  MS_EXCEPTION_IF_NULL(tuple_input);
  if (!common::AnfAlgo::IsTupleOutput(tuple_input)) {
    auto abs = tuple_input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    MS_LOG(WARNING) << "The Function only split the output type is tuple type but got" << abs->ToString();
    return -1;
  }
  MS_EXCEPTION_IF_NULL(plant_inputs);
  auto input_size = GetOutputElementNum(tuple_input);
  if (tuple_input->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(tuple_input, prim::kPrimMakeTuple)) {
    auto make_tuple = tuple_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(make_tuple);
    for (size_t j = 0; j < tuple_input_num; ++j) {
      // using for graph kernel
      auto dyn_input_node = common::AnfAlgo::GetInputNode(make_tuple, j);
      MS_EXCEPTION_IF_NULL(dyn_input_node);
      // Handle tuple nested scenes.
      if (dyn_input_node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(dyn_input_node, prim::kPrimMakeTuple)) {
        input_size += LongToSize(SplitTupleInputs(graph, dyn_input_node, plant_inputs));
        continue;
      }
      (void)plant_inputs->emplace_back(dyn_input_node);
    }
    return input_size;
  }
  for (size_t index = 0; index < input_size; ++index) {
    auto dynamic_input_node = CreateTupleGetItemNode(graph, tuple_input, index);
    (void)plant_inputs->emplace_back(dynamic_input_node);
  }
  return input_size;
}

static bool IsNotSequenceOfTensor(const abstract::AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractTensor>()) {
    return false;
  }

  if (abs->isa<abstract::AbstractSequence>()) {
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->size() == 0) {
      return true;
    }

    return IsNotSequenceOfTensor(seq_abs->elements()[0]);
  }

  return true;
}

AnfNodePtr ConcatOpPass::ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);

  if (common::AnfAlgo::HasDynamicTupleInput(cnode_ptr)) {
    MS_LOG(INFO) << "Node " << cnode_ptr->fullname_with_scope()
                 << " has dynamic tuple input, can't convert. Node debug string:" << cnode_ptr->DebugString();
    return nullptr;
  }
  std::vector<AnfNodePtr> plant_inputs;
  std::vector<int64_t> dyn_input_sizes;
  plant_inputs.push_back(common::AnfAlgo::GetCNodePrimitiveNode(cnode_ptr));
  size_t input_num = cnode_ptr->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode_ptr, i);
    MS_EXCEPTION_IF_NULL(input_node);
    bool output_is_tuple = common::AnfAlgo::IsTupleOutput(input_node);
    if (output_is_tuple) {
      int64_t dyn_input_size;
      if (IsNotSequenceOfTensor(input_node->abstract())) {
        dyn_input_size = 0;
      } else {
        dyn_input_size = SplitTupleInputs(graph, input_node, &plant_inputs);
      }
      if (dyn_input_size == 0) {
        dyn_input_sizes.push_back(-1);
        plant_inputs.push_back(input_node);
      } else {
        (void)dyn_input_sizes.emplace_back(dyn_input_size);
      }
    } else {
      dyn_input_sizes.push_back(-1);
      plant_inputs.push_back(input_node);
    }
  }
  // Input Axis may be converted into an attribute, and when execed in convert.cc, Axis may be converted back as input.
  dyn_input_sizes.push_back(-1);
  MS_LOG(INFO) << "Concat op pass attr dyn_input_sizes: " << dyn_input_sizes;
  // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    MS_LOG(INFO) << "Step into concat set attr";
    auto new_cnode = graph->NewCNode(plant_inputs);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode_ptr->abstract());
    new_cnode->set_scope(cnode_ptr->scope());
    new_cnode->set_primal_attrs(cnode_ptr->primal_attrs());
    new_cnode->set_attrs(cnode_ptr->attrs());
    new_cnode->set_fullname_with_scope(cnode_ptr->fullname_with_scope() + "-plant_input");
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_cnode);
    return new_cnode;
  }
  return nullptr;
}

STATUS ConcatOpPass::RunInsertSizeAttrPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimConcat)) {
      MS_LOG(INFO) << "Run Concat op pass for concat node " << node->fullname_with_scope();
      auto new_cnode = this->ConvertMakeTupleInputToPlantInputs(func_graph, node->cast<CNodePtr>());
      if (!new_cnode) {
        status = lite::RET_NO_CHANGE;
      } else {
        MS_LOG(INFO) << "Concat op pass create new node: " << new_cnode->fullname_with_scope();
        if (!manager->Replace(node, new_cnode)) {
          MS_LOG(ERROR) << "Concat op pass replace node " << node->fullname_with_scope() << " failed";
          return lite::RET_ERROR;
        }
      }
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run add dynamic input size attr pass at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool ConcatOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = RunInsertSizeAttrPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  return true;
}
#else
bool ConcatOpPass::Run(const FuncGraphPtr &func_graph) { return true; }
#endif
}  // namespace mindspore::opt
