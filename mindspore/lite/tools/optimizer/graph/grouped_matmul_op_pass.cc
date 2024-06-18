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
#include <map>
#include <memory>
#include <vector>
#include <string>
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/array_ops.h"
#include "ops/lite_ops.h"
#include "ops/tuple_get_item.h"
#include "ops/make_tuple.h"
#include "tools/optimizer/graph/grouped_matmul_op_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "mindspore/core/abstract/ops/primitive_infer_map.h"
#include "mindspore/core/utils/anf_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/op_def.h"
#include "extendrt/utils/func_graph_utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
#if !defined(_WIN32) && !defined(_WIN64)
const std::map<std::string, std::map<size_t, TypeId>> OpInputDtypeMap = {{prim::kPrimGroupedMatmul->name(),
                                                                          {{2, TypeId::kNumberTypeFloat16},
                                                                           {3, TypeId::kNumberTypeUInt64},
                                                                           {4, TypeId::kNumberTypeFloat32},
                                                                           {5, TypeId::kNumberTypeFloat16},
                                                                           {6, TypeId::kNumberTypeFloat16}}}};

bool GroupedMatmulOpPass::IsTupleHasDynamicSequence(const abstract::AbstractBasePtr &abstract) {
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
                  [this](const abstract::AbstractBasePtr &abs) { return this->IsTupleHasDynamicSequence(abs); })) {
    return true;
  }
  return false;
}

size_t GroupedMatmulOpPass::GetOutputElementNum(const AnfNodePtr &node) {
  if (node->abstract() != nullptr && IsTupleHasDynamicSequence(node->abstract())) {
    return common::AnfAlgo::GetOutputNumByAbstract(node->abstract());
  }
  return AnfUtils::GetOutputTensorNum(node);
}

CNodePtr GroupedMatmulOpPass::NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg,
                                       const std::vector<AnfNodePtr> &orig_nodes) {
  MS_EXCEPTION_IF_NULL(fg);
  auto node = fg->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(node);
  return node;
}

CNodePtr GroupedMatmulOpPass::CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     size_t output_idx) {
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

void GroupedMatmulOpPass::UseEmptyNodeReplaceNone(const FuncGraphPtr &graph, const std::string &cnode_name,
                                                  const size_t input_idx, std::vector<int64_t> *dyn_input_sizes,
                                                  std::vector<AnfNodePtr> *plant_inputs) {
  MS_EXCEPTION_IF_NULL(dyn_input_sizes);
  MS_EXCEPTION_IF_NULL(plant_inputs);
  if (OpInputDtypeMap.at(cnode_name).find(input_idx) != OpInputDtypeMap.at(cnode_name).end()) {
    // create empty tensor
    auto tensor_type = OpInputDtypeMap.at(cnode_name).at(input_idx);
    std::vector<int64_t> tensor_shape = {0};
    auto empty_tensor = std::make_shared<tensor::Tensor>(tensor_type, tensor_shape);
    // create node
    auto empty_node = std::make_shared<ValueNode>(empty_tensor);
    ValueNodePtr empty_value_node = empty_node->cast<ValueNodePtr>();
    // empty node size is 1
    dyn_input_sizes->emplace_back(1);
    plant_inputs->emplace_back(empty_value_node);
  } else {
    MS_LOG(EXCEPTION) << "Invalid input index. The [" << input_idx << "] in op [" << cnode_name
                      << "] is not in OpInputDtypeMap, cannot use new node replace None.";
  }
}

bool InputArgTypeIsDynamicType(const mindspore::ops::OP_DTYPE input_arg_dtype) {
  if (input_arg_dtype >= mindspore::ops::DT_TUPLE_BOOL && input_arg_dtype <= mindspore::ops::DT_LIST_ANY) {
    return true;
  }
  return false;
}

int64_t GroupedMatmulOpPass::SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
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

bool GroupedMatmulOpPass::IsNotSequenceOfTensor(const abstract::AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractTensor>()) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->size() == 0) {
      return true;
    }
    return this->IsNotSequenceOfTensor(seq_abs->elements()[0]);
  }
  return true;
}

AnfNodePtr GroupedMatmulOpPass::ConvertMakeTupleInputToPlantInputs(const FuncGraphPtr &graph,
                                                                   const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);

  if (common::AnfAlgo::HasDynamicTupleInput(cnode_ptr)) {
    MS_LOG(INFO) << "Node " << cnode_ptr->fullname_with_scope()
                 << " has dynamic tuple input, can't convert. Node debug string:" << cnode_ptr->DebugString();
    return nullptr;
  }

  auto cnode_name = common::AnfAlgo::GetCNodeName(cnode_ptr);
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
    } else if (OpInputDtypeMap.find(cnode_name) != OpInputDtypeMap.end()) {
      // Only op in OpInputDtypeMap can be replace None input.
      auto opdef_ptr = mindspore::ops::GetOpDef(cnode_name);
      MS_EXCEPTION_IF_NULL(opdef_ptr);
      auto input_args = (opdef_ptr)->args_;
      if (i >= input_args.size()) {
        MS_LOG(EXCEPTION) << "The [" << i << "] in op [" << cnode_name << "] is out of op_def args range";
      }
      // When input[i] is None and input[i] type in op_yaml is dynamic type, do replace
      if (common::AnfAlgo::IsNoneInput(cnode_ptr, i) && InputArgTypeIsDynamicType(input_args[i].arg_dtype_)) {
        UseEmptyNodeReplaceNone(graph, cnode_name, i, &dyn_input_sizes, &plant_inputs);
      } else {
        dyn_input_sizes.push_back(-1);
        plant_inputs.push_back(input_node);
      }
    } else {
      dyn_input_sizes.push_back(-1);
      plant_inputs.push_back(input_node);
    }
  }
  // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
  if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
    auto new_cnode = NewCNode(plant_inputs, graph, {cnode_ptr});
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode_ptr->abstract());
    new_cnode->set_scope(cnode_ptr->scope());
    new_cnode->set_primal_attrs(cnode_ptr->primal_attrs());
    new_cnode->set_attrs(cnode_ptr->attrs());
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_cnode);
    return new_cnode;
  }
  return nullptr;
}

STATUS GroupedMatmulOpPass::RunInsertSizeAttrPass(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimGroupedMatmul)) {
      MS_LOG(INFO) << "Run GroupedMatmul op pass for grouped_matmul node " << node->fullname_with_scope();
      auto new_cnode = this->ConvertMakeTupleInputToPlantInputs(func_graph, node->cast<CNodePtr>());
      if (!new_cnode) {
        status = lite::RET_NO_CHANGE;
      } else {
        MS_LOG(INFO) << "GroupedMatmul op pass create new node: " << new_cnode->fullname_with_scope();
        if (!manager->Replace(node, new_cnode)) {
          MS_LOG(ERROR) << "GroupedMatmul op pass replace node " << node->fullname_with_scope() << " failed";
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

bool GroupedMatmulOpPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = RunInsertSizeAttrPass(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  return true;
}
#else
bool GroupedMatmulOpPass::Run(const FuncGraphPtr &func_graph) { return true; }
#endif
}  // namespace mindspore::opt
