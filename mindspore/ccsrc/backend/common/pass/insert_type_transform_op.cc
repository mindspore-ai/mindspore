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

#include "backend/common/pass/insert_type_transform_op.h"

#include <memory>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
const std::vector<PrimitivePtr> need_handled_types = {prim::kPrimMakeTuple, prim::kPrimTupleGetItem};

void SetObjTypeForTupleGetItemNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(kernel_builder);

  // First input of TupleGetItem must be TUPLE_UNFOLD.
  // Second is the index.
  std::vector<KernelObjectType> input_obj_types = {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR};

  // Get actual output type of TupleGetItem node.
  auto abs_type = AnfAlgo::GetAbstractObjectType(node->abstract());
  std::vector<KernelObjectType> output_obj_types = {kernel::TypeIdToKernelObjectType(abs_type)};

  kernel_builder->SetInputsKernelObjectType(input_obj_types);
  kernel_builder->SetOutputsKernelObjectType(output_obj_types);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  node->set_kernel_info(kernel_info);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_builder->Build(), node.get());
}

int64_t SplitTupleInputs(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
                         std::vector<AnfNodePtr> *plant_inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(tuple_input);
  MS_EXCEPTION_IF_NULL(plant_inputs);

  if (!common::AnfAlgo::IsTupleOutput(tuple_input)) {
    auto abs = tuple_input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    MS_LOG(WARNING) << "The Function only split the output type is tuple type but got" << abs->ToString();
    return -1;
  }
  MS_EXCEPTION_IF_NULL(plant_inputs);
  auto input_size = AnfAlgo::GetOutputTensorNum(tuple_input);
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
        input_size += SplitTupleInputs(graph, dyn_input_node, plant_inputs);
        continue;
      }
      (void)plant_inputs->emplace_back(dyn_input_node);
    }
    return input_size;
  }
  for (size_t index = 0; index < input_size; ++index) {
    auto dynamic_input_node = CreatTupleGetItemNode(graph, tuple_input, index);
    MS_LOG(DEBUG) << "Create TupleGetItem node " << dynamic_input_node->fullname_with_scope() << " for tuple node "
                  << tuple_input->fullname_with_scope();
    // The virtual node's object types should be set.
    SetObjTypeForTupleGetItemNode(dynamic_input_node);
    (void)plant_inputs->emplace_back(dynamic_input_node);
  }
  return input_size;
}

AnfNodePtr CreateNewNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &input_list,
                         const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_node);

  auto new_cnode = NewCNode(input_list, func_graph, {origin_node});
  MS_EXCEPTION_IF_NULL(new_cnode);
  // This pass should not have new node whose abstract differs from the original node. So set the original node's
  // abstract.
  new_cnode->set_abstract(origin_node->abstract());
  new_cnode->set_scope(origin_node->scope());
  new_cnode->set_primal_attrs(origin_node->primal_attrs());
  new_cnode->set_attrs(origin_node->attrs());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr) {
    kernel_graph->FrontBackendlMapUpdate(origin_node, new_cnode);
  }

  // Need to reset new cnode's kernel build info because the inputs type and number could be changed after processing
  // methods. Only reset input types.
  UpdateKernelBuildInfo(new_cnode, origin_node);
  return new_cnode;
}

void CheckDynamicInputSize(const CNodePtr &new_cnode, const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(new_cnode);
  MS_EXCEPTION_IF_NULL(origin_node);

  // Original node should have same dyn_input_sizes and kernel object type size.
  KernelBuildInfoPtr origin_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(origin_node);
  MS_EXCEPTION_IF_NULL(origin_kernel_build_info);

  auto dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(origin_node, kAttrDynInputSizes);
  size_t origin_inputs_kernel_obj_type_size = origin_kernel_build_info->GetAllOutputKernelObjectTypes().size();
  if (dyn_input_sizes.size() != origin_inputs_kernel_obj_type_size) {
    MS_LOG(EXCEPTION) << "Dynamic input size array is " << dyn_input_sizes << " with length " << dyn_input_sizes.size()
                      << ". But input kernel object type size is " << origin_inputs_kernel_obj_type_size;
  }

  size_t total_input_size =
    std::accumulate(dyn_input_sizes.begin(), dyn_input_sizes.end(), 0,
                    [](size_t total, const auto &s) { return total + LongToSize((s < 0) ? 1 : s); });
  size_t expanded_input_size = common::AnfAlgo::GetInputTensorNum(new_cnode);
  if (total_input_size != expanded_input_size) {
    MS_LOG(EXCEPTION) << "Total input size calculated by attr kAttrDynInputSizes is " << total_input_size
                      << ". But new node's expanded input size is " << expanded_input_size;
  }
  MS_LOG(DEBUG) << "Dynamic input size is " << total_input_size << " for node " << new_cnode->fullname_with_scope();
}

void UpdateKernelBuildInfo(const CNodePtr &new_cnode, const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(new_cnode);
  MS_EXCEPTION_IF_NULL(origin_node);

  // Inherit from origin kernel build info.
  KernelBuildInfoPtr origin_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(origin_node);
  MS_EXCEPTION_IF_NULL(origin_kernel_build_info);
  auto new_kernel_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(origin_kernel_build_info);
  MS_EXCEPTION_IF_NULL(new_kernel_builder);

  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, new_cnode)) {
    // Check validation of kernel info for two nodes.
    CheckDynamicInputSize(new_cnode, origin_node);

    // Construct new inputs info and set to the new kernel build info.
    std::vector<std::string> inputs_device_format;
    std::vector<TypeId> inputs_device_type;
    auto dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(origin_node, kAttrDynInputSizes);

    for (size_t i = kIndex0; i < dyn_input_sizes.size(); ++i) {
      int64_t s = (dyn_input_sizes[i] < 0) ? 1 : dyn_input_sizes[i];
      for (int64_t j = kIndex0; j < s; ++j) {
        inputs_device_format.push_back(origin_kernel_build_info->GetInputFormat(i));
        inputs_device_type.push_back(origin_kernel_build_info->GetInputDeviceType(i));
      }
    }

    new_kernel_builder->SetInputsFormat(inputs_device_format);
    new_kernel_builder->SetInputsDeviceType(inputs_device_type);
    MS_LOG(DEBUG) << "Input format, device type and kernel object type are " << inputs_device_format << ", "
                  << inputs_device_type << ", " << origin_kernel_build_info->GetAllInputKernelObjectTypes();
  }

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_cnode->set_kernel_info(kernel_info);
  AnfAlgo::SetSelectKernelBuildInfo(new_kernel_builder->Build(), new_cnode.get());
}

void ExtendTupleUnfoldOutput(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  // This method could be called multiple times for the same node.
  // Only need to expand once.
  if (AnfUtils::IsRealKernel(input) && !common::AnfAlgo::HasNodeAttr(kTupleUnfoldExpanded, input->cast<CNodePtr>())) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(input);
    KernelBuildInfoPtr kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(input);
    MS_EXCEPTION_IF_NULL(kernel_build_info);

    // Check output kernel object type.
    auto kernel_obj_type_list = kernel_build_info->GetAllOutputKernelObjectTypes();
    if (kernel_obj_type_list.size() != kSizeOne || kernel_obj_type_list[kIndex0] != KernelObjectType::TUPLE_UNFOLD) {
      MS_LOG(EXCEPTION) << "The node should be with TupleUnfold type. But got " << kernel_obj_type_list.size()
                        << " kernel object types: " << kObjectTypeToString[kernel_obj_type_list[kIndex0]];
    }

    // Temporarily, each output format and type should be identical for TupleUnfold.
    // Original output is folded so there's only one output format and type, we extend them up to 'output_num'.
    std::vector<std::string> outputs_device_format(output_num, kernel_build_info->GetOutputFormat(kIndex0));
    std::vector<TypeId> outputs_device_type(output_num, kernel_build_info->GetOutputDeviceType(kIndex0));

    kernel_build_info->SetOutputsFormat(outputs_device_format);
    kernel_build_info->SetOutputsDeviceType(outputs_device_type);
    common::AnfAlgo::SetNodeAttr(kTupleUnfoldExpanded, MakeValue(true), input);
    MS_LOG(DEBUG) << "Expand output for " << input->fullname_with_scope() << " with format " << outputs_device_format
                  << ", type " << outputs_device_type;
  }
}

// A map of kernel object type pairs to processing functions.
static std::map<ObjectTypePair, ProcessTypeTransformFunc> kTypePairToProcessFunc;

InsertTypeTransformOp::InsertTypeTransformOp(bool multigraph)
    : PatternProcessPass("insert_type_transform_op", multigraph) {
  kTypePairToProcessFunc[{KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TUPLE_UNFOLD}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleUnfoldToTupleUnfold, this, std::placeholders::_1,
              std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
}

const AnfNodePtr InsertTypeTransformOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  if ((node->kernel_info() == nullptr) ||
      (!dynamic_cast<device::KernelInfo *>(node->kernel_info())->has_build_info())) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList new_input_list = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  // If kernel object types are matched, set this flag to true and new node will be created to replace original node.
  bool matched = false;
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(cnode); ++i) {
    const auto &input_node = common::AnfAlgo::GetInputNode(cnode, i);
    // Skip for monad input.
    if (HasAbstractMonad(input_node)) {
      new_input_list.push_back(input_node);
      continue;
    }

    const auto &real_input_node =
      common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false, need_handled_types).first;
    MS_EXCEPTION_IF_NULL(real_input_node);
    if ((real_input_node->kernel_info() == nullptr) ||
        (!dynamic_cast<device::KernelInfo *>(real_input_node->kernel_info())->has_build_info())) {
      MS_LOG(ERROR) << node->fullname_with_scope() << " input index:" << i
                    << ", input node:" << real_input_node->fullname_with_scope() << " doesn't have build info.";
      new_input_list.push_back(input_node);
      continue;
    }

    auto needed_input_type = AnfAlgo::GetInputKernelObjectType(node, i);
    auto current_input_type = AnfAlgo::GetOutputKernelObjectType(real_input_node, 0);
    if ((kObjectTypeToString.count(needed_input_type) == 0) || (kObjectTypeToString.count(current_input_type) == 0)) {
      MS_LOG(EXCEPTION) << "The current input object type " << current_input_type << " or needed input object type "
                        << needed_input_type << " is not valid for node " << node->fullname_with_scope()
                        << " input index:" << i << ", input node:" << real_input_node->fullname_with_scope();
    }
    MS_LOG(DEBUG) << "The current input object type " << kObjectTypeToString[current_input_type]
                  << " or needed input object type " << kObjectTypeToString[needed_input_type]
                  << " is not valid for node " << node->fullname_with_scope() << " input index:" << i
                  << ", input node:" << real_input_node->fullname_with_scope();

    ObjectTypePair type_pair = {current_input_type, needed_input_type};
    if (kTypePairToProcessFunc.count(type_pair) != 0) {
      matched = true;
      MS_LOG(INFO) << "Kernel object type pair of input index " << (i - 1) << " for node "
                   << cnode->fullname_with_scope() << " is " << type_pair.to_string();
      bool new_prim = false;
      AnfNodePtrList processed_input_list = kTypePairToProcessFunc[type_pair](func_graph, input_node, cnode, &new_prim);
      if (new_prim) {
        // If new primitive is created, replace the old one, which is the first element of the input list.
        new_input_list[kIndex0] = processed_input_list[kIndex0];
        // Jump the primitive node the first one, and the rest is the new inputs.
        new_input_list.insert(new_input_list.end(), std::begin(processed_input_list) + kIndex1,
                              processed_input_list.end());
      } else {
        new_input_list.insert(new_input_list.end(), processed_input_list.begin(), processed_input_list.end());
      }
    } else {
      // If this input type is valid, just push back the origin input.
      new_input_list.push_back(input_node);
    }
  }

  if (matched) {
    // Create replacing node, update front-end node map, set kernel build info, inherit attributes, etc. These
    // operations could rely on the origin CNode.
    auto new_node = CreateNewNode(func_graph, new_input_list, cnode);
    return new_node;
  }
  return nullptr;
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleUnfoldToTupleUnfold(const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &input, const CNodePtr &node,
                                                                      bool *new_prim) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // If the input needs to be skipped as ConvertTupleInputToDynamicInput does, return the input node itself for caller
  // to construct input list.
  bool is_bprop_cut = common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimBpropCut);
  bool skip = (is_bprop_cut && input->abstract()->isa<abstract::AbstractSparseTensor>()) ||
              IsPrimitiveCNode(node, prim::kPrimTupleGetItem);
  if (skip) {
    ExtendTupleUnfoldOutput(input);
    return {input};
  }

  AnfNodePtrList plant_inputs;
  int64_t unfold_num = SplitTupleInputs(func_graph, input, &plant_inputs);
  MS_LOG(DEBUG) << "Transform tuple unfold input: " << input->fullname_with_scope() << " to " << unfold_num
                << " inputs.";

  // If input is a real kernel, we need to update input node's kernel build info to 'extend' its output.
  ExtendTupleUnfoldOutput(input);
  return plant_inputs;
}
}  // namespace opt
}  // namespace mindspore
