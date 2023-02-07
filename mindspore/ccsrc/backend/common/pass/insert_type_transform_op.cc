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
#include "include/common/utils/convert_utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace opt {
int64_t SplitTupleInputsForInsertType(const FuncGraphPtr &graph, const AnfNodePtr &tuple_input,
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

  auto input_size = AnfAlgo::GetOutputElementNum(tuple_input);
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
        int64_t dyn_input_size = SplitTupleInputsForInsertType(graph, dyn_input_node, plant_inputs);
        input_size += LongToSize(dyn_input_size);
        continue;
      }
      (void)plant_inputs->emplace_back(dyn_input_node);
    }
    return SizeToLong(input_size);
  }
  for (size_t index = 0; index < input_size; ++index) {
    auto dynamic_input_node = CreatTupleGetItemNode(graph, tuple_input, index);
    MS_LOG(DEBUG) << "Create TupleGetItem node " << dynamic_input_node->fullname_with_scope() << " for tuple node "
                  << tuple_input->fullname_with_scope();
    // The virtual node's object types should be set.
    SetKernelInfoForNewCNode(dynamic_input_node, false);
    (void)plant_inputs->emplace_back(dynamic_input_node);
  }
  return SizeToLong(input_size);
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

  // Inherit from origin kernel build info.
  KernelBuildInfoPtr origin_kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(origin_node);
  MS_EXCEPTION_IF_NULL(origin_kernel_build_info);
  auto new_kernel_builder = std::make_shared<KernelBuildInfoBuilder>(origin_kernel_build_info);
  MS_EXCEPTION_IF_NULL(new_kernel_builder);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_cnode->set_kernel_info(kernel_info);
  AnfAlgo::SetSelectKernelBuildInfo(new_kernel_builder->Build(), new_cnode.get());

  // Need to reset new cnode's kernel build info because the inputs type and number could be changed after processing
  // methods. Only reset input types.
  auto new_prim = GetValueNode<PrimitivePtr>(new_cnode->input(kIndex0));
  auto origin_prim = GetValueNode<PrimitivePtr>(origin_node->input(kIndex0));
  if (kernel::IsDynamicParamKernel(origin_prim->name())) {
    SetKernelInfoForDynamicParamKernel(new_cnode);
  } else if (IsPrimitiveEquals(new_prim, origin_prim)) {
    SetKernelInfoForNewCNode(new_cnode, false);
  } else {
    SetKernelInfoForNewCNode(new_cnode, true);
  }

  // If the primitive is not changed, this means only inputs are updated. So inherit output from origin node.
  if (IsPrimitiveEquals(new_prim, origin_prim)) {
    KernelBuildInfoPtr new_node_build_info = AnfAlgo::GetSelectKernelBuildInfo(new_cnode);
    KernelBuildInfoPtr origin_node_build_info = AnfAlgo::GetSelectKernelBuildInfo(origin_node);
    new_node_build_info->SetOutputsFormat(origin_node_build_info->GetAllOutputFormats());
    new_node_build_info->SetOutputsDeviceType(origin_node_build_info->GetAllOutputDeviceTypes());
    new_node_build_info->SetOutputsKernelObjectType(origin_node_build_info->GetAllOutputKernelObjectTypes());
  }

  return new_cnode;
}

AnfNodePtr CreateRealMakeTupleByMakeTuple(const FuncGraphPtr &func_graph, const CNodePtr &make_tuple_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  // Create RealMakeTuple node and inherit inputs and abstract from MakeTuple node.
  AnfNodePtrList inputs = make_tuple_node->inputs();
  auto prim = NewValueNode(prim::kPrimRealMakeTuple);
  MS_EXCEPTION_IF_NULL(prim);
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "inputs is empty.";
  }
  inputs[kIndex0] = prim;
  CNodePtr real_make_tuple = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(real_make_tuple);
  real_make_tuple->set_abstract(make_tuple_node->abstract());

  SetKernelInfoForNewCNode(real_make_tuple);

  // RealMakeTuple's inputs must be scalar. To avoid failing to select kernel, we must override RealMakeTuple's
  // KernelObjectTypes, which is created from MakeTuple.
  KernelBuildInfoPtr real_make_tuple_build_info = AnfAlgo::GetSelectKernelBuildInfo(real_make_tuple);
  MS_EXCEPTION_IF_NULL(real_make_tuple_build_info);
  auto inputs_obj_types = real_make_tuple_build_info->GetAllInputKernelObjectTypes();
  auto new_obj_types = inputs_obj_types;
  std::transform(new_obj_types.begin(), new_obj_types.end(), new_obj_types.begin(),
                 [](const auto &obj_type) { return KernelObjectType::SCALAR; });
  real_make_tuple_build_info->SetInputsKernelObjectType(new_obj_types);
  MS_LOG(DEBUG) << "Override RealMakeTuple input kernel object types from " << inputs_obj_types << " " << new_obj_types;
  return real_make_tuple;
}

AnfNodePtr CreateRealMakeTupleByTupleUnfoldInput(const FuncGraphPtr &func_graph,
                                                 const AnfNodePtr &node_with_tuple_unfold_output) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_with_tuple_unfold_output);

  auto prim = NewValueNode(prim::kPrimRealMakeTuple);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, node_with_tuple_unfold_output};
  CNodePtr real_make_tuple = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(real_make_tuple);
  // Inherit abstract from TupleUnfold output node.
  real_make_tuple->set_abstract(node_with_tuple_unfold_output->abstract());

  SetKernelInfoForNewCNode(real_make_tuple);

  // Set object type to TupleUnfold so TupleUnfoldToTupleUnfold pattern will be matched.
  KernelBuildInfoPtr real_make_tuple_build_info = AnfAlgo::GetSelectKernelBuildInfo(real_make_tuple);
  MS_EXCEPTION_IF_NULL(real_make_tuple_build_info);
  real_make_tuple_build_info->SetInputsKernelObjectType({KernelObjectType::TUPLE_UNFOLD});

  // Extend tuple_unfold inputs.
  abstract::AbstractTuplePtr tuple_unfold_abs =
    node_with_tuple_unfold_output->abstract()->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_unfold_abs);
  auto builder = AnfAlgo::GetSelectKernelBuildInfo(real_make_tuple);
  MS_EXCEPTION_IF_NULL(builder);
  std::vector<std::string> inputs_format{tuple_unfold_abs->size(), builder->GetInputFormat(kIndex0)};
  std::vector<TypeId> inputs_type{tuple_unfold_abs->size(), builder->GetInputDeviceType(kIndex0)};
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);

  return real_make_tuple;
}

void SetBackOffFlag(const KernelBuildInfoPtr &build_info, const CNodePtr &cnode) {
  std::vector<std::string> back_off_op_list = {prim::kTupleToTensor,  prim::kScalarToTensor,  prim::kTensorToTuple,
                                               prim::kTensorToScalar, prim::kRealMakeTuple,   prim::kRealTupleGetItem,
                                               prim::kRealMakeList,   prim::kRealListGetItem, prim::kTupleSetItem,
                                               prim::kListToTensor,   prim::kTensorToList};
  if (std::find(back_off_op_list.begin(), back_off_op_list.end(), common::AnfAlgo::GetCNodeName(cnode)) !=
      back_off_op_list.end()) {
    build_info->set_valid(false);
  }
}

void SetKernelInfoForNewCNode(const CNodePtr &cnode, bool set_format_type) {
  MS_EXCEPTION_IF_NULL(cnode);
  // In some cases cnode is newly created and has no kernel info.
  if (cnode->kernel_info() == nullptr ||
      (!dynamic_cast<device::KernelInfo *>(cnode->kernel_info())->has_build_info())) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    cnode->set_kernel_info(kernel_info);
    auto builder = std::make_shared<KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode.get());
  }
  KernelBuildInfoPtr build_info = AnfAlgo::GetSelectKernelBuildInfo(cnode);
  MS_EXCEPTION_IF_NULL(build_info);

  // Set input and output object type for subsequent type matching process.
  std::vector<KernelObjectType> input_obj_type;
  std::vector<KernelObjectType> output_obj_type;
  GenerateKernelObjectTypeForNewCNode(cnode, &input_obj_type, &output_obj_type);
  build_info->SetInputsKernelObjectType(input_obj_type);
  build_info->SetOutputsKernelObjectType(output_obj_type);

  if (set_format_type) {
    // Set input and output format.
    std::vector<std::string> inputs_format;
    std::vector<TypeId> inputs_type;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      auto input_node = common::AnfAlgo::GetInputNode(cnode, input_index);
      if (input_node->kernel_info() == nullptr) {
        inputs_format.emplace_back(kOpFormat_DEFAULT);
      } else {
        inputs_format.emplace_back(AnfAlgo::GetPrevNodeOutputFormat(cnode, input_index));
      }
      inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index));
    }

    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    size_t output_num;
    if (output_obj_type[kIndex0] == KernelObjectType::TUPLE_UNFOLD) {
      output_num = AnfAlgo::GetOutputElementNum(cnode);
    } else {
      output_num = kSizeOne;
    }
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      outputs_format.emplace_back(GenerateOutputFormatForNewCNode(cnode));
      outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(cnode, output_index));
    }

    build_info->SetInputsFormat(inputs_format);
    build_info->SetInputsDeviceType(inputs_type);
    build_info->SetOutputsFormat(outputs_format);
    build_info->SetOutputsDeviceType(outputs_type);
  }

  // The node may not be supported in the current device.
  SetBackOffFlag(build_info, cnode);
  MS_LOG(INFO) << "Set kernel info for cnode " << cnode->DebugString() << " " << cnode->fullname_with_scope() << " "
               << build_info->ToString();
}

void SetKernelInfoForDynamicParamKernel(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  cnode->set_kernel_info(kernel_info);
  auto builder = std::make_shared<KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), cnode.get());
  std::vector<KernelObjectType> input_obj_type =
    kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllInputObjectType(cnode));
  std::vector<KernelObjectType> output_obj_type =
    kernel::TypeIdToKernelObjectType(AnfAlgo::GetAllOutputObjectType(cnode));
  builder->SetInputsKernelObjectType(input_obj_type);
  builder->SetOutputsKernelObjectType(output_obj_type);
  // Set input and output format.
  std::vector<std::string> inputs_format;
  std::vector<TypeId> inputs_type;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, input_index);
    inputs_format.emplace_back(kOpFormat_DEFAULT);
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_index));
  }
  std::vector<std::string> outputs_format;
  std::vector<TypeId> outputs_type;
  size_t output_num = AnfAlgo::GetOutputElementNum(cnode);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_format.emplace_back(kOpFormat_DEFAULT);
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(cnode, output_index));
  }
  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_type);
}

void SetKernelInfoForValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  value_node->set_kernel_info(kernel_info);
  auto builder = std::make_shared<KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);

  auto type_id = value_node->value()->type()->type_id();
  std::vector<std::string> inputs_format = {kOpFormat_DEFAULT};
  std::vector<TypeId> inputs_type = {type_id};
  std::vector<std::string> outputs_format = {kOpFormat_DEFAULT};
  std::vector<TypeId> outputs_type = {type_id};

  auto abs_type = AnfAlgo::GetAbstractObjectType(value_node->abstract());
  std::vector<KernelObjectType> input_obj_type = {kernel::TypeIdToKernelObjectType(abs_type)};
  std::vector<KernelObjectType> output_obj_type = {kernel::TypeIdToKernelObjectType(abs_type)};

  builder->SetInputsFormat(inputs_format);
  builder->SetInputsDeviceType(inputs_type);
  builder->SetOutputsFormat(outputs_format);
  builder->SetOutputsDeviceType(outputs_type);
  builder->SetInputsKernelObjectType(input_obj_type);
  builder->SetOutputsKernelObjectType(output_obj_type);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), value_node.get());
}

abstract::AbstractBasePtr GenerateAbsByOpInfer(const PrimitivePtr &primitive, const AnfNodePtrList &input_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto found = abstract::GetPrimitiveInferImpl(primitive);
  if (!found.has_value()) {
    MS_LOG(EXCEPTION) << primitive->name() << " infer is not registered.";
  }

  std::vector<AbstractBasePtr> input_args;
  std::for_each(input_list.begin(), input_list.end(),
                [&input_args](const auto &input) { input_args.emplace_back(input->abstract()); });
  auto infer_impl = found.value();
  auto abs = infer_impl.InferShapeAndType(nullptr, primitive, input_args);
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for " << primitive->name() << " is " << abs->ToString();
  return abs;
}

std::string GenerateOutputFormatForNewCNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode, prim::kPrimRealMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimTupleToTensor)) {
    // We take first input format as the output format because multiple types and formats of RealMakeTuple/TupleToTensor
    // are not supported.
    std::string represent_format = AnfAlgo::GetPrevNodeOutputFormat(cnode, kIndex0);
    return represent_format;
  }
  return kOpFormat_DEFAULT;
}

void GenerateKernelObjectTypeForNewCNode(const CNodePtr &cnode, std::vector<KernelObjectType> *input_obj_type,
                                         std::vector<KernelObjectType> *output_obj_type) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_obj_type);
  MS_EXCEPTION_IF_NULL(output_obj_type);

  // Simply trasverse all inputs and get their object types.
  // But if the input's object type is not set, this will throw exception so must pay attention when using this
  // function.
  auto general_input_obj_type_func = [&]() {
    for (size_t i = kIndex1; i < cnode->inputs().size(); i++) {
      auto input_node = cnode->input(i);
      MS_EXCEPTION_IF_NULL(input_node);
      // Set input kernel object type as input node's output kernel object type.
      if (input_node->kernel_info() == nullptr ||
          (!dynamic_cast<device::KernelInfo *>(input_node->kernel_info())->has_build_info())) {
        auto abs_type = AnfAlgo::GetAbstractObjectType(input_node->abstract());
        input_obj_type->push_back(kernel::TypeIdToKernelObjectType(abs_type));
      } else {
        auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(input_node);
        input_obj_type->push_back(kernel_build_info->GetOutputKernelObjectType(kIndex0));
      }
    }
  };

  if (IsPrimitiveCNode(cnode, prim::kPrimRealMakeTuple)) {
    general_input_obj_type_func();
    output_obj_type->push_back(KernelObjectType::TUPLE);
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleToTensor)) {
    general_input_obj_type_func();
    output_obj_type->push_back(KernelObjectType::TENSOR);
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTensorToTuple)) {
    general_input_obj_type_func();
    output_obj_type->push_back(KernelObjectType::TUPLE);
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    // First input of TupleGetItem must be TUPLE_UNFOLD.
    // Second is the index.
    *input_obj_type = {KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR};
    // Get actual output type of TupleGetItem node.
    auto abs_type = AnfAlgo::GetAbstractObjectType(cnode->abstract());
    output_obj_type->push_back(kernel::TypeIdToKernelObjectType(abs_type));
  } else if (IsPrimitiveCNode(cnode, prim::kPrimRealTupleGetItem)) {
    general_input_obj_type_func();
    // Get actual output type of RealTupleGetItem node.
    auto abs_type = AnfAlgo::GetAbstractObjectType(cnode->abstract());
    output_obj_type->push_back(kernel::TypeIdToKernelObjectType(abs_type));
  } else if (IsPrimitiveCNode(cnode, prim::kPrimTensorToScalar)) {
    general_input_obj_type_func();
    output_obj_type->push_back(KernelObjectType::SCALAR);
  } else {
    // For other ops, defaulty set TENSOR as output object type.
    general_input_obj_type_func();
    output_obj_type->push_back(KernelObjectType::TENSOR);
  }

  MS_LOG(INFO) << "Generate input and output object types for new node " << cnode->fullname_with_scope() << " "
               << cnode->DebugString() << ". Input object types: " << *input_obj_type
               << ". Output object types: " << *output_obj_type;
}

void UpdateAbsForTupleGetItem(const CNodePtr &tuple_get_item_node) {
  MS_EXCEPTION_IF_NULL(tuple_get_item_node);
  if (!IsPrimitiveCNode(tuple_get_item_node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "Node should be TupleGetItem, but got " << tuple_get_item_node->fullname_with_scope() << ", "
                      << tuple_get_item_node->DebugString();
  }
  auto tuple_input = common::AnfAlgo::GetInputNode(tuple_get_item_node, kIndex0);
  MS_EXCEPTION_IF_NULL(tuple_input);
  auto input_abs = tuple_input->abstract();
  MS_EXCEPTION_IF_NULL(input_abs);
  if (!input_abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "TupleGetItem's first input abstract should be Sequence, but got " << input_abs->ToString();
  }

  auto seq_abs = input_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  AbstractBasePtrList seq_element = seq_abs->elements();
  // This method is used for TupleGetItem to RealTupleGetItem converting, the tuple elements must be scalar for now.
  for (const auto &ele : seq_element) {
    if (!ele->isa<abstract::AbstractScalar>()) {
      MS_LOG(EXCEPTION) << "Element of the tuple should be scalar, but got " << ele->ToString();
    }
  }

  int64_t item_index = GetGetitemIndex(tuple_get_item_node);
  tuple_get_item_node->set_abstract(seq_element[item_index]);
}

// A map of kernel object type pairs to processing functions.
static std::map<ObjectTypePair, ProcessTypeTransformFunc> kTypePairToProcessFunc;

// The nodes of which object types should be handled.
const std::vector<PrimitivePtr> need_handled_types = {prim::kPrimMakeTuple, prim::kPrimTupleGetItem};

InsertTypeTransformOp::InsertTypeTransformOp(bool multigraph)
    : PatternProcessPass("insert_type_transform_op", multigraph) {
  kTypePairToProcessFunc[{KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TUPLE_UNFOLD}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleUnfoldToTupleUnfold, this, std::placeholders::_1,
              std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TUPLE}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleUnfoldToTuple, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TUPLE_UNFOLD, KernelObjectType::TENSOR}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleUnfoldToTensor, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TUPLE, KernelObjectType::TUPLE_UNFOLD}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleToTupleUnfold, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TUPLE, KernelObjectType::TENSOR}] =
    std::bind(&InsertTypeTransformOp::ProcessTupleToTensor, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::SCALAR, KernelObjectType::TENSOR}] =
    std::bind(&InsertTypeTransformOp::ProcessScalarToTensor, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TENSOR, KernelObjectType::TUPLE}] =
    std::bind(&InsertTypeTransformOp::ProcessTensorToTuple, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
  kTypePairToProcessFunc[{KernelObjectType::TENSOR, KernelObjectType::SCALAR}] =
    std::bind(&InsertTypeTransformOp::ProcessTensorToScalar, this, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);
}

const AnfNodePtr InsertTypeTransformOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  if ((node->kernel_info() == nullptr) ||
      (!dynamic_cast<device::KernelInfo *>(node->kernel_info())->has_build_info()) ||
      (common::AnfAlgo::GetCNodeName(node) == "MakeTuple")) {
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
    if (HasAbstractMonad(input_node) || (node->kernel_info() == nullptr) ||
        !dynamic_cast<device::KernelInfo *>(node->kernel_info())) {
      new_input_list.push_back(input_node);
      continue;
    }

    const auto &real_input_node =
      common::AnfAlgo::VisitKernelWithReturnType(input_node, kIndex0, false, need_handled_types).first;
    MS_EXCEPTION_IF_NULL(real_input_node);
    if ((real_input_node->kernel_info() == nullptr) ||
        (!dynamic_cast<device::KernelInfo *>(real_input_node->kernel_info())->has_build_info())) {
      MS_LOG(DEBUG) << node->fullname_with_scope() << " input index:" << i
                    << ", input node:" << real_input_node->fullname_with_scope() << " doesn't have build info.";
      new_input_list.push_back(input_node);
      continue;
    }

    auto needed_input_type = AnfAlgo::GetInputKernelObjectType(node, i);
    auto current_input_type = AnfAlgo::GetOutputKernelObjectType(real_input_node, kIndex0);
    if ((kObjectTypeToString.count(needed_input_type) == 0) || (kObjectTypeToString.count(current_input_type) == 0)) {
      MS_LOG(EXCEPTION) << "The current input object type " << current_input_type << " or needed input object type "
                        << needed_input_type << " is not valid for node " << node->fullname_with_scope()
                        << " input index:" << i << ", input node:" << real_input_node->fullname_with_scope();
    }
    MS_LOG(DEBUG) << "The current input object type:" << kObjectTypeToString[current_input_type]
                  << ", needed input object type:" << kObjectTypeToString[needed_input_type]
                  << " for node:" << node->fullname_with_scope() << " input index:" << i
                  << ", input node:" << real_input_node->fullname_with_scope();

    ObjectTypePair type_pair = {current_input_type, needed_input_type};
    if (kTypePairToProcessFunc.count(type_pair) != 0) {
      MS_LOG(INFO) << "Kernel object type pair of input index " << i << " for node pair "
                   << input_node->fullname_with_scope() << " to " << cnode->fullname_with_scope() << " is "
                   << type_pair.to_string();
      bool new_prim = false;
      AnfNodePtrList processed_input_list = kTypePairToProcessFunc[type_pair](func_graph, input_node, cnode, &new_prim);
      if (IsInputUpdated(input_node, processed_input_list)) {
        matched = true;
      }
      if (new_prim) {
        MS_LOG(DEBUG) << "New primtive is " << processed_input_list[kIndex0]->fullname_with_scope() << " to replace "
                      << new_input_list[kIndex0]->fullname_with_scope();
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
    MS_LOG(INFO) << "Create new node " << new_node->fullname_with_scope() << " " << new_node->DebugString()
                 << " to replace " << cnode->fullname_with_scope() << " " << cnode->DebugString();
    return new_node;
  }
  return nullptr;
}

bool InsertTypeTransformOp::IsInputUpdated(const AnfNodePtr &origin_input, const AnfNodePtrList &new_input_list) const {
  MS_EXCEPTION_IF_NULL(origin_input);
  if (new_input_list.empty()) {
    MS_LOG(EXCEPTION) << "The new input list size should be at least 1, but got 0.";
  }

  if (new_input_list.size() == kSizeOne && new_input_list[kIndex0] == origin_input) {
    MS_LOG(INFO) << "Input node " << origin_input->fullname_with_scope() << " " << origin_input->DebugString()
                 << " should not be updated.";
    return false;
  }
  MS_LOG(DEBUG) << "Input node " << origin_input->fullname_with_scope() << " " << origin_input->DebugString()
                << " will be replaced.";
  return true;
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleUnfoldToTupleUnfold(const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &input, const CNodePtr &node,
                                                                      bool *) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // If the input needs to be skipped as ConvertTupleInputToDynamicInput does, return the input node itself for caller
  // to construct input list.
  bool is_bprop_cut = common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimBpropCut);
  bool skip = (is_bprop_cut && input->abstract()->isa<abstract::AbstractSparseTensor>()) ||
              IsPrimitiveCNode(node, prim::kPrimTupleGetItem);
  if (skip) {
    return {input};
  }

  AnfNodePtrList plant_inputs;
  int64_t unfold_num = SplitTupleInputsForInsertType(func_graph, input, &plant_inputs);
  MS_LOG(DEBUG) << "Transform tuple unfold input: " << input->fullname_with_scope() << " to " << unfold_num
                << " inputs.";
  return plant_inputs;
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleUnfoldToTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                                const CNodePtr &node, bool *) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  AnfNodePtrList result;
  AnfNodePtr real_make_tuple_node = nullptr;
  // If TupleUnfold input is a MakeTuple node, replace it with RealMakeTuple node.
  if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    real_make_tuple_node = CreateRealMakeTupleByMakeTuple(func_graph, input->cast<CNodePtr>());
  } else {
    real_make_tuple_node = CreateRealMakeTupleByTupleUnfoldInput(func_graph, input);
  }
  result.push_back(real_make_tuple_node);
  return result;
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleUnfoldToTensor(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtr &input, const CNodePtr &node,
                                                                 bool *) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // Use TupleToTensor op as the input of this node. Then TupleUnfoldToTuple pattern will be matched.
  auto prim = NewValueNode(prim::kPrimTupleToTensor);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, input};
  CNodePtr tuple_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tuple_to_tensor);

  // Data type of the tensor should be set as an attr of TupleToTensor op.
  size_t input_index = GetInputNodeIndex(input, node);
  auto data_type = AnfAlgo::GetInputDeviceDataType(node, input_index);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), tuple_to_tensor);

  // Set abstract for TupleToTensor op according to user node's input shape and type.
  auto abs = GenerateAbsByOpInfer(prim::kPrimTupleToTensor, {input});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TupleToTensor op is " << abs->ToString();
  tuple_to_tensor->set_abstract(abs);

  SetKernelInfoForNewCNode(tuple_to_tensor);
  // Set object type to TUPLE for TupleUnfoldToTuple pattern to be matched.
  KernelBuildInfoPtr tuple_to_tensor_build_info = AnfAlgo::GetSelectKernelBuildInfo(tuple_to_tensor);
  MS_EXCEPTION_IF_NULL(tuple_to_tensor_build_info);
  tuple_to_tensor_build_info->SetInputsKernelObjectType({KernelObjectType::TUPLE});
  return {tuple_to_tensor};
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleToTupleUnfold(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                                const CNodePtr &node, bool *new_prim) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // This pattern only supports user node is a TupleGetItem node.
  // If this pattern is matched but the user node is not TupleGetItem, throw exception.
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "Tuple to TupleUnfold pattern should have TupleGetItem as user node, but got "
                      << node->fullname_with_scope() << ", " << node->DebugString();
  }

  auto prim = NewValueNode(prim::kPrimRealTupleGetItem);
  MS_EXCEPTION_IF_NULL(prim);
  // Use original inputs except the primitive.
  AnfNodePtrList new_inputs = {prim, input};

  // For TupleGetItem node, the second input value node's kernel info must be in case of nullptr.
  if (common::AnfAlgo::GetInputTensorNum(node) != kSizeTwo) {
    MS_LOG(EXCEPTION) << "Input number of TupleGetItem node " << node->DebugString() << " should be 2. But got "
                      << common::AnfAlgo::GetInputTensorNum(node);
  }
  auto index_input = node->input(kIndex2);
  MS_EXCEPTION_IF_NULL(index_input);
  if (index_input->kernel_info() == nullptr && index_input->isa<ValueNode>()) {
    SetKernelInfoForValueNode(index_input->cast<ValueNodePtr>());
    // Because the index is used as real kernel RealTupleGetItem's second input, we must add TupleGetItem's index to
    // kernel graph so that its device address will be allocated.
    auto kg = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kg);
    MS_LOG(DEBUG) << "Add value " << index_input->DebugString() << " to kernel graph.";
    kg->AddValueNodeToGraph(index_input->cast<ValueNodePtr>());
  }

  // Need to update TupleGetItem abstract.
  UpdateAbsForTupleGetItem(node);

  // The primitive of user is changed.
  *new_prim = true;
  return new_inputs;
}

AnfNodePtrList InsertTypeTransformOp::ProcessTupleToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                           const CNodePtr &node, bool *) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // Simply insert TupleToTensor op between 'input' and 'node'.
  auto prim = NewValueNode(prim::kPrimTupleToTensor);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, input};
  CNodePtr tuple_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tuple_to_tensor);

  // Data type of the tensor should be set as an attr of TupleToTensor op.
  size_t input_index = GetInputNodeIndex(input, node);
  auto data_type = AnfAlgo::GetInputDeviceDataType(node, input_index);
  common::AnfAlgo::SetNodeAttr(kAttrDType, TypeIdToType(data_type), tuple_to_tensor);

  // Set abstract for TupleToTensor op according to user node's input shape and type.
  auto abs = GenerateAbsByOpInfer(prim::kPrimTupleToTensor, {input});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TupleToTensor op is " << abs->ToString();
  tuple_to_tensor->set_abstract(abs);

  SetKernelInfoForNewCNode(tuple_to_tensor);
  return {tuple_to_tensor};
}

AnfNodePtrList InsertTypeTransformOp::ProcessScalarToTensor(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                            const CNodePtr &node, bool *new_prim) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // Simply insert ScalarToTensor op between 'input' and 'node'.
  auto prim = NewValueNode(prim::kPrimScalarToTensor);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, input};
  CNodePtr scalar_to_tensor = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(scalar_to_tensor);

  // Data type of the tensor should be set as an attr of ScalarToTensor op.
  size_t input_index = GetInputNodeIndex(input, node);
  auto data_type = AnfAlgo::GetInputDeviceDataType(node, input_index);
  common::AnfAlgo::SetNodeAttr("dtype", TypeIdToType(data_type), scalar_to_tensor);

  auto abs = GenerateAbsByOpInfer(prim::kPrimScalarToTensor, {input});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for ScalarToTensor op is " << abs->ToString();
  scalar_to_tensor->set_abstract(abs);

  SetKernelInfoForNewCNode(scalar_to_tensor);
  return {scalar_to_tensor};
}

AnfNodePtrList InsertTypeTransformOp::ProcessTensorToTuple(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                           const CNodePtr &node, bool *) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // Create TensorToTuple op.
  auto prim = NewValueNode(prim::kPrimTensorToTuple);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, input};
  CNodePtr tensor_to_tuple = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_to_tuple);

  auto abs = GenerateAbsByOpInfer(prim::kPrimTensorToTuple, {input});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TensorToTuple op is " << abs->ToString();
  tensor_to_tuple->set_abstract(abs);

  SetKernelInfoForNewCNode(tensor_to_tuple);
  return {tensor_to_tuple};
}

AnfNodePtrList InsertTypeTransformOp::ProcessTensorToScalar(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                            const CNodePtr &node, bool *new_prim) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(node);

  // Create TensorToScalar op.
  auto prim = NewValueNode(prim::kPrimTensorToScalar);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {prim, input};
  CNodePtr tensor_to_scalar = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_to_scalar);

  auto abs = GenerateAbsByOpInfer(prim::kPrimTensorToScalar, {input});
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for TensorToScalar op is " << abs->ToString();
  tensor_to_scalar->set_abstract(abs);

  SetKernelInfoForNewCNode(tensor_to_scalar);
  return {tensor_to_scalar};
}
}  // namespace opt
}  // namespace mindspore
