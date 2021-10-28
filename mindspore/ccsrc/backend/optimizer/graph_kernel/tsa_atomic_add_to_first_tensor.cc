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

#include "backend/optimizer/graph_kernel/tsa_atomic_add_to_first_tensor.h"
#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <utility>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
class TsaChecker : public AtomicAddChecker {
 public:
  explicit TsaChecker(const PrimitivePtr &target) { target_type_ = target; }
  virtual ~TsaChecker() = default;

 protected:
  bool CanActivateAtomicAdd(const AnfNodePtr &anf_node) override {
    if (!FindCandidate(anf_node)) {
      return false;
    }

    auto tsa_cnode = atomic_add_info_.atomic_add_node;
    if (!utils::isa<ParameterPtr>(tsa_cnode->input(1))) {
      return false;
    }

    return true;
  }
};

AnfNodePtr TsaAtomicAddToFirstTensor::FindTsaFirstRealInputInGraph(const KernelGraphPtr &, const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  auto first_input = atomic_add_node_->input(1)->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(first_input);
  auto parameters = sub_graph->parameters();
  bool hit = false;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i] == first_input) {
      tsa_first_input_index_ = i;
      hit = true;
      break;
    }
  }
  if (!hit) {
    MS_LOG(EXCEPTION) << "Cannot find tensor scatter add first input in sub-graph parameters!";
  }

  return cnode->input(tsa_first_input_index_ + 1);  // CNode input have a primitive, so add 1.
}

AnfNodePtr TsaAtomicAddToFirstTensor::ProcessTsaFirstNode(const KernelGraphPtr &main_graph, const AnfNodePtr &node) {
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  // find first input of tsa
  auto tsa_first_input = FindTsaFirstRealInputInGraph(main_graph, node);
  auto users = mng->node_users()[tsa_first_input];
  if (users.size() == 1 && !(utils::isa<ValueNodePtr>(tsa_first_input) || utils::isa<ParameterPtr>(tsa_first_input))) {
    return tsa_first_input;
  }
  // Create composite op's sub-graph.
  auto new_sub_graph = std::make_shared<FuncGraph>();
  auto parameter = new_sub_graph->add_parameter();
  auto kernel_with_index = AnfAlgo::VisitKernel(tsa_first_input, 0);
  parameter->set_abstract(GetOutputAbstract(kernel_with_index.first, kernel_with_index.second));
  parameter->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::string parameter_format;
  TypeId parameter_type;
  if (utils::isa<ValueNodePtr>(kernel_with_index.first)) {
    auto tensor = GetValueNode<tensor::TensorPtr>(kernel_with_index.first);
    MS_EXCEPTION_IF_NULL(tensor);
    parameter_format = kOpFormat_DEFAULT;
    parameter_type = tensor->data_type();
  } else {
    parameter_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    parameter_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder para_info_builder;
  para_info_builder.SetOutputsFormat({parameter_format});
  para_info_builder.SetOutputsDeviceType({parameter_type});
  para_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  para_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  AnfAlgo::SetSelectKernelBuildInfo(para_info_builder.Build(), parameter.get());

  // Create inner op.
  auto identity_node =
    CreateCNode({NewValueNode(std::make_shared<Primitive>("Reshape")), parameter}, new_sub_graph,
                {.format = GetFormat(parameter), .shape = GetShape(parameter), .type = GetType(parameter)});
  SetNodeAttrSafely("shape", MakeValue(GetDeviceShape(parameter)), identity_node);

  // Makeup sub-graph.
  new_sub_graph->set_output(identity_node);
  auto new_composite_node = main_graph->NewCNode({NewValueNode(new_sub_graph), tsa_first_input});
  new_composite_node->set_abstract(identity_node->abstract());
  SetNewKernelInfo(new_composite_node, new_sub_graph, {tsa_first_input}, {identity_node});
  auto graph_attr = ExtractGraphKernelName(TopoSort(new_sub_graph->get_return()), "", "tsa_identity");
  new_sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(graph_attr));
  new_sub_graph->set_attr("composite_type", MakeValue("tsa_identity"));

  return new_composite_node;
}

void TsaAtomicAddToFirstTensor::CorrectKernelBuildInfo(const AnfNodePtr &composite_node,
                                                       const AnfNodePtr &modified_input, bool) {
  // Change kernel build info with modify input
  auto kernel_info = static_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_outputs_format = origin_kernel_build_info->GetAllOutputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();
  auto origin_outputs_type = origin_kernel_build_info->GetAllOutputDeviceTypes();
  auto origin_processor = origin_kernel_build_info->processor();

  std::vector<std::string> &modified_inputs_format = origin_inputs_format;
  std::vector<TypeId> &modified_inputs_type = origin_inputs_type;
  std::vector<std::string> new_outputs_format;
  std::vector<TypeId> new_outputs_type;
  for (size_t i = 0; i < origin_outputs_format.size(); ++i) {
    if (real_output_num_ > 1 && i == reduce_real_output_index_) {
      continue;
    }
    new_outputs_format.push_back(origin_outputs_format[i]);
    new_outputs_type.push_back(origin_outputs_type[i]);
  }

  auto kernel_with_index = AnfAlgo::VisitKernel(modified_input, 0);
  modified_inputs_format[tsa_first_input_index_] =
    AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
  modified_inputs_type[tsa_first_input_index_] =
    AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);

  kernel::KernelBuildInfo::KernelBuildInfoBuilder new_info_builder;
  new_info_builder.SetInputsFormat(modified_inputs_format);
  new_info_builder.SetInputsDeviceType(modified_inputs_type);
  new_info_builder.SetOutputsFormat(new_outputs_format);
  new_info_builder.SetOutputsDeviceType(new_outputs_type);
  new_info_builder.SetProcessor(origin_processor);
  new_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  new_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto new_selected_info = new_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void TsaAtomicAddToFirstTensor::ProcessOriginCNode(const AnfNodePtr &composite_node, const AnfNodePtr &outter_node) {
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // modify input
  composite_node->cast<CNodePtr>()->set_input(tsa_first_input_index_ + 1, outter_node);
  CreateInplaceAssignNodeAndCorrectReturn(sub_graph, sub_graph->parameters()[tsa_first_input_index_]);

  CorrectAbstract(composite_node);
  CorrectKernelBuildInfo(composite_node, outter_node);

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name = ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "tensor_scatter_add_modified");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to tensor scatter add graph " << new_graph_name;
}

void TsaAtomicAddToFirstTensor::ProcessTsa(const KernelGraphPtr &main_graph, const AnfNodePtr &anf_node,
                                           const FuncGraphManagerPtr &mng) {
  auto origin_composite_node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_composite_node);

  // Create identity node.
  auto outter_node = ProcessTsaFirstNode(main_graph, anf_node);

  // Insert extra input(broadcast node output) to composite node, and make origin TensorScatterAdd inplaceassign to it.
  // Note: if it's single output, this will increase total memory because of a fake out.
  ProcessOriginCNode(origin_composite_node, outter_node);

  // Insert update_state_node to keep execution order.
  auto update_state_node = InsertUpdateState(main_graph, origin_composite_node);

  // Replace origin ReduceSum's user with atomic clean output
  ProcessOriginCNodeUser(main_graph, origin_composite_node, outter_node, update_state_node, mng);
  MS_LOG(INFO) << "Target node: " << origin_composite_node->fullname_with_scope()
               << ", outer node: " << outter_node->fullname_with_scope();
}

bool TsaAtomicAddToFirstTensor::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  std::shared_ptr<AtomicAddChecker> atomic_add_checker =
    std::make_shared<TsaChecker>(std::make_shared<Primitive>("TensorScatterAdd"));
  if (atomic_add_checker == nullptr) {
    return changed;
  }

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    if (!atomic_add_checker->Check(node)) {
      continue;
    }
    auto atomic_add_info = atomic_add_checker->GetAtomicAddInfo();
    atomic_add_node_ = atomic_add_info.atomic_add_node;
    reduce_real_output_index_ = atomic_add_info.reduce_real_output_index;
    real_output_num_ = atomic_add_info.real_output_num;
    ProcessTsa(kernel_graph, node, mng);
    changed = true;
  }

  if (changed) {
    UpdateMng(mng, func_graph);
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
