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

#include <unordered_map>
#include <algorithm>
#include <utility>

#include "extendrt/graph_compiler/default_graph_compiler.h"
#include "extendrt/graph_compiler/factory.h"
#include "extendrt/execution_plan.h"
#include "extendrt/mock/lite_runtime/converters.h"
#include "backend/graph_compiler/graph_partition.h"
#include "ops/core_ops.h"
#include "include/common/utils/utils.h"

namespace mindspore {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static constexpr auto ms_infer_backend_name = "mindspore_lite_backend";

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Compile(FuncGraphPtr graph) {
  MS_LOG(INFO) << "DefaultGraphCompiler::Compile";

  inner_context_ = ContextUtils::Convert(context_.get());

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Partition Partition FunctionGraph Begin";
  auto graph_segments = Partition(graph);
  if (graph_segments.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Partition partition graph failed";
    return nullptr;
  }
  MS_LOG(DEBUG) << "DefaultGraphCompiler::Partition Partition FunctionGraph End";

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Schedule Graph Execute Plan Begin";
  auto execution_plan = Schedule(graph_segments, graph);
  if (execution_plan == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Partition partition graph failed";
    return nullptr;
  }
  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Schedule Graph Execute Plan End";
  return execution_plan;
}

std::vector<GraphSegmentPtr> DefaultGraphCompiler::Partition(const FuncGraphPtr &graph) {
  auto partition = std::make_shared<compile::GraphPartition>(ms_infer_cut_list, ms_infer_backend_name);
  if (partition == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Partition create graph partition failed, maybe not enough memory";
    return std::vector<GraphSegmentPtr>();
  }

  // if the context target is cpu, graph should convert to NHWC, call related pass

  // multi_target set false
  bool is_multi_target;
  return partition->Partition(graph, &is_multi_target);
}

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Schedule(
  const std::vector<GraphSegmentPtr> &graph_segments, FuncGraphPtr func_graph) {
  auto execution_plan = std::make_shared<infer::ExecutionPlan>();
  anf_tensor_map_.clear();

  std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *> *input_isolate_map =
    new std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *>();
  std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *> *output_isolate_map =
    new std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *>();

  // Convert FuncGraph Input and Output AnfNode to Tensor and save in Execution Plan
  auto graph_inputs = func_graph->get_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph inputs node failed";
    delete input_isolate_map;
    delete output_isolate_map;
    return nullptr;
  }
  std::vector<AnfNodePtr> graph_outputs;
  auto graph_output = func_graph->output();
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph output node failed";
    delete input_isolate_map;
    delete output_isolate_map;
    return nullptr;
  }
  graph_outputs.emplace_back(graph_output);
  auto graph_input_tensors = CreateTensors(graph_inputs);
  if (graph_input_tensors.size() != graph_inputs.size()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule create graph input tensors failed";
    delete input_isolate_map;
    delete output_isolate_map;
    return nullptr;
  }
  execution_plan->SetInputs(graph_input_tensors);
  auto graph_output_tensors = CreateTensors(graph_outputs);
  if (graph_output_tensors.size() != graph_outputs.size()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule create graph output tensor failed";
    delete input_isolate_map;
    delete output_isolate_map;
    return nullptr;
  }
  execution_plan->SetOutputs(graph_output_tensors);
  execution_plan->SetContext(inner_context_);

  for (auto graph_segment : graph_segments) {
    FuncGraphPtr fg = nullptr;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(graph_segment->nodes_);
    auto execution_flow = this->Schedule(graph_segment, inputs, outputs);
    if (execution_flow == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule schedule graph segment failed";
      delete input_isolate_map;
      delete output_isolate_map;
      return nullptr;
    }
    execution_flow->SetContext(inner_context_);

    for (size_t i = 0; i < execution_flow->GetInputs().size(); i++) {
      auto input_tensor = execution_flow->GetInputs()[i];
      auto input_node = inputs[i];
      auto it = anf_tensor_map_.find(input_node);
      if (it != anf_tensor_map_.end()) {
        auto outter_tensor = it->second;
        (*input_isolate_map)[input_tensor] = outter_tensor;
      } else {
        anf_tensor_map_[input_node] = input_tensor;
      }
    }

    for (size_t i = 0; i < execution_flow->GetOutputs().size(); i++) {
      auto output_tensor = execution_flow->GetOutputs()[i];
      auto output_node = outputs[i];
      auto it = anf_tensor_map_.find(output_node);
      if (it != anf_tensor_map_.end()) {
        auto outter_tensor = it->second;
        (*output_isolate_map)[output_tensor] = outter_tensor;
      } else {
        anf_tensor_map_[output_node] = output_tensor;
      }
    }

    execution_plan->AddExecutionFlow(execution_flow);
  }
  execution_plan->SetInputsMap(input_isolate_map);
  execution_plan->SetOutputsMap(output_isolate_map);

  return execution_plan;
}

infer::abstract::Tensor *DefaultGraphCompiler::CreateTensor(AnfNodePtr node) {
  if (node->isa<CNode>()) {
  } else if (node->isa<Parameter>()) {
    auto parameter_node = node->cast<ParameterPtr>();
    if (parameter_node == nullptr) {
      MS_LOG(ERROR) << "parameter node is nullptr";
      return nullptr;
    }
    ShapeVector shape_vector;
    TypeId data_type = kTypeUnknown;
    auto status = GetDTAndShapeFromParameter(parameter_node, &data_type, &shape_vector);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "get data_type and shape failed";
      return nullptr;
    }
    if (data_type == kObjectTypeString) {
      MS_LOG(ERROR) << "Not support String type";
      return nullptr;
    }
    std::vector<int> lite_shape;
    std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(lite_shape),
                   [](int64_t dim) { return static_cast<int>(dim); });
    auto lite_tensor = new lite::Tensor(data_type, lite_shape);
    if (lite_tensor == nullptr) {
      MS_LOG(ERROR) << "New tensor failed, may be memory is not enough";
      return nullptr;
    }
    anf_tensor_map_[node] = lite_tensor;
    return lite_tensor;
  }
  return nullptr;
}

Status DefaultGraphCompiler::GetDTAndShapeFromParameter(ParameterPtr parameter, TypeId *data_type,
                                                        ShapeVector *shape_vector) {
  MS_ASSERT(parameter != nullptr && data_type != nullptr && shape_vector != nullptr);
  auto abstract_base = parameter->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "abstract base is nullptr";
    return kLiteError;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "abstract tensor is nullptr";
    return kLiteError;
  }
  return GetDTAndShapeFromAbTensor(abstract_tensor, data_type, shape_vector);
}

Status DefaultGraphCompiler::GetDTAndShapeFromAbTensor(const abstract::AbstractTensorPtr &abstract, TypeId *data_type,
                                                       ShapeVector *shape_vector) {
  MS_ASSERT(abstract != nullptr && data_type != nullptr && shape_vector != nullptr);
  if (abstract->element() == nullptr) {
    MS_LOG(ERROR) << "'element' of abstract is nullptr";
    return kLiteError;
  }
  auto type_ptr = abstract->element()->GetTypeTrack();
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "type of abstract is nullptr";
    return kLiteError;
  }
  *data_type = type_ptr->type_id();
  if (!utils::isa<abstract::ShapePtr>(abstract->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of Parameter should be ShapePtr";
    return kLiteError;
  }
  *shape_vector = utils::cast<abstract::ShapePtr>(abstract->BuildShape())->shape();
  return kSuccess;
}

std::vector<infer::abstract::Tensor *> DefaultGraphCompiler::CreateTensors(const std::vector<AnfNodePtr> &nodes) {
  std::vector<infer::abstract::Tensor *> tensors;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(tensors),
                 [this](AnfNodePtr node) { return this->CreateTensor(node); });
  return tensors;
}

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> DefaultGraphCompiler::TransformSegmentToAnfGraph(
  const AnfNodePtrList &lst) {
  if (lst.empty()) {
    MS_LOG(EXCEPTION) << "Input anf node list is empty";
  }
  FuncGraphPtr fg = nullptr;
  {
    // limit the lifetime of guard.
    MS_EXCEPTION_IF_NULL(lst[0]->cast<CNodePtr>());
    MS_EXCEPTION_IF_NULL(lst[0]->cast<CNodePtr>()->func_graph());
    TraceGuard guard(std::make_shared<TraceSegmentTransform>(lst[0]->cast<CNodePtr>()->func_graph()->debug_info()));
    fg = std::make_shared<FuncGraph>();
  }
  AnfNodePtrList inputs;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> eqv;
  // Merge CNodes into a AnfGraph that represents a linear instruction segment
  for (auto n : lst) {
    MS_EXCEPTION_IF_NULL(n);
    if (!n->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Inst is not CNode";
    }
    auto &inps = n->cast<CNodePtr>()->inputs();
    if (inps.empty()) {
      MS_LOG(EXCEPTION) << "Input is empty";
    }
    if (!IsValueNode<Primitive>(inps[0]) &&
        !(IsValueNode<FuncGraph>(inps[0]) &&
          inps[0]->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>()->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL))) {
      MS_LOG(EXCEPTION) << "Input[0] must be a Primitive ValueNode";
    }
    auto fn = inps[0];
    std::vector<AnfNodePtr> args{fn};
    if (IsPrimitive(fn, prim::kPrimDepend) && inps.size() >= kDependInputSize &&
        eqv.find(inps[kDependAttachNodeIndex]) == eqv.end()) {
      args.emplace_back(RefSubGraphNode(fg, inps[kRealInputIndexInDepend], &inputs, &eqv));
      const size_t value_start_index = 2;
      for (size_t i = value_start_index; i < inps.size(); ++i) {
        args.emplace_back(NewValueNode(MakeValue(0)));
      }
    } else {
      (void)std::transform(
        std::begin(inps) + 1, std::end(inps), std::back_inserter(args),
        [&fg, &inputs, &eqv, this](const AnfNodePtr &a) { return this->RefSubGraphNode(fg, a, &inputs, &eqv); });
    }
    TraceGuard tg(std::make_shared<TraceSegmentTransform>(n->debug_info()));
    MS_EXCEPTION_IF_NULL(fg);
    eqv[n] = fg->NewCNode(args);
    eqv[n]->set_abstract(n->abstract());
    eqv[n]->set_kernel_info(n->kernel_info_ptr());
  }
  mindspore::HashSet<AnfNodePtr> eqv_keys;
  for (auto &e : eqv) {
    (void)eqv_keys.emplace(e.first);
  }
  auto mgr = lst[0]->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto outputs = GetOutput(lst, mgr->node_users(), eqv_keys);
  AnfNodePtr fg_output;
  if (outputs.size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_args),
                         [&eqv](const AnfNodePtr &o) -> AnfNodePtr { return eqv[o]; });
    // Set output for AnfGraph
    fg_output = fg->NewCNode(output_args);
  } else {
    if (outputs.empty()) {
      MS_LOG(EXCEPTION) << "Output is empty.";
    }
    fg_output = eqv[outputs[0]];
  }
  fg->set_output(fg_output);
  return std::make_tuple(fg, inputs, outputs);
}

AnfNodePtrList DefaultGraphCompiler::GetOutput(const AnfNodePtrList &nodes, const NodeUsersMap &users,
                                               const mindspore::HashSet<AnfNodePtr> &seen) {
  AnfNodePtrList output;
  if (users.size() == 0) {
    return output;
  }
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto iter = users.find(node);
    if (iter == users.end()) {
      continue;
    }
    auto &node_users = iter->second;
    const bool has_outer_user = std::any_of(std::begin(node_users), std::end(node_users),
                                            [&seen](const std::pair<AnfNodePtr, int64_t> &u) -> bool {
                                              const bool is_outer_user = (seen.find(u.first) == seen.end());
                                              return is_outer_user;
                                            });
    if (has_outer_user) {
      output.emplace_back(node);
    }
  }
  return output;
}

AnfNodePtr DefaultGraphCompiler::RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                                 AnfNodePtrList *inputs_ptr,
                                                 mindspore::HashMap<AnfNodePtr, AnfNodePtr> *eqv_ptr) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(inputs_ptr);
  MS_EXCEPTION_IF_NULL(eqv_ptr);
  MS_EXCEPTION_IF_NULL(node);
  auto &inputs = *inputs_ptr;
  auto &eqv = *eqv_ptr;
  if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    eqv[node] = node;
  } else if (eqv.find(node) == eqv.end()) {
    inputs.push_back(node);
    eqv[node] = fg->add_parameter();
    eqv[node]->set_abstract(node->abstract());
    eqv[node]->set_kernel_info(node->kernel_info_ptr());
  }
  return eqv[node];
}

std::shared_ptr<infer::abstract::ExecutionFlow> DefaultGraphCompiler::Schedule(const GraphSegmentPtr &graph_segment,
                                                                               const std::vector<AnfNodePtr> &inputs,
                                                                               const std::vector<AnfNodePtr> &outputs) {
  // implementation by hangangqiang
  return nullptr;
}

static std::shared_ptr<infer::abstract::GraphCompiler> DefaultGraphCompilerCreator(
  const std::shared_ptr<Context> &ctx) {
  auto graph_compiler = std::make_shared<DefaultGraphCompiler>(ctx);
  return graph_compiler;
}
REG_GRAPH_COMPILER(kDefaultCompiler, DefaultGraphCompilerCreator);
}  // namespace mindspore
