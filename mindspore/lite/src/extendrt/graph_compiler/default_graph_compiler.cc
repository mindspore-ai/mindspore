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
#include "extendrt/utils/func_graph_utils.h"
#include "ir/manager.h"
#include "base/base_ref.h"
#include "abstract/abstract_value.h"
#include "extendrt/graph_compiler/single_graph_compiler.h"
#include "extendrt/graph_compiler/anfnode_tensor_adapter.h"

namespace mindspore {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static constexpr auto ms_infer_backend_name = "mindspore_lite_backend";

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Compile(FuncGraphPtr graph) {
  MS_LOG(INFO) << "DefaultGraphCompiler::Compile";

  inner_context_ = ContextUtils::Convert(context_.get());
  if (inner_context_->Init() != RET_OK) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile init inner context failed";
    return nullptr;
  }

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Partition FunctionGraph Begin";
  auto graph_segments = Partition(graph);
  if (graph_segments.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile partition graph failed";
    return nullptr;
  }
  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Partition FunctionGraph End";

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Schedule Graph Execute Plan Begin";
  auto execution_plan = Schedule(graph_segments, graph);
  if (execution_plan == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile Schedule graph segments failed";
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
  // convert the graph to NHWC format, this is because current nnacl ops only support NHWC
  auto status = FuncGraphUtils::UnifyGraphToNHWCFormat(graph);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Partition unify graph to NHWC failed";
    return std::vector<GraphSegmentPtr>();
  }

  // multi_target set false
  bool is_multi_target;
  return partition->Partition(graph, &is_multi_target);
}

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Schedule(
  const std::vector<GraphSegmentPtr> &graph_segments, FuncGraphPtr func_graph) {
  auto execution_plan = std::make_shared<infer::ExecutionPlan>();
  anf_tensor_map_.clear();

  // set func graph manager
  auto func_manager = func_graph->manager();
  if (func_manager == nullptr) {
    func_manager = Manage(func_graph, true);
    func_graph->set_manager(func_manager);
  }

  // Convert FuncGraph Input and Output AnfNode to Tensor and save in Execution Plan
  auto graph_inputs = func_graph->get_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph inputs node failed";
    return nullptr;
  }
  std::vector<AnfNodePtr> graph_outputs;
  auto graph_output = func_graph->output();
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule get graph output node failed";
    return nullptr;
  }
  graph_outputs.emplace_back(graph_output);
  std::vector<infer::abstract::Tensor *> graph_input_tensors;
  auto graph_output_tensors = CreateTensors(graph_outputs);
  if (graph_output_tensors.size() != graph_outputs.size()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Schedule create graph output tensor failed";
    return nullptr;
  }
  for (size_t i = 0; i < graph_outputs.size(); i++) {
    auto output_node = graph_outputs[i];
    auto output_tensor = graph_output_tensors[i];
    auto it = anf_tensor_map_.find(output_node);
    if (it == anf_tensor_map_.end()) {
      anf_tensor_map_[output_node] = output_tensor;
    }
  }
  execution_plan->SetOutputs(graph_output_tensors);
  execution_plan->SetContext(inner_context_);
  std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *> *input_isolate_map =
    new std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *>();
  std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *> *output_isolate_map =
    new std::unordered_map<infer::abstract::Tensor *, infer::abstract::Tensor *>();
  for (auto graph_segment : graph_segments) {
    if (graph_segment->nodes_.size() == 1) {
      auto node = graph_segment->nodes_[0];
      if (node->isa<CNode>()) {
        auto cnode = node->cast<CNodePtr>();
        auto inps = cnode->inputs();
        if (!inps.empty()) {
          auto primitive = inps[0];
          if (IsPrimitive(primitive, prim::kPrimReturn)) {
            continue;
          }
        }
      }
    }
    FuncGraphPtr fg = nullptr;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = FuncGraphUtils::TransformSegmentToAnfGraph(graph_segment->nodes_);
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
      auto it = std::find_if(graph_inputs.begin(), graph_inputs.end(),
                             [&input_node](const AnfNodePtr &node) { return node == input_node; });
      if (it != graph_inputs.end()) {
        input_tensor->set_category(lite::GRAPH_INPUT);
        graph_input_tensors.emplace_back(input_tensor);
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
  execution_plan->SetInputs(graph_input_tensors);
  execution_plan->SetInputsMap(input_isolate_map);
  execution_plan->SetOutputsMap(output_isolate_map);

  return execution_plan;
}

infer::abstract::Tensor *DefaultGraphCompiler::CreateTensor(AnfNodePtr node) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor cnode is nullptr";
      return nullptr;
    }
    auto abstract = cnode->abstract();
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor abstract is nullptr";
      return nullptr;
    }
    if (utils::isa<abstract::AbstractTensorPtr>(abstract)) {
      auto tensor = infer::TensorAdapter::Convert2Tensor(abstract, node->fullname_with_scope());
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensor from abstract failed, abstract : " << abstract;
        return nullptr;
      }
      return tensor;
    }
  } else if (node->isa<Parameter>()) {
    auto parameter_node = node->cast<ParameterPtr>();
    if (parameter_node == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor parameter node is nullptr";
      return nullptr;
    }
    ShapeVector shape_vector;
    TypeId data_type = kTypeUnknown;
    auto status = GetDTAndShapeFromParameter(parameter_node, &data_type, &shape_vector);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor get data_type and shape failed";
      return nullptr;
    }
    if (data_type == kObjectTypeString) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor not support String type";
      return nullptr;
    }
    std::vector<int> lite_shape;
    std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(lite_shape),
                   [](int64_t dim) { return static_cast<int>(dim); });
    auto lite_tensor = new lite::Tensor(data_type, lite_shape);
    if (lite_tensor == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateTensor new tensor failed, may be memory is not enough";
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

std::shared_ptr<infer::abstract::ExecutionFlow> DefaultGraphCompiler::Schedule(const GraphSegmentPtr &graph_segment,
                                                                               const std::vector<AnfNodePtr> &inputs,
                                                                               const std::vector<AnfNodePtr> &outputs) {
  // implementation by hangangqiang
  auto compiler = std::make_shared<mindspore::infer::SingleGraphCompiler>(inner_context_);
  infer::abstract::CompileOption option;
  return compiler->Compile(graph_segment, inputs, outputs, option);
}

static std::shared_ptr<infer::abstract::GraphCompiler> DefaultGraphCompilerCreator(
  const std::shared_ptr<Context> &ctx) {
  auto graph_compiler = std::make_shared<DefaultGraphCompiler>(ctx);
  return graph_compiler;
}
REG_GRAPH_COMPILER(kDefaultCompiler, DefaultGraphCompilerCreator);
}  // namespace mindspore
