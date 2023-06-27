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

#include <algorithm>
#include <unordered_map>

#include "abstract/abstract_value.h"
#include "backend/graph_compiler/graph_partition.h"
#include "base/base_ref.h"
#include "extendrt/execution_plan.h"
#include "extendrt/graph_compiler/anfnode_tensor_adapter.h"
#include "extendrt/graph_compiler/default_graph_compiler.h"
#include "extendrt/graph_compiler/factory.h"
#include "extendrt/mock/lite_runtime/converters.h"
#include "extendrt/utils/func_graph_utils.h"
#include "ir/manager.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/op_name.h"
#include "src/extendrt/graph_compiler/compile_result_builder.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn,   prim::kPrimPartial,
                                                            prim::kPrimSwitch,   prim::kPrimMakeTuple,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static constexpr auto ms_infer_backend_name = "mindspore_lite_backend";

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Compile(FuncGraphPtr graph) {
  MS_LOG(INFO) << "DefaultGraphCompiler::Compile";

  inner_context_ = ContextUtils::Convert(context_.get());
  if (inner_context_ == nullptr || inner_context_->Init() != RET_OK) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile init inner context failed";
    return nullptr;
  }

  option_ = std::make_shared<CompileOption>();
  auto format_value = graph->get_attr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    option_->format = Format(GetValue<int64_t>(format_value));
  }

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Partition FunctionGraph Begin";
  auto graph_segments = Partition(graph);
  if (graph_segments.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile partition graph failed";
    return nullptr;
  }
  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Partition FunctionGraph End";

  MS_LOG(DEBUG) << "DefaultGraphCompiler::Compile Schedule Graph Execute Plan Begin";
  auto execution_plan = NonCFGCompile(graph_segments, graph);
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
    return {};
  }

  // multi_target set false
  bool is_multi_target;
  return partition->Partition(graph, &is_multi_target);
}

CompileResultPtr DefaultGraphCompiler::Compile(const GraphSegmentPtr &segment, const std::vector<AnfNodePtr> &inputs,
                                               const std::vector<AnfNodePtr> &outputs) {
  auto builder = std::make_shared<CompileResultBuilder>(option_->format);
  return builder->Build(segment, inputs, outputs);
}

std::vector<InferKernel *> DefaultGraphCompiler::Schedule(const CompileResultPtr &compile_result) {
  if (MS_UNLIKELY(scheduler_ == nullptr)) {
    scheduler_ = std::make_shared<SingleGraphScheduler>(this->inner_context_, option_);
  }
  return {scheduler_->Schedule(compile_result)};
}

std::vector<AnfNodePtr> DefaultGraphCompiler::GetGraphOutput(AnfNodePtr origin_output) {
  std::vector<AnfNodePtr> graph_outputs;
  if (!origin_output->isa<CNode>()) {
    // not cnode, return origin output node
    graph_outputs.emplace_back(origin_output);
    return graph_outputs;
  }

  // cnode, if MakeTuple, split it into multiple output node
  auto cnode = origin_output->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cast origin output node into cnode failed";
    return {};
  }

  if (!IsPrimitive(cnode->input(0), prim::kPrimMakeTuple)) {
    // not MakeTuple node, return origin output node
    graph_outputs.emplace_back(origin_output);
    return graph_outputs;
  }

  // MakeTuple Node, get the input node
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    graph_outputs.emplace_back(cnode->input(i));
  }
  return graph_outputs;
}

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::NonCFGCompile(
  const std::vector<GraphSegmentPtr> &graph_segments, const FuncGraphPtr &func_graph) {
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
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile get graph inputs node failed";
    return nullptr;
  }
  auto graph_output = func_graph->output();
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile get graph output node failed";
    return nullptr;
  }
  std::vector<AnfNodePtr> graph_outputs = GetGraphOutput(graph_output);
  std::vector<InferTensor *> graph_input_tensors;
  auto graph_output_tensors = CreateTensors(graph_outputs);
  if (graph_output_tensors.size() != graph_outputs.size()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile create graph output tensor failed";
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
  auto *output_isolate_map = new std::unordered_map<InferTensor *, InferTensor *>();
  for (const auto &graph_segment : graph_segments) {
    if (graph_segment->nodes_.size() == 1) {
      auto &node = graph_segment->nodes_[0];
      if (opt::CheckPrimitiveType(node, prim::kPrimReturn) || opt::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
        continue;
      }
    }
    FuncGraphPtr fg = nullptr;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = FuncGraphUtils::TransformSegmentToAnfGraph(graph_segment->nodes_);
    auto compile_result = this->Compile(graph_segment, inputs, outputs);
    if (compile_result == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile convert to CompileResult failed";
      delete output_isolate_map;
      return nullptr;
    }
    auto kernels = this->Schedule(compile_result);
    if (kernels.size() != 1 || kernels[0] == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile schedule graph segment failed";
      delete output_isolate_map;
      return nullptr;
    }
    auto kernel = kernels[0];
    kernel->set_context(inner_context_.get());
    for (size_t i = 0; i < kernel->in_tensors().size(); i++) {
      auto input_tensor = kernel->in_tensors()[i];
      auto input_node = inputs[i];
      auto it = std::find_if(graph_inputs.begin(), graph_inputs.end(),
                             [&input_node](const AnfNodePtr &node) { return node == input_node; });
      if (it != graph_inputs.end()) {
        input_tensor->set_category(lite::GRAPH_INPUT);
        graph_input_tensors.emplace_back(input_tensor);
      }
    }
    for (size_t i = 0; i < kernel->out_tensors().size(); i++) {
      auto output_tensor = kernel->out_tensors()[i];
      auto output_node = outputs[i];
      auto it = anf_tensor_map_.find(output_node);
      if (it != anf_tensor_map_.end()) {
        auto outter_tensor = it->second;
        (*output_isolate_map)[output_tensor] = outter_tensor;
      } else {
        anf_tensor_map_[output_node] = output_tensor;
      }
    }
    execution_plan->AddKernel(kernel);
  }
  execution_plan->SetInputs(graph_input_tensors);
  execution_plan->SetOutputsMap(output_isolate_map);

  return execution_plan;
}

InferTensor *DefaultGraphCompiler::CreateTensor(const AnfNodePtr &node) {
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
    if (utils::isa<mindspore::abstract::AbstractTensorPtr>(abstract)) {
      auto tensor = TensorAdapter::Convert2Tensor(abstract, abstract->name());
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

Status DefaultGraphCompiler::GetDTAndShapeFromParameter(const ParameterPtr &parameter, TypeId *data_type,
                                                        ShapeVector *shape_vector) {
  MS_ASSERT(parameter != nullptr && data_type != nullptr && shape_vector != nullptr);
  auto abstract_base = parameter->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "abstract base is nullptr";
    return kLiteError;
  }
  auto abstract_tensor = utils::cast<mindspore::abstract::AbstractTensorPtr>(abstract_base);
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "abstract tensor is nullptr";
    return kLiteError;
  }
  return GetDTAndShapeFromAbTensor(abstract_tensor, data_type, shape_vector);
}

Status DefaultGraphCompiler::GetDTAndShapeFromAbTensor(const mindspore::abstract::AbstractTensorPtr &abstract,
                                                       TypeId *data_type, ShapeVector *shape_vector) {
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
  if (!utils::isa<mindspore::abstract::ShapePtr>(abstract->BuildShape())) {
    MS_LOG(ERROR) << "Shape of Abstract of Parameter should be ShapePtr";
    return kLiteError;
  }
  *shape_vector = utils::cast<mindspore::abstract::ShapePtr>(abstract->BuildShape())->shape();
  return kSuccess;
}

std::vector<InferTensor *> DefaultGraphCompiler::CreateTensors(const std::vector<AnfNodePtr> &nodes) {
  std::vector<InferTensor *> tensors;
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        MS_LOG(ERROR) << "cast to CNode with nullptr";
        return {};
      }
      auto abstract = cnode->abstract();
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "get abstract is nullptr for node " << node->fullname_with_scope();
        return {};
      }
      if (abstract->isa<mindspore::abstract::AbstractTuple>()) {
        auto abstract_tuple = abstract->cast<mindspore::abstract::AbstractTuplePtr>();
        if (abstract_tuple == nullptr) {
          MS_LOG(ERROR) << "cast to Abstract Tuple with nullptr";
          return {};
        }
        auto elements = abstract_tuple->elements();
        for (const auto &element : elements) {
          if (utils::isa<mindspore::abstract::AbstractTensorPtr>(element)) {
            auto tensor = TensorAdapter::Convert2Tensor(element, element->name());
            if (tensor == nullptr) {
              MS_LOG(ERROR) << "Create tensor from abstract failed, abstract : " << element;
              return {};
            }
            tensors.emplace_back(tensor);
          }
        }
      } else {
        auto tensor = this->CreateTensor(node);
        tensors.emplace_back(tensor);
      }
    } else {
      // node must be Parameter
      auto tensor = this->CreateTensor(node);
      tensors.emplace_back(tensor);
    }
  }
  return tensors;
}

static std::shared_ptr<infer::abstract::GraphCompiler> DefaultGraphCompilerCreator(
  const std::shared_ptr<Context> &ctx) {
  auto graph_compiler = std::make_shared<DefaultGraphCompiler>(ctx);
  return graph_compiler;
}
REG_GRAPH_COMPILER(kDefaultCompiler, DefaultGraphCompilerCreator);
}  // namespace mindspore::lite
