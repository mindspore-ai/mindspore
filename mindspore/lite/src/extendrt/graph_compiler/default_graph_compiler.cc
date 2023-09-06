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
#include "src/common/common.h"
#include "extendrt/delegate/factory.h"

namespace mindspore::lite {
static const std::vector<PrimitivePtr> ms_infer_cut_list = {prim::kPrimReturn, prim::kPrimPartial, prim::kPrimSwitch,
                                                            prim::kPrimBpropCut, prim::kPrimSwitchLayer};
static constexpr auto ms_infer_backend_name = "mindspore_lite_backend";

void DefaultGraphCompiler::InitCompileOption(const FuncGraphPtr &graph) {
  if (option_ != nullptr) {
    MS_LOG(INFO) << "CompileOption is already inited.";
    return;
  }
  option_ = std::make_shared<CompileOption>();
  auto format_value = graph->get_attr(mindspore::ops::kFormat);
  if (format_value != nullptr) {
    option_->graph_format = Format(GetValue<int64_t>(format_value));
  }
  auto input_format_value = graph->get_attr(kInputFormat);
  if (input_format_value != nullptr) {
    option_->graph_input_format = Format(GetValue<int32_t>(input_format_value));
  }

  if (inner_context_->IsDeviceTypeEnabled(lite::DT_ASCEND)) {
    option_->backend = kernel::kBackendAscend;
  }
}

void DefaultGraphCompiler::ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) {
  const ConfigInfos config_infos;
  auto &device_contexts = context_->MutableDeviceInfo();
  if (device_contexts.empty()) {
    MS_LOG(ERROR) << "no context found";
  }
  auto device_type = device_contexts.at(0)->GetDeviceType();
  auto provider = device_contexts.at(0)->GetProvider();
  auto delegate =
    DelegateRegistry<ExtendDelegate *>::GetInstance().GetDelegate(device_type, provider, context_, config_infos);
  if (delegate != nullptr) {
    delegate->ReplaceNodes(graph);
  }
  // other delegate // implement by plugin later
}

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::Compile(FuncGraphPtr graph) {
  MS_LOG(INFO) << "DefaultGraphCompiler::Compile";

  inner_context_ = ContextUtils::Convert(context_.get());
  if (inner_context_ == nullptr || inner_context_->Init() != RET_OK) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::Compile init inner context failed";
    return nullptr;
  }

  InitCompileOption(graph);

  ReplaceNodes(graph);

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
  auto builder = std::make_shared<CompileResultBuilder>(option_);
  return builder->Build(segment, inputs, outputs);
}

std::vector<InferKernel *> DefaultGraphCompiler::Schedule(const CompileResultPtr &compile_result) {
  if (MS_UNLIKELY(scheduler_ == nullptr)) {
    scheduler_ = std::make_shared<SingleGraphScheduler>(this->inner_context_, context_, option_);
  }
  return {scheduler_->Schedule(compile_result)};
}

std::vector<AnfNodePtr> DefaultGraphCompiler::SkipMakeTuple(const AnfNodePtr &origin_node) {
  if (!origin_node->isa<CNode>()) {
    // not cnode, return origin node
    return {origin_node};
  }
  auto cnode = origin_node->cast<CNodePtr>();
  MS_ASSERT(cnode != nullptr);

  if (!IsPrimitive(cnode->input(0), prim::kPrimMakeTuple)) {
    return {origin_node};
  }

  std::vector<AnfNodePtr> results;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto real_nodes = SkipMakeTuple(cnode->input(i));
    results.insert(results.end(), real_nodes.begin(), real_nodes.end());
  }
  return results;
}

Status DefaultGraphCompiler::UpdateSubGraphInoutMap(const kernel::KernelExec &subgraph, const AnfNodePtrList &inputs,
                                                    const AnfNodePtrList &outputs) {
  // add subgraph_input_map: subgraph.input-tensors --> anfnode
  auto count = inputs.size();
  // not support tuple as an input of cnode now except return and tuplegetitem, but return is skipped before and
  // tuplegetitem is not a cut point.
  if (MS_UNLIKELY(count != subgraph.in_tensors().size())) {
    MS_LOG(ERROR) << "Subgraph has " << subgraph.in_tensors().size() << " inputs while segment has " << count
                  << " inputs.";
    return kLiteError;
  }
  for (size_t i = 0; i < count; i++) {
    subgraph_input_map_[subgraph.in_tensors()[i]] = inputs[i];
  }
  // add subgraph_output_map: anfnode --> subgraph.output-tensors
  count = outputs.size();
  // not support tuple as an input of cnode now except return and tuplegetitem, but return is skipped before and
  // tuplegetitem is not a cut point.
  if (MS_UNLIKELY(count != subgraph.out_tensors().size())) {
    MS_LOG(ERROR) << "Subgraph has " << subgraph.out_tensors().size() << " outputs while segment has " << count
                  << " outputs.";
    return kLiteError;
  }
  for (size_t i = 0; i < count; i++) {
    subgraph_output_map_[outputs[i]] = subgraph.out_tensors()[i];
  }
  return kSuccess;
}

std::tuple<AnfNodePtrList, AnfNodePtrList> DefaultGraphCompiler::GetSegmentInout(const GraphSegment &graph_segment) {
  FuncGraphPtr fg = nullptr;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = FuncGraphUtils::TransformSegmentToAnfGraph(graph_segment.nodes_);
  // TransformSegmentToAnfGraph puts all input and weight into 'inputs'. In inference, we erase weight.
  for (auto iter = inputs.begin(); iter != inputs.end();) {
    if (utils::isa<ParameterPtr>(*iter) && (utils::cast<ParameterPtr>(*iter))->has_default()) {
      iter = inputs.erase(iter);
    } else {
      iter++;
    }
  }
  // maketuple and tuplegetitem make nosense in inference, skip nodes with these types for outputs
  AnfNodePtrList real_outputs;
  real_outputs.reserve(outputs.size());
  for (auto &output : outputs) {
    std::vector<AnfNodePtr> seg_outputs = SkipMakeTuple(output);
    real_outputs.insert(real_outputs.end(), seg_outputs.begin(), seg_outputs.end());
  }
  return std::make_tuple(inputs, real_outputs);
}

Status DefaultGraphCompiler::CreateExecPlanKernels(const std::vector<GraphSegmentPtr> &graph_segments,
                                                   std::vector<AnfNodePtrList> *segments_outputs) {
  MS_ASSERT(execution_plan_ != nullptr);
  for (const auto &graph_segment : graph_segments) {
    if (graph_segment == nullptr) {
      MS_LOG(ERROR) << "graph segment is nullptr.";
      return kLiteNullptr;
    }
    if (graph_segment->nodes_.size() == 1) {
      auto &node = graph_segment->nodes_[0];
      if (opt::CheckPrimitiveType(node, prim::kPrimReturn)) {
        continue;
      }
    }
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(inputs, outputs) = GetSegmentInout(*graph_segment);
    // maketuple tuplegetitem is deleted inside of Compile
    auto compile_result = this->Compile(graph_segment, inputs, outputs);
    if (compile_result == nullptr) {
      MS_LOG(ERROR) << "DefaultGraphCompiler::CreateExecPlanKernels convert to CompileResult failed";
      return kLiteError;
    }
    auto kernels = this->Schedule(compile_result);
    if (kernels.size() != 1) {
      MS_LOG(ERROR) << "Only support one subgraph from one graph segment now, got " << kernels.size();
      return kLiteError;
    }
    auto kernel = kernels[0];
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Schedule failed, return nullptr.";
      return kLiteError;
    }
    auto ret = UpdateSubGraphInoutMap(*kernel, inputs, outputs);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateSubGraphInoutMap failed: " << ret;
      return ret;
    }
    segments_outputs->emplace_back(outputs);
    execution_plan_->AddKernel(kernel);
  }
  return kSuccess;
}

Status DefaultGraphCompiler::CreateExecPlanInputs(const FuncGraphPtr &func_graph) {
  MS_ASSERT(graph_input_tensors_.empty());
  auto graph_inputs = func_graph->get_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile get graph inputs node failed";
    return kLiteError;
  }
  for (const auto &input : func_graph->get_inputs()) {
    if (!utils::isa<ParameterPtr>(input)) {
      MS_LOG(ERROR) << "Not supported graph input: " << input;
      return kLiteError;
    }
    auto parameter = utils::cast<ParameterPtr>(input);
    auto tensor = TensorAdapter::Convert2Tensor(parameter, option_->graph_input_format);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create graph input tensor failed, input : " << input->fullname_with_scope();
      return kLiteError;
    }
    tensor->set_category(GRAPH_INPUT);
    graph_input_tensors_.push_back(tensor);
    subgraph_output_map_[input] = tensor;
  }
  execution_plan_->SetInputs(graph_input_tensors_);
  return kSuccess;
}

Status DefaultGraphCompiler::CreateExecPlanOutputs(const FuncGraphPtr &func_graph,
                                                   const std::vector<AnfNodePtrList> &segments_outputs) {
  MS_ASSERT(execution_plan_ != nullptr);
  anf_tensor_map_.clear();
  auto graph_output = func_graph->output();
  if (graph_output == nullptr) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile get graph output node failed";
    return kLiteError;
  }
  std::vector<AnfNodePtr> graph_outputs = SkipMakeTuple(graph_output);

  auto graph_output_tensors = DefaultGraphCompiler::CreateTensors(graph_outputs);
  if (graph_output_tensors.size() != graph_outputs.size()) {
    MS_LOG(ERROR) << "DefaultGraphCompiler::NonCFGCompile create graph output tensor failed";
    return kLiteError;
  }
  for (size_t i = 0; i < graph_outputs.size(); i++) {
    auto output_node = graph_outputs[i];
    auto output_tensor = graph_output_tensors[i];
    auto it = anf_tensor_map_.find(output_node);
    if (it != anf_tensor_map_.end()) {
      MS_LOG(ERROR) << "Can not find corresponding tensor for graph output node: "
                    << output_node->fullname_with_scope();
      return kLiteError;
    }
    anf_tensor_map_[output_node] = output_tensor;
  }
  execution_plan_->SetOutputs(graph_output_tensors);

  auto *output_isolate_map = new std::unordered_map<InferTensor *, InferTensor *>();
  for (size_t i = 0; i < execution_plan_->GetKernels().size(); i++) {
    auto kernel = execution_plan_->GetKernels()[i];
    if (MS_UNLIKELY(kernel->out_tensors().size() != segments_outputs[i].size())) {
      MS_LOG(ERROR) << "Subgraph has " << kernel->in_tensors().size() << " outputs while segment has "
                    << segments_outputs[i].size() << " outputs.";
      delete output_isolate_map;
      return kLiteError;
    }
    for (size_t j = 0; j < kernel->out_tensors().size(); j++) {
      auto output_tensor = kernel->out_tensors()[j];
      auto &output_node = segments_outputs[i][j];
      auto it = anf_tensor_map_.find(output_node);
      if (it != anf_tensor_map_.end()) {
        auto outter_tensor = it->second;
        (*output_isolate_map)[output_tensor] = outter_tensor;
      }
    }
  }
  execution_plan_->SetOutputsMap(output_isolate_map);
  return kSuccess;
}

Status DefaultGraphCompiler::IsolateSubGraphs() {
  auto *subgraph_isolate_map = new std::unordered_map<InferTensor *, InferTensor *>();
  for (auto &kernel : execution_plan_->GetKernels()) {
    auto &in_tensors = kernel->in_tensors();
    for (size_t i = 0; i < in_tensors.size(); i++) {
      auto &input = in_tensors[i];
      auto anf_node = subgraph_input_map_.find(input);
      if (anf_node == subgraph_input_map_.end()) {
        MS_LOG(ERROR) << "Can not find corresponding anf_node for " << i << "th input of subgraph " << kernel->name();
        delete subgraph_isolate_map;
        return kLiteError;
      }
      auto output = subgraph_output_map_.find(anf_node->second);
      if (output == subgraph_output_map_.end()) {
        MS_LOG(ERROR) << "Can not find corresponding output tensor for anf_node: "
                      << anf_node->second->fullname_with_scope();
        delete subgraph_isolate_map;
        return kLiteError;
      }
      (*subgraph_isolate_map)[input] = output->second;
    }
  }
  this->execution_plan_->SetInputsMap(subgraph_isolate_map);
  return kSuccess;
}

std::shared_ptr<infer::abstract::ExecutionPlan> DefaultGraphCompiler::NonCFGCompile(
  const std::vector<GraphSegmentPtr> &graph_segments, const FuncGraphPtr &func_graph) {
  execution_plan_ = std::make_shared<infer::ExecutionPlan>();
  execution_plan_->SetContext(inner_context_);

  // set func graph manager
  auto func_manager = func_graph->manager();
  if (func_manager == nullptr) {
    func_manager = Manage(func_graph, true);
    func_graph->set_manager(func_manager);
  }
  std::vector<AnfNodePtrList> segments_outputs;
  auto ret = CreateExecPlanKernels(graph_segments, &segments_outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create graph subgraphs failed";
    return nullptr;
  }
  ret = CreateExecPlanInputs(func_graph);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create graph input tensors failed";
    return nullptr;
  }
  ret = CreateExecPlanOutputs(func_graph, segments_outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create graph output tensors failed";
    return nullptr;
  }
  ret = IsolateSubGraphs();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Isolate subgraphs failed";
    return nullptr;
  }
  return execution_plan_;
}

std::vector<InferTensor *> DefaultGraphCompiler::CreateTensors(const std::vector<AnfNodePtr> &nodes) {
  std::vector<InferTensor *> tensors;
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto tmp = TensorAdapter::Convert2Tensor(cnode);
      if (tmp.empty()) {
        MS_LOG(ERROR) << "Create tensors from cnode failed, node : " << node->fullname_with_scope();
        for (auto tensor : tensors) {
          delete tensor;
        }
        return {};
      }
      (void)tensors.insert(tensors.cend(), tmp.begin(), tmp.end());
      continue;
    }
    if (node->isa<Parameter>()) {
      auto param_node = node->cast<ParameterPtr>();
      auto tensor = TensorAdapter::Convert2Tensor(param_node);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensors from parameter failed, node : " << node->fullname_with_scope();
        return {};
      }
      tensors.emplace_back(tensor);
      continue;
    }
    if (node->isa<ValueNode>()) {
      auto value_node = node->cast<ValueNodePtr>();
      auto tensor = TensorAdapter::Convert2Tensor(value_node);
      if (tensor == nullptr) {
        MS_LOG(ERROR) << "Create tensors from value node failed, node : " << node->fullname_with_scope();
        return {};
      }
      tensors.emplace_back(tensor);
      continue;
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
