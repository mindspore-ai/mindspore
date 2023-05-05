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

#include "plugin/device/ascend/hal/hardware/ge_graph_executor.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "include/backend/kernel_graph.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/optimizer/ge_optimization.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "runtime/dev.h"
#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};

void GetMeRetDataType(const AbstractBasePtr &cnode_data, std::vector<TypeId> *me_types) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<abstract::AbstractTensor>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    if (me_type == kObjectTypeTensorType) {
      me_type = dyn_cast<TensorType>(cnode_data->BuildType())->element()->type_id();
      me_types->emplace_back(me_type);
    }
    return;
  }
  if (cnode_data->isa<abstract::AbstractScalar>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    me_types->emplace_back(me_type);
  }
  auto abstract_tuple = cnode_data->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto elements = abstract_tuple->elements();
  for (size_t i = 0; i < abstract_tuple->size(); ++i) {
    GetMeRetDataType(elements[i], me_types);
  }
}

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      // need ref shape when auto parallel
      auto build_shape = para->abstract()->BuildShape();
      if (build_shape != nullptr) {
        (void)tensor->set_shape(build_shape->cast<abstract::ShapePtr>()->shape());
        MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
      }
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

std::vector<transform::GeTensorPtr> GetInputTensors(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_input_map;
  std::vector<tensor::TensorPtr> init_input;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      (void)init_input_map.emplace(para->name(), value->cast<std::shared_ptr<tensor::Tensor>>());
    }
  }
  (void)std::transform(init_input_map.begin(), init_input_map.end(), std::back_inserter(init_input),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });
  return transform::ConvertInputTensors(init_input, kOpFormat_NCHW);
}

void RunGEInitGraph(const FuncGraphPtr &anf_graph) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";

  std::vector<transform::GeTensorPtr> ge_outputs;
  transform::RunOptions run_options;

  run_options.name = "init_subgraph." + anf_graph->ToString();
  if (transform::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(WARNING) << "Can not find " << run_options.name
                    << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  std::vector<transform::GeTensorPtr> ge_tensors;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }

    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (transform::GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ge_tensors = GetInputTensors(anf_graph);
      ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(INFO) << "Exec broadcast graph success.";
    }
  }
}

void UpdateOutputNodeShape(const AnfNodePtr &node, size_t index, TypeId output_type, const ShapeVector &output_shape) {
  MS_EXCEPTION_IF_NULL(node);
  std::string name;
  if (node->isa<CNode>()) {
    name = common::AnfAlgo::GetCNodeName(node);
  }
  size_t total_output_num = AnfAlgo::GetOutputElementNum(node);
  if (index >= total_output_num) {
    MS_LOG(EXCEPTION) << "Invalid output index " << index << ", node " << node->fullname_with_scope() << " has "
                      << total_output_num << " outputs.";
  }
  std::vector<TypeId> types = {};
  std::vector<ShapeVector> shapes = {};
  for (size_t i = 0; i < total_output_num; ++i) {
    if (i == index && kIgnoreGEShapeOps.count(name) == 0) {
      types.push_back(output_type);
      shapes.push_back(output_shape);
    } else {
      types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
      shapes.emplace_back(common::AnfAlgo::GetOutputInferShape(node, i));
    }
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, node.get());
}

void SetDynamicShapeAttr(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->output());
  for (auto &node : nodes) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(INFO) << "Set Dynamic Shape Attr to Node : " << node->fullname_with_scope();
      kernel_graph->SetGraphDynamicAttr(true);
      return;
    }
  }
}
}  // namespace

void GeGraphExecutor::AllocInputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &inputs = kernel_graph->inputs();
  for (const auto &input : inputs) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(input, 0)};
    builder->SetOutputsDeviceType(output_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input.get());
  }

  for (const auto &input_node : inputs) {
    if (!input_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << input_node->fullname_with_scope() << " is not parameter, continue";
      continue;
    }
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);

    size_t tensor_size;
    if (kernel_graph->is_dynamic_shape()) {
      tensor_size = 0;
    } else {
      std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    }

    auto device_address_ptr =
      std::make_shared<cpu::CPUDeviceAddress>(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id, kCPUDevice, 0);
    device_address_ptr->set_is_ptr_persisted(false);
    AnfAlgo::SetOutputAddr(device_address_ptr, 0, input_node.get());
  }
}

void GeGraphExecutor::AllocOutputHostMemory(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (output_node->isa<Parameter>()) {
      continue;
    }

    auto i = output_with_index.second;
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, i);
    auto output_device_addr =
      std::make_shared<cpu::CPUDeviceAddress>(nullptr, 0, kOpFormat_DEFAULT, output_type_id, kCPUDevice, 0);
    AnfAlgo::SetOutputAddr(output_device_addr, i, output_node.get());

    if (common::AnfAlgo::IsNopNode(output_node)) {
      auto [real_node, real_idx] = common::AnfAlgo::GetPrevNodeOutput(output_node, i, true);
      if (real_node != output_node || real_idx != i) {
        AnfAlgo::SetOutputAddr(output_device_addr, real_idx, real_node.get());
      }
    }
  }
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  AscendGraphOptimization::GetInstance().OptimizeACLGraph(kg);
  (void)BuildDFGraph(kg, GetParams(kg), false);
  SetDynamicShapeAttr(kg);
  AllocInputHostMemory(kg);
  AllocOutputHostMemory(kg);
  kg->set_run_mode(RunMode::kGraphMode);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    kg->set_is_loop_count_sink(true);
  }
  // copy init weight to device
  RunGEInitGraph(kg);
  return true;
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "GE run graph " << graph_name << " start.";
  // copy input from device to host
  const auto &cur_inputs = graph->get_inputs();
  std::vector<tensor::TensorPtr> input_tensors;
  for (const auto &input : cur_inputs) {
    MS_EXCEPTION_IF_NULL(input);
    auto output_addr = AnfAlgo::GetMutableOutputAddr(input, 0);
    auto shapes = trans::GetRuntimePaddingShape(input, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(input, 0);
    auto tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(output_addr, false);
    tensor->data_sync();
    input_tensors.emplace_back(std::move(tensor));
  }
  auto ge_inputs = transform::ConvertInputTensors(input_tensors, kOpFormat_NCHW);

  // call ge rungraph
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  if (kg != nullptr) {
    graph_name = kg->GetFuncGraph()->ToString();
  }
  transform::RunOptions run_options;
  run_options.name = graph_name;
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  AnfNodePtr output = graph->get_return()->input(1);
  MS_EXCEPTION_IF_NULL(output);
  std::vector<TypeId> me_types;
  auto output_c = output->cast<CNodePtr>()->abstract();
  // get output node data types
  GetMeRetDataType(output_c, &me_types);
  std::vector<transform::GeTensorPtr> ge_outputs;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
    transform::Status ret = transform::RunGraphAsync(graph_runner, run_options, ge_inputs, &ge_outputs);
    MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }
  if (me_types.size() != ge_outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size, me_type's size " << me_types.size() << " tensor size "
                      << ge_outputs.size();
  }
  // copy output from host to device
  auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  if (graph_outputs.size() != ge_outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size, graph's size " << graph_outputs.size() << " tensor size "
                      << ge_outputs.size();
  }

  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    const auto &tensor = ge_outputs[i];
    auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx);
    ::ge::Placement dp = tensor->GetTensorDesc().GetPlacement();
    auto &&ge_data_uni = tensor->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    MS_EXCEPTION_IF_NULL(ge_data);
    if (dp == ::ge::kPlacementHost) {
      constexpr int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
        MS_LOG(EXCEPTION) << "Skip zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data)
                          << ", bytes not aligned with expected.";
      }
      if (me_types[i] == TypeId::kObjectTypeString) {
        MS_LOG(EXCEPTION) << "It is not supported that Output node " << output_node->DebugString()
                          << "'s output data type is string now.";
      }
      MS_LOG(INFO) << "Zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data) << " as aligned with "
                   << kTensorAlignBytes << " types.";
      output_addr->set_is_ptr_persisted(false);
      output_addr->set_from_mem_pool(false);
      output_addr->set_deleter(deleter);
      output_addr->set_ptr(ge_data);
      output_addr->SetSize(tensor->GetSize());
    } else {
      MS_LOG(EXCEPTION) << "It is not supported that Output node " << output_node->DebugString()
                        << "'s output data's placement is device now.";
    }

    auto actual_shapes = tensor->GetTensorDesc().GetShape().GetDims();
    UpdateOutputNodeShape(output_node, idx, me_types[i], actual_shapes);
  }

  ConfigManager::GetInstance().ResetConfig();
  ConfigManager::GetInstance().ResetIterNum();

  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

FuncGraphPtr GeGraphExecutor::BuildDFGraph(const FuncGraphPtr &anf_graph,
                                           const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph.ir", anf_graph, true);
  }
#endif
  // if queue name is not empty, set datasink mode
  string queue_name = ConfigManager::GetInstance().dataset_param().queue_name();
  if (queue_name != "") {
    ConfigManager::GetInstance().set_dataset_mode(DatasetMode::DS_SINK_MODE);
  }
  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

  GeDeviceResManager::CreateSessionAndGraphRunner(IsGeTrain());
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(ERROR) << "Can not found GraphRunner";
    return nullptr;
  }

  return anf_graph;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
