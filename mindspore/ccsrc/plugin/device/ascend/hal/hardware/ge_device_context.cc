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

#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/scoped_long_running.h"
#include "abstract/abstract_value.h"
#include "backend/common/session/kernel_graph.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "profiler/device/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/optimizer/ge_optimization.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/config.h"
#include "runtime/dev.h"
#include "distributed/init.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
using mindspore::transform::OptionMap;

constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
constexpr char kGeDumpMode[3][7] = {"all", "input", "output"};
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};

std::string GetGraphName(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  std::string name;
  if (kg == nullptr) {
    name = graph->ToString();
  } else {
    FuncGraphPtr origin_graph = kg->GetFuncGraph();
    MS_EXCEPTION_IF_NULL(origin_graph);
    name = origin_graph->ToString();
  }
  return name;
}

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
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

std::string ShapesToString(const ShapeArray &shapes) {
  std::stringstream buffer;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      buffer << ",";
    }
    buffer << "[";
    const auto &shape = shapes[i];
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j != 0) {
        buffer << ",";
      }
      buffer << shape[j];
    }
    buffer << "]";
  }
  return buffer.str();
}

OptionMap GetComputeGraphOptions(const ShapeArray &input_shapes, bool is_dynamic_shape) {
  OptionMap options{};
  if (common::GetEnv("GE_TRAIN") == "1") {
    (void)options.emplace("ge.exec.variable_acc", "1");
  }
  if (!is_dynamic_shape) {
    return options;
  }
  (void)options.emplace("ge.exec.dynamicGraphExecuteMode", "dynamic_execute");
  (void)options.emplace("ge.exec.dataInputsShapeRange", ShapesToString(input_shapes));
  return options;
}

bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto converter = transform::NewConverter(anf_graph);
  if (export_air) {
    MS_LOG(INFO) << "Set DfGraphConvertor training : false";
    transform::SetTraining(converter, false);
  }
  transform::BuildGraph(anf_graph->ToString(), converter, init_inputs_map);
  transform::GenerateBroadcastGraph(converter, init_inputs_map);
  transform::GenerateCheckpointGraph(converter);
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }

  std::string graph_name = anf_graph->ToString();
  std::string init_graph = "init_subgraph." + graph_name;
  std::string checkpoint_name = "save." + GetGraphName(anf_graph);
  const auto options = GetComputeGraphOptions(converter->input_shapes(), converter->dynamic_shape_inputs());
  MS_LOG(INFO) << "Set options of compute graph: " << graph_name << " to " << MapToString(options);
  (void)transform::AddGraph(graph_name, transform::GetComputeGraph(converter), options);
  (void)transform::AddGraph(init_graph, transform::GetInitGraph(converter));
  (void)transform::AddGraph(BROADCAST_GRAPH_NAME, transform::GetBroadcastGraph(converter));

  transform::Status ret = transform::AddGraph(checkpoint_name, transform::GetSaveCheckpointGraph(converter));
  if (ret == transform::Status::SUCCESS) {
    transform::SetAnfGraph(checkpoint_name, anf_graph);
  }

  return true;
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
      init_input_map.emplace(para->name(), value->cast<std::shared_ptr<tensor::Tensor>>());
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
  size_t total_output_num = AnfAlgo::GetOutputTensorNum(node);
  if (index >= total_output_num) {
    MS_LOG(EXCEPTION) << "Invalid output index " << index << ", node " << node->fullname_with_scope() << " has "
                      << total_output_num << " outputs.";
  }
  std::vector<TypeId> types = {};
  std::vector<ShapeVector> shapes = {};
  for (size_t i = 0; i < total_output_num; ++i) {
    if (i == index && kIgnoreGEShapeOps.count(name) == 0) {
      // Using GE infer result to update shape
      // TODO(hbhu_bin): when kIgnoreGEShapeOps's size > 5, you should refactor this code
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

bool IsGeTrain() {
  auto env_ge = common::GetEnv("MS_ENABLE_GE");
  auto env_training = common::GetEnv("MS_GE_TRAIN");
  if (env_ge == "1" && env_training == "1") {
    return true;
  }
  return false;
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
  opt::GeOptimization(kg);
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
  auto graph_name = graph->ToString();
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
    output_addr->set_ptr(device_context_->device_res_manager_->AllocateMemory(tensor->GetSize()));
    output_addr->SetSize(tensor->GetSize());
    output_addr->set_is_ptr_persisted(false);

    if (output_addr->GetSize() < LongToSize(UlongToLong(tensor->GetSize()))) {
      MS_LOG(EXCEPTION) << "Output node " << output_node->DebugString() << "'s mem size " << output_addr->GetSize()
                        << " is less than actual output size " << tensor->GetSize();
    }
    // memcpy_s does not support data that more than 2GB
    (void)memcpy(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), tensor->GetData(), tensor->GetSize());
    auto actual_shapes = tensor->GetTensorDesc().GetShape().GetDims();
    UpdateOutputNodeShape(output_node, idx, me_types[i], actual_shapes);
  }
  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

bool GeDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const { return true; }

RunMode GeDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const { return RunMode::kGraphMode; }

void GeDeviceContext::Initialize() {
  if (initialized_) {
    return;
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to  heterogeneous mode
  int32_t is_heterogenous = 0;
  (void)rtGetIsHeterogenous(&is_heterogenous);
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, is_heterogenous == 1);
  InitGe(ms_context);
  initialized_ = true;
}

void GeDeviceContext::Destroy() {
  (void)FinalizeGe(MsContext::GetInstance());
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
}

void GeDeviceResManager::Initialize() {
  if (mem_manager_ == nullptr) {
    mem_manager_ = std::make_shared<cpu::CPUMemoryManager>();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
}

void GeDeviceResManager::Destroy() {
  // Release memory.
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
    mem_manager_ = nullptr;
  }
}

void *GeDeviceResManager::AllocateMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false);
}

void GeDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

std::vector<void *> GeDeviceResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list) const {
  return mem_manager_->MallocContinuousMemFromMemPool(size_list);
}

DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress(void *const device_ptr, size_t device_size,
                                                         const string &format, TypeId type_id, const ShapeVector &shape,
                                                         const UserDataPtr &user_data) const {
  auto device_address = std::make_shared<cpu::CPUDeviceAddress>(device_ptr, device_size, format, type_id,
                                                                device_context_->device_context_key_.device_name_,
                                                                device_context_->device_context_key_.device_id_);
  device_address->set_host_shape(shape);
  return device_address;
}

void GeDeviceContext::InitGe(const std::shared_ptr<MsContext> &inst_context) {
  MS_EXCEPTION_IF_NULL(inst_context);

  if (inst_context->get_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT)) {
    return;
  }

  if (static_cast<bool>(inst_context->get_param<uint32_t>(MS_CTX_GE_REF))) {
    inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
    return;
  }

  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  std::map<std::string, std::string> ge_options;
  GetGeOptions(inst_context, &ge_options);
  {
    // Release GIL before calling into (potentially long-running) C++ code
    mindspore::ScopedLongRunning long_running;
    if (::ge::GEInitialize(ge_options) != ::ge::GRAPH_SUCCESS) {
      MS_LOG(EXCEPTION) << "Initialize GE failed!";
    }
  }
  inst_context->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  return;
}

void GeDeviceContext::GetGeOptions(const std::shared_ptr<MsContext> &ms_context_ptr,
                                   std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(ms_context_ptr);
  MS_EXCEPTION_IF_NULL(ge_options);

  (*ge_options)["device_id"] = "0";
  // set up dump options
  auto dump_env = common::GetEnv(kMindsporeDumpConfig);
  if (!dump_env.empty()) {
    auto &dump_parser = DumpJsonParser::GetInstance();
    dump_parser.Parse();
    (*ge_options)["ge.exec.enableDump"] = std::to_string(static_cast<int>(dump_parser.async_dump_enabled()));
    (*ge_options)["ge.exec.dumpPath"] = dump_parser.path();
    // Parse() make sure that input_output is less than 3.
    (*ge_options)["ge.exec.dumpMode"] = kGeDumpMode[dump_parser.input_output()];
    // DumpStep is set to "all" by default
    if (dump_parser.iteration_string() != "all") {
      (*ge_options)["ge.exec.dumpStep"] = dump_parser.iteration_string();
    }
    MS_LOG(INFO) << "The enable dump state is " << (*ge_options)["ge.exec.enableDump"] << ", save dump path is "
                 << (*ge_options)["ge.exec.dumpPath"] << ", dump mode is " << kGeDumpMode[dump_parser.input_output()]
                 << ", dump step is " << dump_parser.iteration_string() << ".";
  }
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  (*ge_options)["ge.exec.profilingMode"] = std::to_string(static_cast<int>(profiler_manager->GetProfilingEnableFlag()));
  if (profiler_manager->GetProfilingEnableFlag()) {
    (*ge_options)["ge.exec.profilingOptions"] = profiler_manager->GetProfilingOptions();
  }

  (*ge_options)["rank_table_file"] = "";
  auto env_ddk_version = common::GetEnv("DDK_VERSION");
  if (!env_ddk_version.empty()) {
    (*ge_options)["ge.DDK_version"] = env_ddk_version;
  } else {
    (*ge_options)["ge.DDK_version"] = "1.60.T17.B830";
  }
  (*ge_options)["graphType"] = "1";

  if (ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE) != "0") {
    (*ge_options)["ge.graphMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE);
  }

  if (ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE) != "0") {
    (*ge_options)["ge.variableMemoryMaxSize"] = ms_context_ptr->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
  }

  bool training = IsGeTrain();
  if (training) {
    (*ge_options)["ge.graphRunMode"] = "1";
  }

  SetDisableReuseMemoryFlag(ge_options);
  SetHcclOptions(ms_context_ptr, ge_options);

  auto env_job_id = common::GetEnv("JOB_ID");
  if (!env_job_id.empty()) {
    (*ge_options)["ge.exec.jobId"] = env_job_id;
  } else {
    (*ge_options)["ge.exec.jobId"] = "0";
    MS_LOG(WARNING) << "JOB_ID is not set in ENV. Now set to default value 0";
  }

  auto env_fe_flag = common::GetEnv("FE_FLAG");
  if (!env_fe_flag.empty()) {
    (*ge_options)["ge.feFlag"] = env_fe_flag;
    MS_LOG(INFO) << "Use FE, make sure fe lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
  }

  auto env_aicpu_flag = common::GetEnv("AICPU_FLAG");
  if (!env_aicpu_flag.empty()) {
    (*ge_options)["ge.aicpuFlag"] = env_aicpu_flag;
    MS_LOG(INFO) << "Use AICPU, make sure aicpu lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
  }

  auto env_op_precision = common::GetEnv("MS_GE_OP_PRECISION");
  if (!env_op_precision.empty()) {
    (*ge_options)["ge.exec.op_precision_mode"] = env_op_precision;
    MS_LOG(INFO) << "Use MS_GE_OP_PRECISION, op precision mode path:" << env_op_precision;
  }

  auto proto_lib_path = common::GetEnv("OPTION_PROTO_LIB_PATH");
  if (!proto_lib_path.empty()) {
    char real_path[PATH_MAX] = {0};
    if (realpath(proto_lib_path.c_str(), real_path)) {
      proto_lib_path = real_path;
      (*ge_options)["ge.opsProtoLibPath"] = proto_lib_path;
    }
  } else {
    MS_LOG(WARNING) << "Set proto lib path failed!";
  }

  if (training) {
    (*ge_options)["ge.exec.precision_mode"] = "allow_mix_precision";
  } else {
    (*ge_options)["ge.exec.precision_mode"] = "force_fp16";
  }

  (*ge_options)["ge.enableSmallChannel"] = "1";

  // Disable the global variable acc, only enable it while adding training graph in pipeline
  (*ge_options)["ge.exec.variable_acc"] = "0";

  // ge heterogeneous mode
  if (ms_context_ptr->get_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS)) {
    (*ge_options)["ge.socVersion"] = "Ascend310P3";
  }
}

void GeDeviceContext::SetDisableReuseMemoryFlag(std::map<std::string, std::string> *ge_options) const {
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_disable_reuse_memory = common::GetEnv("DISABLE_REUSE_MEMORY");
  if (!env_disable_reuse_memory.empty()) {
    (*ge_options)["ge.exec.disableReuseMemory"] = env_disable_reuse_memory;
  } else {
    (*ge_options)["ge.exec.disableReuseMemory"] = "0";
    MS_LOG(WARNING) << "DISABLE_REUSE_MEMORY is not set in ENV. Now set to default value 0";
  }
}

void GeDeviceContext::SetHcclOptions(const std::shared_ptr<MsContext> &inst_context,
                                     std::map<std::string, std::string> *ge_options) {
  MS_EXCEPTION_IF_NULL(inst_context);
  MS_EXCEPTION_IF_NULL(ge_options);
  auto env_table_file = common::GetEnv("RANK_TABLE_FILE");
  auto env_rank_id = common::GetEnv("RANK_ID");
  auto env_device_id = std::to_string(inst_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  auto env_cluster_info = common::GetEnv("HELP_CLUSTER");
  if (!(env_table_file.empty() || env_rank_id.empty()) || !(env_cluster_info.empty() || env_rank_id.empty())) {
    MS_LOG(INFO) << "Initialize Ge for distribute parameter";
    if (!env_table_file.empty()) {
      MS_LOG(INFO) << "Use hccl, make sure hccl lib is set in OPTION_EXEC_EXTERN_PLUGIN_PATH.";
      (*ge_options)["ge.exec.rankTableFile"] = env_table_file;
    }
    auto env_hccl_flag = common::GetEnv("HCCL_FLAG");
    if (!env_hccl_flag.empty()) {
      (*ge_options)["ge.exec.hcclFlag"] = env_hccl_flag;
    }
    (*ge_options)["ge.exec.isUseHcom"] = "1";
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    (*ge_options)["ge.exec.rankId"] = env_rank_id;
    (*ge_options)["ge.exec.podName"] = env_rank_id;
    (*ge_options)["ge.graphRunMode"] = "1";
  } else {
    // device id is still needed for non-distribute case
    (*ge_options)["ge.exec.deviceId"] = env_device_id;
    MS_LOG(INFO) << "No hccl mode. "
                 << "If use hccl, make sure [RANK_TABLE_FILE,RANK_ID,DEVICE_ID,DEPLOY_MODE] all be set in ENV.";
  }

  auto env_deploy_mode = common::GetEnv("DEPLOY_MODE");
  if (!env_deploy_mode.empty()) {
    (*ge_options)["ge.exec.deployMode"] = env_deploy_mode;
  } else {
    (*ge_options)["ge.exec.deployMode"] = "0";
    MS_LOG(WARNING) << "DEPLOY_MODE is not set in ENV. Now set to default value 0";
  }
}

bool GeDeviceContext::FinalizeGe(const std::shared_ptr<MsContext> &inst_context) {
  MS_EXCEPTION_IF_NULL(inst_context);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    return true;
  }
  inst_context->decrease_param<uint32_t>(MS_CTX_GE_REF);
  if (inst_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    inst_context->set_param<uint32_t>(MS_CTX_GE_REF, 0);
    try {
      transform::ClearGeSessionAndRunner();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (::ge::GEFinalize() != ::ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    inst_context->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = "
                 << inst_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  }
  return true;
}

void GeDeviceResManager::CreateSessionAndGraphRunner(bool is_training) {
  std::shared_ptr<::ge::Session> sess = transform::GetGeSession();
  if (sess == nullptr) {
    transform::SessionOptions options;
    if (is_training) {
      options["ge.trainFlag"] = "1";
      options["ge.streamNum"] = "100";
      options["ge.enabledLocalFmkop"] = "1";
      options["ge.hcomParallel"] = "1";
    } else {
      options["ge.trainFlag"] = "0";
    }

    options["ge.enablePrintOpPass"] = "0";
    sess = transform::NewSession(options);
    transform::SetGeSession(sess);
  }

  transform::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = transform::NewGraphRunner(options);
  transform::SetGraphRunner(graph_runner);
}

FuncGraphPtr GeGraphExecutor::BuildDFGraph(const FuncGraphPtr &anf_graph,
                                           const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
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

DeprecatedInterface *GeDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>(this);
  }
  return deprecated_interface_.get();
}

constexpr auto kGeDevice = "GE";
MS_REGISTER_DEVICE(kGeDevice, GeDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
