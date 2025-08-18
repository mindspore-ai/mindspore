/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/executor/ge_graph_executor.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <sstream>

#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/graph_ir/utils.h"
#include "tools/profiler/profiling.h"
#include "include/utils/utils.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "abstract/abstract_value.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "backend/ge_backend/executor/ge_memory_allocator.h"
#include "backend/ge_backend/executor/ge_utils.h"
#include "plugin/ascend/res_manager/ascend_res_manager.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "ge/ge_graph_compile_summary.h"
#include "op_proto/inc/array_ops.h"
#include "primitive/nn_op_name.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/utils/compile_cache_context.h"
#include "utils/singleton.h"
#include "ir/graph_utils.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_map.h"
#include "backend/ge_backend/pass/ge_backend_optimization.h"
#include "mindspore/core/include/ir/tensor_new.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};
using mindspore::session::KernelWithIndex;
static int64_t param_null_idx = 0;

void GetMeRetDataType(const AbstractBasePtr &cnode_data, std::vector<TypeId> *me_types) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<abstract::AbstractNone>()) {
    return;
  }

  if (cnode_data->isa<abstract::AbstractTensor>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    if (me_type == kObjectTypeTensorType) {
      me_type = dyn_cast<TensorType>(cnode_data->BuildType())->element()->type_id();
      (void)me_types->emplace_back(me_type);
    }
    return;
  }
  if (cnode_data->isa<abstract::AbstractScalar>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    (void)me_types->emplace_back(me_type);
    return;
  }
  auto abstract_tuple = cnode_data->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto elements = abstract_tuple->elements();
  for (size_t i = 0; i < abstract_tuple->size(); ++i) {
    GetMeRetDataType(elements[i], me_types);
  }
}

backend::ge_backend::TensorOrderMap GetDefaultParams(const FuncGraphPtr &anf_graph,
                                                     std::map<std::string, ShapeVector> *origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  backend::ge_backend::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    // if param name is empty, set name with "null_idx" to avoid duplicate names when process params in
    // DfGraphConvertor::params_
    if (para->name().empty()) {
      para->set_name("null_" + std::to_string(param_null_idx++));
    }
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      origin_shape->emplace(para->name(), tensor->shape_c());
      // need ref shape when auto parallel
      auto build_shape = para->abstract()->BuildShape();
      if (build_shape != nullptr) {
        (void)tensor->MetaTensor::set_shape(build_shape->cast<abstract::ShapePtr>()->shape());
        MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
      }
      res.emplace(para->name(), tensor);
      MS_LOG(DEBUG) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

void RevertOriginShape(const KernelGraphPtr &anf_graph, const std::map<std::string, ShapeVector> &origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  backend::ge_backend::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto it = origin_shape.find(para->name());
      if (it == origin_shape.end()) {
        MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << origin_shape;
        continue;
      }
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      (void)tensor->MetaTensor::set_shape(it->second);
      MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
    }
  }
}

void SetDynamicShapeAttr(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->output());
  for (auto &node : nodes) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(DEBUG) << "Set Dynamic Shape Attr to Node : " << node->fullname_with_scope();
      kernel_graph->SetGraphDynamicAttr(true);
      return;
    }
  }
}

bool BuildFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  if (!AddFakeGraph(anf_graph)) {
    MS_LOG(ERROR) << "Add fake graph failed";
    return false;
  }
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  return true;
}

void ClearForwardOutputAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_flag(kFlagPyNativeRunInGraph)) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    auto parameter = input->cast<ParameterPtr>();
    if (parameter != nullptr) {
      if (parameter->has_user_data(kForwardOutput)) {
        auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(parameter, 0, false);
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        auto device_address = kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(device_address);
        auto new_kernel_tensor = DeviceAddressUtils::CloneEmptyKernelTensor(kernel_tensor);
        AnfAlgo::SetOutputAddr(new_kernel_tensor->device_address(), new_kernel_tensor->tensor_storage_info(), 0,
                               parameter);
        MS_LOG(DEBUG) << "Clear old address " << device_address.get() << " and set new address "
                      << new_kernel_tensor->device_address().get() << " to parameter " << parameter->name();
      }
    }
  }
}

void UpdateFMTracker(size_t feature_memory_size, const std::string &graph_name) {
  device::tracker::CALL_MEMORY_TRACKER(AllocMemBlock, 0, feature_memory_size, "Ascend",
                                       device::ascend::AscendMemAdapter::GetInstance()->GetActualPeakMemory(), 0, 0,
                                       false, false);
  device::tracker::CALL_MEMORY_TRACKER(FreeMemBlock, 0, 0, 0);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "RunGeGraph", "RunGeGraph", graph_name, false);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "RunGeGraph", feature_memory_size, 0,
                                                 memory::mem_pool::MemType::kGeFixed);
}

bool CacheFileExists(const std::string &name) {
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto dep_files_hash = compile_cache_context.CompileCacheDepFilesHash();
  auto ge_graph_key = name;
  if (!dep_files_hash.empty()) {
    ge_graph_key = dep_files_hash + "_" + ge_graph_key;
  }
  auto ge_cache_path = compile_cache_context.GetCompilerCachePath() + kGeCache;
  ge_graph_key = NormalizeString(ge_graph_key);
  auto cache_idx_file = ge_cache_path + "/" + ge_graph_key + ".idx";
  struct stat buffer;
  bool ret = stat(cache_idx_file.c_str(), &buffer) == 0;
  MS_LOG(INFO) << "Cached index file name: " << cache_idx_file << " exists: " << ret;
  return ret;
}

void SetOutput(GeDeviceResManagerPtr res_manager, GeTensor *ge_output, const AnfNodePtr &output_node, size_t idx) {
  MS_EXCEPTION_IF_NULL(output_node);
  if (output_node->isa<ValueNode>() ||
      (output_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(output_node->cast<ParameterPtr>()))) {
    auto &&ge_data_uni = ge_output->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    deleter(ge_data);
    return;
  }
  auto actual_shapes = ge_output->GetTensorDesc().GetShape().GetDims();
  for (size_t i = 0; i < actual_shapes.size(); ++i) {
    if (actual_shapes[i] < 0) {
      MS_LOG(EXCEPTION) << "Output shape must be greater than 0, but got " << actual_shapes;
    }
  }
  auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx, false);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  auto output_addr = output_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(output_addr);
  output_addr->SetSize(ge_output->GetSize());
  auto &&ge_data_uni = ge_output->ResetData();
  auto ge_data = ge_data_uni.release();
  output_kernel_tensor->set_is_ptr_persisted(false);
  output_addr->set_from_mem_pool(false);
  output_addr->set_ptr(ge_data);
  auto placement = ge_output->GetTensorDesc().GetPlacement();
  if (placement == ::ge::kPlacementHost) {
    MS_LOG(DEBUG) << output_node->DebugString() << "'s output data's placement is host";
    size_t size = ge_output->GetSize();
    void *mem = res_manager->AllocateMemory(size);
    if (mem == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Allocate memory failed, memory size:" << size << ", output_node: " << output_node->ToString();
    }
    output_addr->set_from_mem_pool(true);
    output_addr->set_ptr(mem);
    auto *ascend_addr = output_addr.get();
    MS_EXCEPTION_IF_NULL(ascend_addr);
    ascend_addr->SetDeviceType(device::DeviceType::kAscend);
    auto tensor = tensor::from_spec(output_kernel_tensor->dtype_id(), output_kernel_tensor->GetShapeVector(),
                                    device::DeviceType::kCPU);
    MS_EXCEPTION_IF_NULL(tensor);
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    auto tmp_ptr = tensor->device_address()->GetMutablePtr();
    auto tmp_device_address = dynamic_cast<device::DeviceAddress *>(tensor->device_address().get());
    MS_EXCEPTION_IF_NULL(tmp_device_address);
    tmp_device_address->set_ptr(ge_data);
    tmp_device_address->SetSize(size);
    tensor->set_format(output_kernel_tensor->format());
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(ERROR) << "Failed to sync all stream.";
    }
    SyncCopy(output_kernel_tensor.get(), tensor.get(), ascend_addr->stream_id());
    tmp_device_address->set_ptr(tmp_ptr);
  }
  // Update shape in kernel tensor.
  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->SetShapeVector(actual_shapes);
  std::ostringstream zerocopy_log;
  zerocopy_log << "[ZeroCopy] Update output " << output_node->DebugString() << " address:" << output_addr->ToString()
               << ", shape:" << actual_shapes;
  MS_LOG(DEBUG) << zerocopy_log.str();
  MS_VLOG(VL_GE_EXECUTOR) << zerocopy_log.str();
}

void SetDynamicOutputs(const std::vector<KernelWithIndex> &graph_outputs, std::vector<GeTensor> *ge_outputs,
                       GeDeviceResManagerPtr res_manager) {
  MS_EXCEPTION_IF_NULL(res_manager);
  size_t ge_outputs_index = 0;
  size_t ge_outputs_size = ge_outputs->size();
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    if (ge_outputs_index >= ge_outputs_size) {
      MS_LOG(EXCEPTION) << "GE data access is out of bounds, which the current index value is " << ge_outputs_index
                        << ", the total number of GE output is " << ge_outputs_size << ".";
    }
    SetOutput(res_manager, &((*ge_outputs)[ge_outputs_index++]), output_node, idx);
  }
}

void SetDynamicOutputsForKernel(const std::vector<GeTensor> &ge_outputs,
                                const std::vector<kernel::KernelTensor *> &kernel_outputs, const AnfNodeWeakPtr &node,
                                GeDeviceResManagerPtr res_manager) {
  size_t index_ge = 0;
  for (size_t index_kernel = 0; index_kernel < kernel_outputs.size(); ++index_kernel) {
    auto kernel_output = kernel_outputs[index_kernel];
    if (kernel_output == nullptr) {
      continue;
    }

    if (common::AnfAlgo::IsMonadType(kernel_output->dtype_id())) {
      continue;
    }

    if (index_ge >= ge_outputs.size()) {
      MS_LOG(EXCEPTION) << "index " << index_ge << " is larger than ge_outputs size: " << ge_outputs.size();
    }
    auto ge_output = ge_outputs[index_ge];
    SetOutput(res_manager, &ge_output, node.lock(), index_kernel);
    ++index_ge;
  }
}

void UpdateTracker(const std::string &task_name, const std::string &node_name, const std::string &graph_str,
                   memory::mem_pool::MemType mem_type, device::DeviceAddress *device_tensor) {
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node_name, graph_str, false);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, task_name, mem_type, device_tensor->GetSize(),
                                                 device_tensor);
}

static inline std::string GetNodeInfo(const AnfNodeWeakPtr &node) {
  auto input_node = node.lock();
  MS_EXCEPTION_IF_NULL(input_node);
  return input_node->DebugString();
}
}  // namespace

void GeGraphExecutor::SetFlagIgnoreDevicePtr(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &param_list = graph->parameters();
  for (const auto &param_node : param_list) {
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_user_data(backend::ge_backend::kNoNeedAllocDeviceAddress)) {
      auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(param_node, 0, false);
      MS_EXCEPTION_IF_NULL(output_kernel_tensor);
      output_kernel_tensor->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
      MS_LOG(INFO) << "Node " << param_node->fullname_with_scope()
                   << " does not need device memory, so set kDeviceAddressFlagIgnoreDevicePtr.";
    }
  }
}

void GeGraphExecutor::InitGraphInfo(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  SetFlagIgnoreDevicePtr(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  BuildInputDataGeTensor(kg);
  BuildOutputDataGeTensor(kg);
}

void GeGraphExecutor::BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start BuildInputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  std::vector<GeTensor> ge_inputs;
  std::vector<kernel::KernelTensor *> kernel_tensors;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
  std::vector<AnfNodeWeakPtr> ge_input_nodes;
  auto ge_input_list = kernel_graph->user_data<backend::ge_backend::GEInputList>();
  if (ge_input_list) {
    ge_input_nodes = ge_input_list->ge_inputs;
  }
  for (const auto &node_wptr : ge_input_nodes) {
    auto node = node_wptr.lock();
    if (!node) {
      MS_LOG(ERROR) << "Get node lock failed, kerne graph: " << kernel_graph->ToString();
      continue;
    }
    auto name = node->fullname_with_scope();
    MS_LOG(INFO) << "Build input ge tensor: " << name << ", kernel graph: " << kernel_graph->graph_id();
    auto output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, 0, false);
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_addr = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_addr);
    (void)kernel_tensors.emplace_back(output_kernel_tensor.get());
    auto shapes = AnfAlgo::GetRuntimePaddingShape(node, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
    auto ge_tensor_desc = device::ascend::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    if (output_addr->GetMutablePtr() != nullptr) {
      MS_LOG(INFO) << "Node: " << name << " Has addr, size: " << output_addr->GetSize();
      if (ge_tensor.SetData(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                            [](void *) {}) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "SetData failed, ge input data " << ge_inputs.size() << " name: " << name
                          << " size: " << output_addr->GetSize();
      }
      MS_LOG(INFO) << "ge input data " << ge_inputs.size() << " name: " << name << " size: " << output_addr->GetSize();
    }
    // The device address of input tensor may change every step.
    // Always keep the input node address consistent with the input tensor address.
    (void)need_update_input.emplace_back(node, ge_inputs.size());
    (void)ge_inputs.emplace_back(std::move(ge_tensor));
  }
  input_datas_[kernel_graph.get()] = {ge_inputs, kernel_tensors, need_update_input};
  MS_LOG(INFO) << "BuildInputDataGeTensor finish.";
}

void GeGraphExecutor::BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start BuildOutputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  std::vector<GeTensor> ge_outputs;
  std::vector<kernel::KernelTensor *> kernel_tensors;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    auto real_index = output_node->isa<ValueNode>() ? 0 : index;
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, real_index, false);
    (void)kernel_tensors.emplace_back(kernel_tensor.get());
    auto shapes = AnfAlgo::GetRuntimePaddingShape(output_node, real_index);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
    auto ge_tensor_desc = device::ascend::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    (void)ge_outputs.emplace_back(std::move(ge_tensor));
    (void)graph_outputs.emplace_back(output_node, index);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    ge_outputs.size() == graph_outputs.size(),
    "The size of ge_outputs and graph_outputs check error, kernel graph: " + kernel_graph->ToString());
  output_datas_[kernel_graph.get()] = {ge_outputs, kernel_tensors, graph_outputs};
  MS_LOG(INFO) << "BuildOutputDataGeTensor finish.";
}

bool GeGraphExecutor::BuildGraph(const KernelGraphPtr &graph,
                                 const backend::ge_backend::TensorOrderMap &tensor_order_map) {
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  auto init_compile_cache = compile_cache_context.init_compile_cache();
  auto name = GetGraphName(graph);
  bool has_cache = CacheFileExists(name);
  if (use_compile_cache && init_compile_cache && has_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    if (!BuildFakeGraph(graph)) {
      return false;
    }
  } else {
    (void)BuildDFGraph(graph, tensor_order_map, false);
  }
  return true;
}

bool GeGraphExecutor::CompileGraph(const KernelGraphPtr &graph,
                                   const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "ge graph executor compile graph " << graph->ToString();
  std::map<std::string, ShapeVector> origin_shape;
  const auto &tensor_order_map = GetDefaultParams(graph, &origin_shape);
  (void)BuildGraph(graph, tensor_order_map);

  SetDynamicShapeAttr(graph);
  backend::ge_backend::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  // create loop var, must run before compile_graph, otherwise sink_size not take effort
  RunInitGraph(run_options.name);

  if (graph->is_dynamic_shape()) {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    auto ret = graph_runner->CompileGraph(run_options);
    if (ret != backend::ge_backend::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
    }
  } else {
    ::ge::CompiledGraphSummaryPtr ge_graph_summary = nullptr;
    {
      // Release GIL before calling into (potentially long-running) C++ code
      GilReleaseWithCheck gil_release;
      auto ret = graph_runner->CompileGraph(run_options, &ge_graph_summary);
      if (ret != backend::ge_backend::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
      }
    }
    GraphSummary summary(ge_graph_summary);
    MS_LOG(INFO) << "Graph " << run_options.name << " summary: " << summary.ToString();
    ge_message_manager_.SetSummary(run_options.name, summary);
    ge_message_manager_.SetFeatureMemory(run_options.name, summary.fixed_memory_size);
    ge_message_manager_.SetStream(run_options.name, summary.stream_num);
    AddRefCorrespondPairs(graph, summary.io_indexes);
    // if not static, set graph dynamic
    if (!summary.is_static) {
      graph->SetGraphDynamicAttr(true);
    }
  }
  GEMemoryAllocator::ProcessGraphDeviceAddress(graph, ge_res_manager_);

  graph->set_run_mode(device::RunMode::kGraphMode);
  graph->set_memory_managed_by_ge(true);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    graph->set_is_loop_count_sink(true);
  }
  RevertOriginShape(graph, origin_shape);
  return true;
}

std::vector<std::pair<uint32_t, uint32_t>> GeGraphExecutor::GetGraphRefIndexes(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &key = GetGraphName(graph);
  if (io_indexes_.count(key) != 0) {
    return io_indexes_.at(key);
  }
  return {};
}

void GeGraphExecutor::SetGraphWorkspaceMemory(const KernelGraphPtr &graph, void *device_ptr, size_t size) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  backend::ge_backend::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto ret = graph_runner->UpdateRefreshableMemory(run_options, device_ptr, size);
  if (ret != backend::ge_backend::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateRefreshableMemory for graph " << run_options.name << " failed.";
  }
}

size_t GeGraphExecutor::GetGraphWorkSpaceMemory(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  if (!ge_message_manager_.SummaryExist(graph_name)) {
    MS_LOG(INFO) << "The summary of graph: " << graph_name << " is not exist.";
    return 0;
  }
  auto summary = ge_message_manager_.GetSummary(graph_name);
  return summary.workspace_memory_size;
}

void GeGraphExecutor::AllocGERefreshableFeatureMemory(const KernelGraphPtr &graph) {
  MS_LOG(INFO) << "Start AllocGERefreshableFeatureMemory";
  size_t size = GetGraphWorkSpaceMemory(graph);
  if (size == 0) {
    return;
  }
  auto device_ptr = ge_res_manager_->AllocateMemory(size, kDefaultStreamIndex);
  SetGraphWorkspaceMemory(graph, device_ptr, size);
  graph_refreshable_feature_mem_[graph] = device_ptr;
}

void GeGraphExecutor::FreeGERefreshableFeatureMemory(const KernelGraphPtr &graph) {
  if (graph_refreshable_feature_mem_.find(graph) == graph_refreshable_feature_mem_.end() ||
      graph_refreshable_feature_mem_[graph] == nullptr) {
    return;
  }

  ge_res_manager_->FreeMemory(graph_refreshable_feature_mem_[graph]);
  graph_refreshable_feature_mem_[graph] = nullptr;
}

void GeGraphExecutor::AllocInputMemory(const device::DeviceAddressPtr &input_address) const {
  MS_EXCEPTION_IF_NULL(input_address);
  if (input_address->GetPtr() != nullptr) {
    MS_LOG(INFO) << "The device_address " << input_address << " memory has been alloc.";
    return;
  }

  if (!ge_res_manager_->AllocateMemory(input_address.get(), kDefaultStreamIndex)) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Device(id:" << device_id
                      << ") memory isn't enough and alloc failed, kernel name: Split tuple outputs, alloc size: "
                      << input_address->GetSize() << "B.";
  }
}

void GeGraphExecutor::AllocGEInputOutputMemory(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Start Alloc GE input and output Memory";

  // input
  std::vector<kernel::KernelTensor *> input_datas;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
  auto in_iter = input_datas_.find(graph.get());
  if (in_iter != input_datas_.end()) {
    input_datas = in_iter->second.kernel_tensors;
    need_update_input = in_iter->second.need_update_input;
  }

  for (size_t i = 0; i < input_datas.size(); ++i) {
    // for dynamic shape input , alloc later
    if (common::AnfAlgo::IsNodeOutputDynamicShape(need_update_input[i].first.lock())) {
      continue;
    }
    auto kernel_tensor = input_datas[i];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto input_address = kernel_tensor->device_address().get();
    if (input_address->GetPtr() == nullptr) {
      auto mem_type = memory::mem_pool::MemType::kKernel;
      if (need_update_input[i].first.lock()->isa<Parameter>() &&
          need_update_input[i].first.lock()->cast<ParameterPtr>()->has_default()) {
        mem_type = memory::mem_pool::MemType::kWeight;
      }
      UpdateTracker("AllocInputs", need_update_input[i].first.lock()->fullname_with_scope(), graph->ToString(),
                    mem_type, input_address);
      if (!ge_res_manager_->AllocateMemory(input_address, kDefaultStreamIndex)) {
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Device(id:" << device_id
                          << ") memory isn't enough and alloc failed, kernel name: Split tuple outputs, alloc size: "
                          << input_address->GetSize() << "B.";
      }
    }
  }

  // output
  std::vector<kernel::KernelTensor *> output_datas;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
  auto out_iter = output_datas_.find(graph.get());
  if (out_iter != output_datas_.end()) {
    output_datas = out_iter->second.kernel_tensors;
    graph_outputs = out_iter->second.graph_outputs;
  }

  // for io_index
  auto graph_name = GetGraphName(graph);
  if (ge_message_manager_.SummaryExist(graph_name)) {
    auto io_index = ge_message_manager_.GetSummary(graph_name).io_indexes;

    for (auto io : io_index) {
      auto input_kernel_tensor = input_datas[io.first];
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      auto output_kernel_tensor = output_datas[io.second];
      MS_EXCEPTION_IF_NULL(output_kernel_tensor);
      if (output_kernel_tensor->device_ptr() != nullptr &&
          output_kernel_tensor->device_ptr() != input_kernel_tensor->device_ptr()) {
        MS_LOG(INFO) << "The io_index[" << io.first << ", " << io.second << "] ptr is already exist and not same.";
        continue;
      }
      output_kernel_tensor->set_pointer_ref_count(input_kernel_tensor);
      output_kernel_tensor->set_is_ptr_persisted(input_kernel_tensor->is_ptr_persisted());
    }
  }

  for (size_t i = 0; i < output_datas.size(); ++i) {
    auto output_kernel_tensor = output_datas[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_address = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_address);
    if (common::AnfAlgo::IsDynamicShape(graph_outputs[i].first.lock())) {
      // dynamic shape output no need to alloc device memory
      continue;
    }
    if (output_address->GetPtr() == nullptr) {
      UpdateTracker("AllocOutputs", graph_outputs[i].first.lock()->fullname_with_scope(), graph->ToString(),
                    memory::mem_pool::MemType::kGraphOutput, output_address);
      if (!ge_res_manager_->AllocateMemory(output_address, kDefaultStreamIndex)) {
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Device(id:" << device_id
                          << ") memory isn't enough and alloc failed, kernel name: Split tuple outputs, alloc size: "
                          << output_address->GetSize() << "B.";
      }
    }
  }
}

void GeGraphExecutor::FreeInputOutputMemory(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Start Free GE input and output Memory";

  // input, free memory not weight
  std::vector<kernel::KernelTensor *> input_datas;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
  auto in_iter = input_datas_.find(graph.get());
  if (in_iter != input_datas_.end()) {
    input_datas = in_iter->second.kernel_tensors;
    need_update_input = in_iter->second.need_update_input;
  }

  for (size_t i = 0; i < input_datas.size(); ++i) {
    // is_ptr_persisted may reset in actors compile
    if (need_update_input[i].first.lock()->isa<Parameter>() &&
        need_update_input[i].first.lock()->cast<ParameterPtr>()->has_default()) {
      MS_LOG(DEBUG) << "The input[" << i << "] is a weight, skip free memory.";
      continue;
    }
    auto input_kernel_tensor = input_datas[i];
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    auto input_address = input_kernel_tensor->device_address().get();
    if (input_address->GetPtr() != nullptr && !input_kernel_tensor->is_ptr_persisted()) {
      MS_LOG(DEBUG) << "Free the memory of input[" << i << "], deivce_address: " << input_address
                    << ", ptr: " << input_address->GetPtr();
      ge_res_manager_->FreeMemory(input_address);
    }
  }

  // output, free memory not persist(same address as input weight)
  std::vector<kernel::KernelTensor *> output_datas;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
  auto out_iter = output_datas_.find(graph.get());
  if (out_iter != output_datas_.end()) {
    output_datas = out_iter->second.kernel_tensors;
    graph_outputs = out_iter->second.graph_outputs;
  }

  for (size_t i = 0; i < output_datas.size(); ++i) {
    if (graph_outputs[i].first.lock()->isa<Parameter>() &&
        graph_outputs[i].first.lock()->cast<ParameterPtr>()->has_default()) {
      MS_LOG(DEBUG) << "The output[" << i << "] is a weight, skip free memory.";
      continue;
    }
    auto output_kernel_tensor = output_datas[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_address = output_kernel_tensor->device_address().get();
    if (output_address->GetPtr() != nullptr && !output_kernel_tensor->is_ptr_persisted()) {
      MS_LOG(DEBUG) << "Free the memory of output[" << i << "], deivce_address: " << output_address
                    << ", ptr: " << output_address->GetPtr();
      ge_res_manager_->FreeMemory(output_address);
    }
  }
}

device::DeviceAddressPtr GeGraphExecutor::CreateDeviceAddress(const KernelTensorPtr &kernel_tensor,
                                                              bool is_need_alloc_mem) const {
  auto address = kernel_tensor->device_address();
  if (is_need_alloc_mem) {
    ge_res_manager_->AllocateMemory(address.get(), kDefaultStreamIndex);
  }
  return address;
}

void GeGraphExecutor::AllocGEFixMemory() const {
  MS_LOG(INFO) << "Start AllocGEFixMemory";
  auto alloc_func = [this](size_t size) { return ge_res_manager_->AllocateStaticMemory(size); };
  auto update_func = [](bool is_refreshable, const backend::ge_backend::RunOptions &options, const void *const memory,
                        size_t size) {
    MS_LOG(INFO) << "Update GE fixed memory, graph name: " << options.name << ", is_refreshable: " << is_refreshable
                 << ", size: " << size << ", memory: " << memory;
    if (size == 0) {
      MS_LOG(INFO) << "GE fixed memory size == 0, skip update GE fixed memory.";
      return Status::SUCCESS;
    }
    if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat)) {
      std::cout << "[MS_RUNTIME_PROF]"
                << "Update GE fixed memory, graph name: " << options.name << ", is_refreshable: " << is_refreshable
                << ", size: " << size << ", memory: " << memory << std::endl;
    }
    auto graph_runner = backend::ge_backend::GetGraphRunner();
    MS_EXCEPTION_IF_NULL(graph_runner);
    if (is_refreshable) {
      return graph_runner->SetFixedMemory(options, memory, size);
    }
    return graph_runner->UpdateFeatureMemory(options, memory, size);
  };
  GEMemoryManager::Instance().AllocGEMemory(alloc_func, update_func);
}

void GeGraphExecutor::InitGEFixMemory(const KernelGraphPtr &graph, size_t stream_id) const {
  backend::ge_backend::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  if (!ge_message_manager_.SummaryExist(run_options.name)) {
    MS_LOG(INFO) << "The summary of graph: " << run_options.name << " is not exist.";
    return;
  }
  auto summary = ge_message_manager_.GetSummary(run_options.name);
  GEMemoryAllocator::AllocGraphMemory(run_options, graph, summary, stream_id, ge_res_manager_);
}

bool GeGraphExecutor::CompileGraphForKernel(const KernelGraphPtr &graph) { return CompileGraph(graph, {}); }

void GeGraphExecutor::AddRefCorrespondPairs(const KernelGraphPtr &graph,
                                            const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Start convert io_indexes to ref_map, kernel graph: " << graph->ToString();

  std::map<session::AnfWithOutIndex, session::AnfWithOutIndex> ref_out_in_map = {};
  auto graph_inputs_all = graph->parameters();
  std::vector<AnfNodePtr> graph_inputs = {};
  std::map<uint32_t, uint32_t> input_index_ge_to_kg;
  uint32_t ge_index = 0;
  uint32_t kg_index = 0;
  for (auto &node : graph_inputs_all) {
    MS_EXCEPTION_IF_NULL(node);
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (HasAbstractMonad(node)) {
      MS_LOG(INFO) << "Input node: " << node->DebugString() << " is a monad parameter, skip.";
      ++kg_index;
      continue;
    }

    if (abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(INFO) << "Input node: " << node->DebugString() << " is a tuple/list parameter, skip.";
      continue;
    }

    graph_inputs.emplace_back(node);
    input_index_ge_to_kg[ge_index] = kg_index;
    ++kg_index;
    ++ge_index;
  }

  std::vector<common::KernelWithIndex> graph_outputs_all = {};
  common::AnfAlgo::GetRealInputs(graph->get_return(), &graph_outputs_all);
  std::vector<common::KernelWithIndex> graph_outputs = {};
  std::map<uint32_t, uint32_t> output_index_ge_to_kg;

  ge_index = 0;
  kg_index = 0;
  for (auto &node_with_index : graph_outputs_all) {
    if (common::AnfAlgo::IsNoOuputNode(node_with_index.first) || HasAbstractMonad(node_with_index.first)) {
      MS_LOG(INFO) << "Output node: " << node_with_index.first->fullname_with_scope()
                   << " is a no output node or monad node, skip.";
      ++kg_index;
      continue;
    }

    graph_outputs.emplace_back(node_with_index);
    output_index_ge_to_kg[ge_index] = kg_index;
    ++kg_index;
    ++ge_index;
  }

  std::set<uint32_t> inputs_index;
  std::vector<std::pair<uint32_t, uint32_t>> kg_io_indexes;
  for (auto in_out_index : io_indexes) {
    if (in_out_index.first >= graph_inputs.size() || in_out_index.second >= graph_outputs.size()) {
      MS_LOG(EXCEPTION) << "The io_indexes out of range, input index: " << in_out_index.first
                        << ", output index: " << in_out_index.second << ", graph input size: " << graph_inputs.size()
                        << ", graph output size: " << graph_outputs.size();
    }

    session::AnfWithOutIndex origin_node = std::make_pair(graph_inputs[in_out_index.first], 0);
    session::AnfWithOutIndex final_node = graph_outputs[in_out_index.second];
    if (origin_node.first == final_node.first) {
      if (inputs_index.find(in_out_index.first) == inputs_index.end()) {
        kg_io_indexes.emplace_back(
          std::make_pair(input_index_ge_to_kg[in_out_index.first], output_index_ge_to_kg[in_out_index.second]));
        inputs_index.insert(in_out_index.first);
      }
      MS_LOG(INFO) << "The origin node is same as final node, node: " << origin_node.first->fullname_with_scope();
      continue;
    }

    if (ref_out_in_map.count(final_node) != 0) {
      MS_LOG(INFO) << "The node is already in ref_out_in_map, node: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }
    // if input node is not abstract ref, set ref may cause memory reuse error
    auto abs = origin_node.first->abstract();
    if (!abs->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(INFO) << "The node is not abstract tensor: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }

    ref_out_in_map.emplace(final_node, origin_node);
    if (inputs_index.find(in_out_index.first) == inputs_index.end()) {
      kg_io_indexes.emplace_back(
        std::make_pair(input_index_ge_to_kg[in_out_index.first], output_index_ge_to_kg[in_out_index.second]));
      inputs_index.insert(in_out_index.first);
    }

    MS_LOG(INFO) << "Convert io_index [" << in_out_index.first << ", " << in_out_index.second
                 << "] to ref_out_in_map, final_node: " << final_node.first->fullname_with_scope()
                 << ", index:" << final_node.second << ", origin_node: " << origin_node.first->fullname_with_scope()
                 << ", index: " << origin_node.second;
  }

  io_indexes_[GetGraphName(graph)] = kg_io_indexes;
  graph->set_ref_out_in_map(ref_out_in_map);
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);

  auto graph_name = GetGraphName(graph);
  uint64_t start_time = profiler::GetClockSyscnt();

  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  is_init_graph_run_[graph] = false;

  auto ret = CompileGraph(kg, compile_options);

  (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, start_time,
                                  profiler::GetClockSyscnt(), 1);
  InitGEFixMemory(kg, kDefaultStreamIndex);
  (void)ge_res_manager_->BindDeviceToCurrentThread(true);
  return ret;
}

void GeGraphExecutor::OptimizeBeforeCompileGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  backend::ge_backend::opt::GEUnifyMindIR(graph);
  std::set<KernelGraphPtr> memo;
  backend::ge_backend::opt::OptimizeGEGraph(graph, &memo);
}

size_t GeGraphExecutor::GetGraphFeatureMemory(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(WARNING) << "Need Profile Memory, graph: " << graph_name
                  << ", stream: " << ge_message_manager_.GetStream(graph_name);
  if (disable_ge_kernel_) {
    auto max_static_memory_size = ge_res_manager_->GetMaxUsedMemorySize();
    auto feature_memory_size = ge_message_manager_.GetFeatureMemory(graph_name);
    auto total_memory_size = max_static_memory_size + feature_memory_size;
    device::ascend::AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(total_memory_size);
    UpdateFMTracker(feature_memory_size, graph_name);
    return feature_memory_size;
  }
  return 0;
}
int64_t GeGraphExecutor::CurGraphSinkSize(std::string graph_name) {
  int64_t sink_size = -1;
  auto result = graph_sink_size_.find(graph_name);
  if (result != graph_sink_size_.end()) {
    sink_size = result->second;
  } else {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE &&
        ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK)) {
      sink_size = ConfigManager::GetInstance().iter_num();
    }
    MS_LOG(INFO) << "Graph [" << graph_name << "] sink size is " << sink_size;
    graph_sink_size_.insert(std::pair(graph_name, sink_size));
  }
  return sink_size;
}

bool GeGraphExecutor::RunGraphRefModeInnner(const FuncGraphPtr &graph, const std::vector<GeTensor> &ge_inputs,
                                            std::vector<GeTensor> *ge_outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "GE run graph start in ref mode, graph: " << graph_name << ".";
  RunInitGraph(graph_name);
  (void)ge_res_manager_->BindDeviceToCurrentThread(false);

  // call ge rungraph
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  backend::ge_backend::RunOptions run_options;
  run_options.name = graph_name;
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  bool is_dynamic_shape = kg->is_dynamic_shape();
  if (memory::mem_pool::IsMemoryPoolRecycle() && !is_dynamic_shape) {
    auto max_static_memory_size = ge_res_manager_->GetMaxUsedMemorySize();
    auto feature_memory_size = ge_message_manager_.GetFeatureMemory(graph_name);
    if (feature_memory_size != 0) {
      size_t total_memory_size = max_static_memory_size + feature_memory_size;
      size_t max_hbm_memory_size =
        static_cast<size_t>(device::ascend::AscendMemAdapter::GetInstance()->GetMsUsedHbmSize());
      device::ascend::AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(total_memory_size);
      UpdateFMTracker(feature_memory_size, graph_name);
      if (common::IsNeedMemoryStatistic()) {
        MS_LOG(WARNING) << "Now Memory Status, graph: " << graph_name
                        << ", max_static_memory_size: " << max_static_memory_size
                        << ", feature_memory_size: " << feature_memory_size
                        << ", max_moc_memory_size: " << max_hbm_memory_size;
      }
      if (total_memory_size > max_hbm_memory_size) {
        MS_LOG(EXCEPTION) << "Memory pool not enough, graph: " << graph_name
                          << ", max_static_memory_size: " << max_static_memory_size
                          << ", feature_memory_size: " << feature_memory_size
                          << ", max_moc_memory_size: " << max_hbm_memory_size;
      }
    }
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    backend::ge_backend::Status ret =
      backend::ge_backend::RunGraphWithStreamAsync(graph_runner, run_options, stream, ge_inputs, ge_outputs);
    if (ret != backend::ge_backend::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }
  return true;
}

bool GeGraphExecutor::RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::TensorPtr> &inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name
               << " is dynamic:" << kg->is_dynamic_shape();
  std::vector<GeTensor> ge_inputs = GenerateInputGeTensor(kg);
  std::vector<GeTensor> ge_outputs = GenerateOutputGeTensor(kg);
  auto ret = RunGraphRefModeInnner(graph, ge_inputs, &ge_outputs, ge_res_manager_->GetStream());
  if (!ret) {
    return ret;
  }
  if (kg->is_dynamic_shape()) {
    auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    SetDynamicOutputs(graph_outputs, &ge_outputs, ge_res_manager_);
    auto sync_ret = ge_res_manager_->SyncStream(ge_res_manager_->GetStream());
    if (!sync_ret) {
      MS_LOG(EXCEPTION) << "Sync stream failed";
    }
  }
  ClearForwardOutputAddress(kg);
  return true;
}

bool GeGraphExecutor::RunGraphRefModeForKernel(const KernelGraphPtr &graph, const AnfNodeWeakPtr &node,
                                               const std::vector<kernel::KernelTensor *> &inputs,
                                               const std::vector<kernel::KernelTensor *> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(stream);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name;
  bool is_dynamic_shape = graph->is_dynamic_shape();
  std::vector<GeTensor> ge_inputs = CreateInputGeTensorList(inputs, graph);
  std::vector<GeTensor> ge_outputs = CreateOutputGeTensorList(outputs, graph);
  auto ret = RunGraphRefModeInnner(graph, ge_inputs, &ge_outputs, stream);
  if (!ret) {
    return ret;
  }
  if (is_dynamic_shape) {
    MS_LOG(INFO) << "Update outputs for graph: " << graph_name;
    SetDynamicOutputsForKernel(ge_outputs, outputs, node, ge_res_manager_);
  }
  return true;
}

std::vector<GeTensor> GeGraphExecutor::CreateInputGeTensorList(const std::vector<kernel::KernelTensor *> &tensors,
                                                               const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(graph.get());
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  bool is_dynamic_shape = graph->is_dynamic_shape();
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  size_t index = 0;
  for (const auto &tensor : tensors) {
    // input monad is nullptr
    if (tensor == nullptr) {
      continue;
    }
    // remove monad
    if (common::AnfAlgo::IsMonadType(tensor->dtype_id())) {
      continue;
    }

    if (index >= ge_inputs.size()) {
      MS_LOG(EXCEPTION) << "The index " << index << " >= ge_inputs size: " << ge_inputs.size();
    }

    if (is_dynamic_shape) {
      auto ge_tensor_desc =
        device::ascend::TransformUtil::GetGeTensorDesc(tensor->GetShapeVector(), tensor->dtype_id(), kOpFormat_DEFAULT);
      MS_EXCEPTION_IF_NULL(ge_tensor_desc);
      ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
      (void)ge_inputs[index].SetTensorDesc(*ge_tensor_desc);
    }

    if (tensor->device_ptr() == nullptr) {
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      GEMemoryAllocator::AllocUnuseInput(graph, tensor, ge_res_manager_);
    }

    if (tensor->device_ptr() != ge_inputs[index].GetData() || tensor->size() != ge_inputs[index].GetSize()) {
      if (ge_inputs[index].SetData(reinterpret_cast<uint8_t *>(tensor->device_ptr()), tensor->size(), [](void *) {}) !=
          ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Set ge tensor addr failed! addr size is " << tensor->size();
      }
    }

    MS_LOG(DEBUG) << "For graph: " << graph->ToString() << ", update input GeTensor[" << index
                  << "], shape: " << tensor->GetShapeVector() << ", tensor_size: " << tensor->size()
                  << ", dtype: " << tensor->dtype_id() << ", ptr: " << tensor->device_ptr()
                  << ", kernel tensor: " << tensor->ToString();
    ++index;
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::CreateOutputGeTensorList(const std::vector<kernel::KernelTensor *> &tensors,
                                                                const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(graph.get());
  if (iter == output_datas_.end()) {
    return ge_outputs;
  }
  bool is_dynamic_shape = graph->is_dynamic_shape();
  const auto &output_datas = iter->second.ge_outputs;
  ge_outputs = output_datas;

  size_t index = 0;
  for (const auto &tensor : tensors) {
    // input monad is nullptr
    if (tensor == nullptr) {
      continue;
    }
    // remove monad
    if (common::AnfAlgo::IsMonadType(tensor->dtype_id())) {
      continue;
    }
    if (index >= ge_outputs.size()) {
      MS_LOG(EXCEPTION) << "The index " << index << " >= ge_outputs size: " << ge_outputs.size();
    }

    if (is_dynamic_shape) {
      ge_outputs[index].SetData(nullptr, 0U, [](void *) {});
      continue;
    }

    if (tensor->device_ptr() != ge_outputs[index].GetData() || tensor->size() != ge_outputs[index].GetSize()) {
      if (ge_outputs[index].SetData(reinterpret_cast<uint8_t *>(tensor->device_ptr()), tensor->size(), [](void *) {}) !=
          ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Set ge tensor addr failed! addr size is " << tensor->size();
      }
    }
    MS_LOG(DEBUG) << "For graph: " << graph->ToString() << ", update output GeTensor[" << index
                  << "], shape: " << tensor->GetShapeVector() << ", tensor_size: " << tensor->size()
                  << ", dtype: " << tensor->dtype_id() << ", ptr: " << tensor->device_ptr()
                  << ", kernel tensor: " << tensor->ToString();
    ++index;
  }
  return ge_outputs;
}

void GeGraphExecutor::DoAsyncCkpt(const FuncGraphPtr &graph) {
  static std::string env = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC");
  if (env != "1") {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto need_async_ckpt = ms_context->get_param<bool>(MS_CTX_NEED_CKPT);
  if (!need_async_ckpt) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  if (kg == nullptr) {
    return;
  }
  auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
  auto save_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
  auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
  MS_LOG(DEBUG) << "cur_step:" << cur_step << ", save_steps: " << save_steps
                << ", last_triggered_step:" << last_triggered_step;
  if (cur_step >= (last_triggered_step + save_steps)) {
    if (SkipOrResetCopyAction()) {
      MS_LOG(INFO) << "Enable async d2h copy";
      SavePrevStepWeight(kg->GetRootWeights(), ge_res_manager_->GetCopyDataStream());
    }
    if (kg->has_attr(kIsRefGraph) && GetValue<bool>(kg->get_attr(kIsRefGraph)) && SkipOrResetSyncAction()) {
      MS_LOG(INFO) << "Ref graph sync once action";
      ge_res_manager_->SyncCopyStream();
    }
  }
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::TensorPtr> &inputs,
                               std::vector<tensor::TensorPtr> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  DoAsyncCkpt(graph);

  if (!RunGraphRefMode(graph, inputs)) {
    (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeRunGraph_" + graph_name, start_time,
                                    profiler::GetClockSyscnt(), 1);
    return false;
  }

  if (graph->has_flag(backend::ge_backend::kGraphFlagHasGetNext)) {
    MS_LOG(DEBUG) << "Reset ConfigManager, graph: " << graph_name;
    ConfigManager::GetInstance().ResetConfig();
    ConfigManager::GetInstance().ResetIterNum();
  }
  (void)profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

FuncGraphPtr GeGraphExecutor::BuildDFGraph(
  const FuncGraphPtr &anf_graph, const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_inputs_map,
  bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  return anf_graph;
}

string GeGraphExecutor::ExportDFGraph(const std::string &file_name, const FuncGraphPtr &anf_graph,
                                      bool is_save_to_file) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto graph_name = GetGraphName(anf_graph);
  return backend::ge_backend::ExportDFGraph(file_name, graph_name, is_save_to_file);
}

std::vector<GeTensor> GeGraphExecutor::GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(kernel_graph.get());
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  for (size_t i = 0; i < iter->second.kernel_tensors.size(); ++i) {
    auto output_kernel_tensor = iter->second.kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_addr = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_addr);
    if (is_dynamic_shape) {
      auto ge_tensor_desc = device::ascend::TransformUtil::GetGeTensorDesc(
        output_kernel_tensor->GetShapeVector(), output_kernel_tensor->dtype_id(),
        kernel::GetFormatFromEnumToStr(output_kernel_tensor->format()));
      MS_EXCEPTION_IF_NULL(ge_tensor_desc);
      ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
      (void)ge_inputs[i].SetTensorDesc(*ge_tensor_desc);
    }
    auto node_output_addr = output_addr->GetMutablePtr();
    if (node_output_addr == nullptr) {
      auto input_node = iter->second.need_update_input[i].first.lock();
      MS_EXCEPTION_IF_NULL(input_node);
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      GEMemoryAllocator::AllocUnuseInput(kernel_graph, input_node, output_addr, ge_res_manager_);
      node_output_addr = output_addr->GetMutablePtr();
    }
    std::ostringstream zerocopy_log;
    zerocopy_log << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update input "
                 << GetNodeInfo(iter->second.need_update_input[i].first) << " address to "
                 << output_kernel_tensor->ToString();
    MS_LOG(DEBUG) << zerocopy_log.str();
    MS_VLOG(VL_GE_EXECUTOR) << zerocopy_log.str();
    if (node_output_addr != ge_inputs[i].GetData() || output_addr->GetSize() != ge_inputs[i].GetSize()) {
      (void)ge_inputs[i].SetData(static_cast<uint8_t *>(node_output_addr), output_addr->GetSize(), [](void *) {});
    }
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(kernel_graph.get());
  if (iter == output_datas_.end()) {
    return ge_outputs;
  }
  const auto &output_datas = iter->second.ge_outputs;
  ge_outputs = output_datas;

  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  for (size_t idx = 0; idx < iter->second.kernel_tensors.size(); ++idx) {
    if (is_dynamic_shape) {
      ge_outputs[idx].SetData(nullptr, 0U, [](void *) {});
      continue;
    }
    auto output_node = iter->second.graph_outputs[idx].first.lock();
    auto index = iter->second.graph_outputs[idx].second;
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_CHECK_FAIL(
      idx < ge_outputs.size(),
      "GenerateOutputGeTensor idx is greater equal than ge_outputs size, idx: " + std::to_string(idx) +
        ", ge outputs size: " + std::to_string(ge_outputs.size()) + ", kernel graph: " + kernel_graph->ToString());
    auto output_kernel_tensor = iter->second.kernel_tensors[idx];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_device_addr = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_addr);
    auto node_output_device_addr = output_device_addr->GetMutablePtr();
    MS_LOG(INFO) << "Output addr " << node_output_device_addr;
    if (node_output_device_addr == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Output " << output_node->fullname_with_scope() << ", index: " << index
        << " address is nullptr, kernel graph: " << kernel_graph->ToString()
        << ", addr memory size: " << output_device_addr->GetSize()
        << "\n Maybe memory is not enough, memory statistics:"
        << device::ascend::AscendMemAdapter::GetInstance()->DevMemStatistics();
    }
    std::ostringstream zerocopy_log;
    zerocopy_log << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update output "
                 << output_node->DebugString() << " out_idx " << index << " address to "
                 << output_kernel_tensor->ToString();
    MS_LOG(DEBUG) << zerocopy_log.str();
    MS_VLOG(VL_GE_EXECUTOR) << zerocopy_log.str();
    if (node_output_device_addr != ge_outputs[idx].GetData() ||
        output_device_addr->GetSize() != ge_outputs[idx].GetSize()) {
      (void)ge_outputs[idx].SetData(reinterpret_cast<uint8_t *>(node_output_device_addr), output_device_addr->GetSize(),
                                    [](void *) {});
    }
  }
  return ge_outputs;
}

void GeGraphExecutor::RunInitGraph(const std::string &graph_name) {
  backend::ge_backend::RunOptions run_options;
  run_options.name = "init_subgraph." + graph_name;
  if (backend::ge_backend::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(INFO) << "Can not find " << run_options.name << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  auto cur_sink_size = CurGraphSinkSize(graph_name);
  if (pre_sink_size_ == cur_sink_size) {
    return;
  }
  pre_sink_size_ = cur_sink_size;
  MS_LOG(INFO) << "Start run init graph: " << run_options.name << ", sink size:" << cur_sink_size;
  std::vector<backend::ge_backend::GeTensorPtr> ge_outputs;
  std::vector<backend::ge_backend::GeTensorPtr> ge_tensors;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    backend::ge_backend::Status ret = backend::ge_backend::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != backend::ge_backend::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }
    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";
  }
}

void GeGraphExecutor::Initialize() {
  if (initialized_) {
    return;
  }

  ge_res_manager_ = std::make_shared<GeDeviceResManager>();
  ge_res_manager_->Initialize();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey host_key = {device::DeviceType::kAscend, device_id};
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  auto ascend_res_manager = static_cast<device::ascend::AscendResManager *>(device_context->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(ascend_res_manager);
  ascend_res_manager->InitializeForGe();

  CreateSessionAndGraphRunner();
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);

  ge_allocator_ = std::make_shared<GeAllocator>(ge_res_manager_.get());
  backend::ge_backend::Status ret =
    backend::ge_backend::RegisterExternalAllocator(graph_runner, ge_res_manager_->GetStream(), ge_allocator_);
  if (ret != backend::ge_backend::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "RegisterExternalAllocator failed";
  }
  MS_LOG(INFO) << "Create session and graphrunner successful.";

  ms_context->increase_param<uint32_t>(MS_CTX_GE_REF);
  MS_LOG(INFO) << "Init ge successful, ge reference = " << ms_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  initialized_ = true;
  return;
}

void GeGraphExecutor::CreateSessionAndGraphRunner() const {
  auto sess = backend::ge_backend::GetGeSession();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (sess == nullptr) {
    backend::ge_backend::SessionOptions options;
    GetGeSessionOptions(&options);
    SetPassthroughGeOptions("session", &options);
    sess = backend::ge_backend::NewSession(options);
    (void)ge_res_manager_->BindDeviceToCurrentThread(true);
    backend::ge_backend::SetGeSession(sess);
  }

  backend::ge_backend::GraphRunnerOptions options;
  options.sess_ptr = sess;
  auto graph_runner = backend::ge_backend::NewGraphRunner(options);
  backend::ge_backend::SetGraphRunner(graph_runner);
}

void GeGraphExecutor::Finalize() {
  if (!initialized_) {
    return;
  }
  // clear ge op adapter
  device::ascend::OpAdapterMap::get().clear();
  // remove ge graph
  backend::ge_backend::DfGraphManager::GetInstance().ClearGraph();
  // unregister allocator, before ResManager Destroy(use stream)
  UnregisterExternalAllocator();
  // Free GeTensorMemory before device_res_manager_ Destroy
  FreeGeTensorMemory();
  // set GeAllocator's resmanager as nullptr
  if (ge_allocator_ != nullptr) {
    ge_allocator_->ResetResManager();
  }
  // Device resource manager must be destroyed before 'FinalizeGe' unless some runtime APIs will throw exception.
  ge_res_manager_->Destroy();

  // clear graphrunner and session, and GEFinalize
  (void)FinalizeGe();
  initialized_ = false;
}

void GeGraphExecutor::UnregisterExternalAllocator() {
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(INFO) << "The graph_runner is not exist";
    return;
  }
  if (!graph_runner->IsAllocatorRegistered()) {
    return;
  }
  auto ret = backend::ge_backend::UnregisterExternalAllocator(graph_runner, ge_res_manager_->GetStream());
  if (ret != backend::ge_backend::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UnregisterExternalAllocator failed";
  }
}

bool GeGraphExecutor::FinalizeGe() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    return true;
  }
  ms_context->decrease_param<uint32_t>(MS_CTX_GE_REF);
  if (ms_context->get_param<uint32_t>(MS_CTX_GE_REF) == 0) {
    ms_context->set_param<uint32_t>(MS_CTX_GE_REF, 0);
    try {
      backend::ge_backend::ClearGeSessionAndRunner();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Error: " << e.what();
    } catch (...) {
      std::string exName(abi::__cxa_current_exception_type()->name());
      MS_LOG(ERROR) << "Error occurred when deleting GE graph runner and session fail. Exception name: " << exName;
    }
    if (::ge::GEFinalize() != ::ge::GRAPH_SUCCESS) {
      MS_LOG(WARNING) << "Finalize GE failed!";
    }
    ms_context->set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  } else {
    MS_LOG(INFO) << "Ge is used, no need to finalize, tsd reference = "
                 << ms_context->get_param<uint32_t>(MS_CTX_GE_REF) << ".";
  }
  return true;
}

std::unordered_set<std::string> GeGraphExecutor::GetInferParameterNames() {
  return Singleton<InferNeedUpdateParaNames>::Instance().GetInferParameterNames();
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
