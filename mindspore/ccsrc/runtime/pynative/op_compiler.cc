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

#include "runtime/pynative/op_compiler.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "include/backend/anf_runtime_algorithm.h"
#include "ops/nn_op_name.h"
#include "ops/conv_pool_op_name.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/device/device_address_utils.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#ifdef ENABLE_D
#include "transform/acl_ir/acl_adapter_info.h"
#endif

namespace mindspore {
using runtime::DeviceAddressUtils;
namespace pynative {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
mindspore::HashSet<std::string> kExcludedAttr = {"input_names", "output_names", "IsFeatureMapOutput",
                                                 "IsFeatureMapInputList", "pri_format"};
std::vector<std::string> kNumStrCache;

inline std::string GetNumString(int n) {
  if (n >= static_cast<int>(kNumStrCache.size())) {
    return std::to_string(n);
  }

  return kNumStrCache[n];
}

void UpdateRefInfoBeforeCreateKernel(const session::BackendOpRunInfoPtr &op_run_info, const KernelGraphPtr &graph) {
  // Building Graph and Create Kernel is async, under pynative mode.Ref info is bind with kernel.
  // So need to get ref info to generate output addr, before create kernel.
  if (op_run_info->base_op_run_info.device_target != kCPUDevice &&
      op_run_info->base_op_run_info.device_target != kGPUDevice) {
    // just ascend ref mode is diff with cpu and gpu
    return;
  }

  AnfAlgo::AddOutInRefToGraph(graph);
}

void CreateDeviceAddressWithoutWorkspace(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                         bool is_gradient_out) {
  DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateKernelOutputDeviceAddress(device_context, graph, is_gradient_out);
  DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(graph);
  DeviceAddressUtils::UpdateDeviceAddressForRefNode(graph);
}

device::DeviceAddressPtr GetGraphMapToCacheAddress(
  const std::map<KernelWithIndex, device::DeviceAddressPtr> &graph_map_to_cache,
  const KernelWithIndex &kernel_with_index) {
  auto iter = graph_map_to_cache.find(kernel_with_index);
  if (iter != graph_map_to_cache.end()) {
    return iter->second;
  }
  return nullptr;
}

void CacheForGraphInputs(const OpCompilerInfoPtr &op_compiler_info,
                         std::map<KernelWithIndex, device::DeviceAddressPtr> *graph_map_cache) {
  MS_EXCEPTION_IF_NULL(graph_map_cache);
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &inputs = graph->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto node_address = AnfAlgo::GetMutableOutputAddr(input, 0);
    MS_EXCEPTION_IF_NULL(node_address);
    auto kernel_with_index = std::make_pair(input, 0);
    auto cached_address = GetGraphMapToCacheAddress(*graph_map_cache, kernel_with_index);
    if (cached_address == nullptr) {
      cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
      (*graph_map_cache)[kernel_with_index] = cached_address;
    }
    (void)op_compiler_info->inputs_.emplace_back(cached_address);
  }
}

void CacheForGraphOutputs(const OpCompilerInfoPtr &op_compiler_info,
                          std::map<KernelWithIndex, device::DeviceAddressPtr> *graph_map_cache) {
  MS_EXCEPTION_IF_NULL(graph_map_cache);
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &output_nodes = op_compiler_info->graph_output_nodes_;
  for (auto &item_with_index : output_nodes) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (AnfAlgo::GetOutputTensorNum(item_with_index.first) == 0) {
      continue;
    }
    auto node_address = AnfAlgo::GetMutableOutputAddr(item_with_index.first, item_with_index.second, false);
    auto cached_address = GetGraphMapToCacheAddress(*graph_map_cache, item_with_index);
    if (cached_address == nullptr) {
      cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
      (*graph_map_cache)[item_with_index] = cached_address;
    }
    (void)op_compiler_info->outputs_.emplace_back(cached_address);
  }
}

void CacheForGraphValueNodes(const OpCompilerInfoPtr &op_compiler_info,
                             std::map<KernelWithIndex, device::DeviceAddressPtr> *graph_map_cache) {
  MS_EXCEPTION_IF_NULL(graph_map_cache);
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &value_nodes = graph->graph_value_nodes();
  for (auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (!AnfAlgo::OutputAddrExist(value_node, 0, false)) {
      continue;
    }
    auto node_address = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    (*graph_map_cache)[std::make_pair(value_node, 0)] = node_address;

    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>()) {
      auto tensor = node_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      op_compiler_info->value_map_to_tensor_[node_address] = tensor;
    } else {
      op_compiler_info->value_map_to_tensor_[node_address] = nullptr;
    }
  }
}

void CacheForGraphExecuteList(const OpCompilerInfoPtr &op_compiler_info,
                              std::map<KernelWithIndex, device::DeviceAddressPtr> *graph_map_cache) {
  MS_EXCEPTION_IF_NULL(graph_map_cache);
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  auto device_context = op_compiler_info->device_context_;
  const auto &nodes = graph->execution_order();
  for (auto const &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    ExecuteKernelInfo exe_kernel_info;
    exe_kernel_info.kernel_ = node;

    auto &inputs = node->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Invalid inputs.";
    }
    exe_kernel_info.primitive_ = common::AnfAlgo::GetCNodePrimitive(node);

    // Save inputs
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_num; ++i) {
      if (common::AnfAlgo::IsNoneInput(node, i)) {
        (void)exe_kernel_info.inputs_device_address_.emplace_back(nullptr);
        continue;
      }
      session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i, false);
      auto node_address = AnfAlgo::GetMutableOutputAddr(kernel_with_index.first, kernel_with_index.second, false);
      auto cached_address = GetGraphMapToCacheAddress(*graph_map_cache, kernel_with_index);
      if (cached_address == nullptr) {
        cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
        (*graph_map_cache)[kernel_with_index] = cached_address;
      }
      (void)exe_kernel_info.inputs_device_address_.emplace_back(cached_address);
    }

    // Save outputs
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      auto node_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
      auto kernel_with_index = std::make_pair(node, i);
      auto cached_address = GetGraphMapToCacheAddress(*graph_map_cache, kernel_with_index);
      if (cached_address == nullptr) {
        cached_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
        (*graph_map_cache)[kernel_with_index] = cached_address;
      }
      (void)exe_kernel_info.outputs_device_address_.emplace_back(cached_address);
    }

    (void)op_compiler_info->execute_kernel_list_.emplace_back(exe_kernel_info);
  }
}
}  // namespace

OpCompiler::OpCompiler() {
  session_ = session::SessionFactory::Get().Create(kSessionBasic);
  for (size_t i = 0; i < kNumberTypeEnd; i++) {
    (void)kNumStrCache.emplace_back(std::to_string(i));
  }
}

OpCompiler &OpCompiler::GetInstance() {
  static OpCompiler instance;
  return instance;
}

bool OpCompiler::IsInvalidInferResultOp(const std::string &op_name) const {
  static const std::unordered_set<std::string> kInvalidInferResultOp = {kDropoutOpName, kMaxPoolWithArgmaxOpName};
  return kInvalidInferResultOp.find(op_name) != kInvalidInferResultOp.end();
}

KernelGraphPtr OpCompiler::GenerateKernelGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                               const device::DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_run_info->op_prim);
  KernelGraphPtr graph;
  if (op_run_info->op_prim->name() == "PackFunc") {
    auto recent_graph = op_run_info->op_prim->GetAttr("recent_graph");
    MS_EXCEPTION_IF_NULL(recent_graph);
    auto func_graph = recent_graph->cast<FuncGraphPtr>();
    std::vector<KernelGraphPtr> all_out_graph;
    graph = session_->ConstructPackKernelGraph(func_graph, &all_out_graph, device_context->GetDeviceType());
    graph->set_attr(kAttrPackFunction, MakeValue(True));
  } else {
    graph = session_->ConstructSingleOpGraph(op_run_info, op_run_info->base_op_run_info.input_tensor,
                                             op_run_info->base_op_run_info.input_mask,
                                             device_context->GetDeviceType() == device::DeviceType::kAscend);
  }
  graph->set_is_from_single_op(true);
  return graph;
}

void OpCompiler::ConvertGraphToExecuteInfo(const OpCompilerInfoPtr &op_compiler_info) const {
  MS_LOG(DEBUG) << "ConvertGraphToExecuteInfo";
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  op_compiler_info->inputs_.clear();
  op_compiler_info->outputs_.clear();
  op_compiler_info->execute_kernel_list_.clear();

  std::map<KernelWithIndex, device::DeviceAddressPtr> graph_map_to_cache;

  // Save all value nodes
  CacheForGraphValueNodes(op_compiler_info, &graph_map_to_cache);

  // Save all inputs
  CacheForGraphInputs(op_compiler_info, &graph_map_to_cache);

  // Save all outputs
  CacheForGraphOutputs(op_compiler_info, &graph_map_to_cache);

  // Save all kernels
  CacheForGraphExecuteList(op_compiler_info, &graph_map_to_cache);
}

OpCompilerInfoPtr OpCompiler::Compile(const session::BackendOpRunInfoPtr &op_run_info, bool *single_op_cache_hit,
                                      const std::string &device_name, const uint32_t &device_id) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &graph_info = GetSingleOpGraphInfo(op_run_info->base_op_run_info, op_run_info->op_prim);
  const auto &iter = op_compiler_infos_.find(graph_info);
  // Check if the graph cache exists.
  auto &op_executor = runtime::OpExecutor::GetInstance();
  if (iter != op_compiler_infos_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (op_executor.BuildInQueue(iter->second->graph_id_)) {
      op_executor.Wait();
    }
    const auto &op_compiler_info = iter->second;
    MS_EXCEPTION_IF_NULL(op_compiler_info);
    *single_op_cache_hit = true;
    return iter->second;
  }

  MS_LOG(INFO) << "Run Op cache miss " << graph_info;
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeOpCompile,
                                     graph_info, true);

  *single_op_cache_hit = false;
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  py::gil_scoped_acquire acquire_gil;
  KernelGraphPtr graph = GenerateKernelGraph(op_run_info, device_context);
  MS_EXCEPTION_IF_NULL(graph);

  graph->set_run_mode(device::RunMode::kKernelMode);
  bool use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
  auto kernel_executor = device_context->GetKernelExecutor(use_dynamic_shape_process);
  MS_EXCEPTION_IF_NULL(kernel_executor);

  opt::OptimizationWithoutBackend(graph);
  // Unify the MindIR, must be before of the graph optimization.
  kernel_executor->AddMindIRPass(graph);

  // Select kernel and optimize
  kernel_executor->OptimizeGraph(graph);

  UpdateRefInfoBeforeCreateKernel(op_run_info, graph);

  // Create device address for all anf nodes of graph.
  CreateDeviceAddressWithoutWorkspace(graph, device_context, op_run_info->is_gradient_out);

  auto output_nodes = graph->outputs();
  std::vector<KernelWithIndex> outputs_with_index;
  std::vector<size_t> outputs_tensor_num;
  std::vector<std::string> outputs_padding_type;
  bool need_refresh_abstract = IsInvalidInferResultOp(op_run_info->base_op_run_info.op_name);
  for (auto &node : output_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    const auto &output_with_index = common::AnfAlgo::VisitKernel(node, 0);
    (void)outputs_with_index.emplace_back(output_with_index);
    (void)outputs_tensor_num.emplace_back(AnfAlgo::GetOutputTensorNum(output_with_index.first));
    const auto &padding_type = (device_context->GetDeviceType() == device::DeviceType::kAscend
                                  ? AnfAlgo::GetOutputReshapeType(output_with_index.first, output_with_index.second)
                                  : "");
    (void)outputs_padding_type.emplace_back(padding_type);

    MS_EXCEPTION_IF_NULL(output_with_index.first);
    const auto &abstract = output_with_index.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &shape = abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      need_refresh_abstract = true;
    }
  }
  AnfAlgo::UpdateGraphValidRefPair(graph);

  auto op_compiler_info = std::make_shared<OpCompilerInfo>(
    graph_info, graph->graph_id(), graph, device_context, op_run_info->base_op_run_info.need_earse_cache,
    need_refresh_abstract, outputs_with_index, outputs_tensor_num, outputs_padding_type);

  graph->set_graph_info(graph_info);
  ConvertGraphToExecuteInfo(op_compiler_info);
  op_compiler_infos_[graph_info] = op_compiler_info;
  return op_compiler_info;
}

void OpCompiler::BatchBuild(const std::vector<KernelGraphPtr> &graphs, const DeviceContext *device_context,
                            bool is_dynamic) const {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  // The compilation task may be in a child thread that has not yet set rt_context,
  // but the AICPU.so loading needs to use rt_context
  if (!device_context->device_res_manager_->BindDeviceToCurrentThread(true)) {
    MS_LOG(EXCEPTION) << "Bind device failed";
  }
  std::vector<CNodePtr> node_to_build;
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    const auto &nodes = graph->execution_order();
    (void)std::copy(nodes.begin(), nodes.end(), std::back_inserter(node_to_build));
  }
  // Kernel build
  auto kernel_executor = device_context->GetKernelExecutor(is_dynamic);
  MS_EXCEPTION_IF_NULL(kernel_executor);
  kernel_executor->CreateKernel(node_to_build);

  for (const auto &graph : graphs) {
    kernel_executor->PreprocessBeforeRun(graph);
    DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(device_context, graph);
    // Need to execute after PreprocessBeforeRunSingleOpGraph
    runtime::OpRuntimeInfo::CacheGraphOpRuntimeInfo(graph);
  }
}

#ifdef ENABLE_D
std::string GetGraphInfoForAscendSpecial(const pynative::BaseOpRunInfo &op_info, const PrimitivePtr &op_prim,
                                         const std::string &graph_info) {
  std::string ascend_special_info = graph_info;
  MS_EXCEPTION_IF_NULL(op_prim);
  auto op_name = op_prim->name();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice &&
      transform::AclAdapterManager::GetInstance().CheckAclAdapter(op_name)) {
    auto acl_info = transform::AclAdapterManager::GetInstance().GetOpInfo(op_name);
    if (!acl_info.input_selector().empty() || acl_info.output_selector() != nullptr) {
      if (op_info.input_tensor.size() == 0) {
        return ascend_special_info;
      }
      std::vector<ShapeVector> input_shapes;
      (void)std::transform(op_info.input_tensor.begin(), op_info.input_tensor.end(), std::back_inserter(input_shapes),
                           [](const auto &tensor) {
                             MS_EXCEPTION_IF_NULL(tensor);
                             return tensor->shape();
                           });

      auto in_func_map = acl_info.input_selector();
      for (auto [index, in_func] : in_func_map) {
        MS_EXCEPTION_IF_NULL(in_func);
        ascend_special_info += in_func(op_info.input_tensor[index]->data_type(), input_shapes);
      }

      auto out_func = acl_info.output_selector();
      if (out_func != nullptr) {
        auto out_format = out_func(op_info.input_tensor[0]->data_type(), input_shapes);
        ascend_special_info += out_format;
      }
    }
  }
  return ascend_special_info;
}
#endif

std::set<int64_t> GetInputDependValueList(const PrimitivePtr &op_prim) {
  std::set<int64_t> depend_list;
  auto op_infer_opt = abstract::GetPrimitiveInferImpl(op_prim);
  if (op_infer_opt.has_value()) {
    auto op_infer = op_infer_opt.value().Get();
    MS_EXCEPTION_IF_NULL(op_infer);
    if (op_infer != nullptr) {
      depend_list = op_infer->GetValueDependArgIndices();
    }
  }
  return depend_list;
}

std::string OpCompiler::GetSingleOpGraphInfo(const pynative::BaseOpRunInfo &op_info,
                                             const PrimitivePtr &op_prim) const {
  MS_EXCEPTION_IF_NULL(op_prim);
  if (op_info.input_tensor.size() != op_info.input_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << op_info.input_tensor.size()
                      << " should be equal to tensors mask size " << op_info.input_mask.size();
  }
  std::string graph_info = op_info.device_target;

  if (op_info.use_dynamic_shape_process) {
    graph_info += "_1_";
  } else {
    graph_info += "_0_";
  }
  auto op_name = op_prim->name();
  graph_info += op_name;
  bool has_hidden_side_effect;
  {
    PrimitiveReadLock read_lock(op_prim->shared_mutex());
    if (op_info.need_earse_cache) {
      return graph_info;
    }
    has_hidden_side_effect = op_prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_HIDDEN);
    // The value of the attribute affects the operator selection
    const auto &attr_map = op_prim->attrs();
    (void)std::for_each(attr_map.begin(), attr_map.end(), [&graph_info](const auto &element) {
      if (kExcludedAttr.find(element.first) != kExcludedAttr.end()) {
        return;
      }
      MS_EXCEPTION_IF_NULL(element.second);
      graph_info.append(element.second->ToString());
    });
  }
  for (size_t index = 0; index < op_info.input_tensor.size(); ++index) {
    const auto &input_tensor = op_info.input_tensor[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (op_info.use_dynamic_shape_process) {
      graph_info += GetNumString(static_cast<int>(input_tensor->shape().size()));
    } else {
      if (input_tensor->base_shape_ptr() != nullptr) {
        graph_info += input_tensor->base_shape_ptr()->ToString();
      } else {
        if (!input_tensor->shape().empty()) {
          const auto &shape_str =
            std::accumulate(std::next(input_tensor->shape().begin()), input_tensor->shape().end(),
                            std::to_string(input_tensor->shape()[0]),
                            [](std::string cur, size_t n) { return cur.append("-").append(std::to_string(n)); });
          graph_info += shape_str;
        }
      }
    }

    graph_info += GetNumString(input_tensor->data_type());
    // In the case of the same shape, but dtype and format are inconsistent
    auto tensor_addr = input_tensor->device_address();
    if (tensor_addr != nullptr && !has_hidden_side_effect) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_addr);
      MS_EXCEPTION_IF_NULL(p_address);
      graph_info += p_address->format();
      graph_info += p_address->padding_type();
    }
    // For constant input or op depend input value
    const auto &depend_list = GetInputDependValueList(op_prim);
    if (op_info.input_mask[index] == kValueNodeTensorMask ||
        (!depend_list.empty() && depend_list.find(index) != depend_list.end())) {
      graph_info += common::AnfAlgo::GetTensorValueString(input_tensor);
    }
    graph_info += "_";
  }

  // Operator with hidden side effect.
  if (has_hidden_side_effect) {
    (void)graph_info.append("r_").append(std::to_string(op_info.py_prim_id_)).append("_");
  }

#ifdef ENABLE_D
  // Ascend special info.
  graph_info = GetGraphInfoForAscendSpecial(op_info, op_prim, graph_info);
#endif

  return graph_info;
}

void OpCompiler::ClearOpCache(const GraphInfo &graph_info) { (void)op_compiler_infos_.erase(graph_info); }

void OpCompiler::ClearAllCache() { op_compiler_infos_.clear(); }
}  // namespace pynative
}  // namespace mindspore
