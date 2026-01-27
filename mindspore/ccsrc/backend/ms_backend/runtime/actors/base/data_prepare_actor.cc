/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/base/data_prepare_actor.h"
#include <algorithm>
#include <set>

#include "include/utils/callback.h"
#include "ir/anf.h"
#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/loop_count_actor.h"
#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include "backend/ms_backend/runtime/actors/base/profiler_actor.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/common/device_address_utils.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "async/async.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "utils/llm_manager.h"
#include "utils/convert_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace runtime {
namespace {
constexpr size_t kNormalTensorNum = 1;
constexpr size_t kMapTensorNum = 3;
constexpr size_t kMapTensorKeyIndex = 0;
constexpr size_t kMapTensorValueIndex = 1;
constexpr size_t kMapTensorStatusIndex = 2;
constexpr size_t kPinMemThreshold = 1024 << 10;

bool IsDataTakenOverByMemOffload(const DeviceContext *device_context, const DeviceTensorPtr &device_tensor) {
  MS_EXCEPTION_IF_NULL(device_context);
  if (device_context->GetDeviceType() == device::DeviceType::kCPU || device_tensor->GetSize() == 0) {
    return false;
  }
  if (device_context->device_res_manager_ == nullptr ||
      device_context->device_res_manager_->swap_manager() == nullptr) {
    return false;
  }
  return false;
}

void UpdateTracker(const std::string &task_name, const AnfNodePtr &node, const std::string &graph_str,
                   memory::mem_pool::MemType mem_type, const KernelTensorPtr &kernel_tensor) {
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node->fullname_with_scope(), graph_str);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, task_name, mem_type, device_tensor->GetSize(),
                                                 device_tensor.get());
}

void SyncTensorData(const TensorPtr &host_tensor, const KernelTensorPtr &kernel_tensor, const AnfNodePtr &node,
                    const DeviceContext *device_context, OpContext<KernelTensor> *const context,
                    GraphExecutionStrategy strategy, bool skip_h2d = false) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(context);
  auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  const bool taken_over_by_swap_manager = IsDataTakenOverByMemOffload(device_context, device_tensor);
  if (taken_over_by_swap_manager) {
    MS_LOG(EXCEPTION) << "Offload parameter is not supported when optimize_input is false.";
  }
  bool need_alloc_memory = !taken_over_by_swap_manager && (device_tensor->GetPtr() == nullptr);
  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  auto mem_type =
    node->isa<ValueNode>() ? memory::mem_pool::MemType::kConstantValue : memory::mem_pool::MemType::kWeight;
  if (need_alloc_memory) {
    UpdateTracker("SyncTensorData", node, graph_str, mem_type, kernel_tensor);
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, *context, *device_context, node->fullname_with_scope(),
                                                  device_tensor->GetSize());
    }
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsOutput, "SyncTensorData", device::GetDeviceNameByType(device_tensor->GetDeviceType()),
      device_tensor->GetPtr(), kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(),
      kernel_tensor->tensor_storage_info());
    if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor.get());
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: SyncTensorData, device address class ptr: " << output_address
                      << ", node: " << node->fullname_with_scope() << ", graph: " << graph_str
                      << ", device address size: " << device_tensor->GetSize()
                      << ", device address addr: " << device_tensor->GetPtr();
    }
  }

  auto get_tensor_by_index = [&host_tensor](size_t index) {
    if (!host_tensor->isa<tensor::MapTensor>()) {
      return host_tensor;
    }
    const auto &map_tensor = host_tensor->cast<tensor::MapTensorPtr>();
    MS_EXCEPTION_IF_NULL(map_tensor);
    switch (index) {
      case kMapTensorKeyIndex:
        return map_tensor->key_tensor();
      case kMapTensorValueIndex:
        return map_tensor->value_tensor();
      case kMapTensorStatusIndex:
        return map_tensor->status_tensor();
      default:
        MS_LOG(EXCEPTION) << "Invalid index:" << index << " for map tensor:" << host_tensor->ToString();
    }
  };

  ShapeVector host_shape = {};
  // GetRuntimePaddingShape doesn't support the value tuple node.
  if (!node->isa<ValueNode>()) {
    host_shape = AnfAlgo::GetRuntimePaddingShape(node, 0);
  }
  auto get_tensor_num = (host_tensor->isa<tensor::MapTensor>() ? kMapTensorNum : kNormalTensorNum);
  for (size_t i = 0; i < get_tensor_num; ++i) {
    const auto &real_host_tensor = get_tensor_by_index(i);
    MS_EXCEPTION_IF_NULL(real_host_tensor);
    // Copy data from host tensor to device.
    auto host_tensor_size = LongToSize(real_host_tensor->DataNBytes());
    auto host_tensor_type = real_host_tensor->data_type();
    if (node->isa<ValueNode>()) {
      host_shape = real_host_tensor->shape();
    }
    if (!skip_h2d && (!device_context->device_res_manager_->SyncAllStreams() ||
                      !SyncCopy(kernel_tensor.get(), real_host_tensor.get(), kDefaultStreamIndex))) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                               ", host tensor size: " + std::to_string(host_tensor_size) +
                               ", host tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                               ", device tensor size: " + std::to_string(device_tensor->GetSize());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy, (*context), error_info);
    }
  }
}

void ValueTupleToValue(const ValuePtr &value, std::vector<ValuePtr> *const values) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(values);
  if (value->isa<ValueSequence>()) {
    auto value_tuple = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      MS_EXCEPTION_IF_NULL(element);

      if (element->isa<ValueSequence>()) {
        ValueTupleToValue(element, values);
      } else {
        (void)values->emplace_back(element);
      }
    }
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    MS_EXCEPTION_IF_NULL(csr_tensor->GetIndptr());
    MS_EXCEPTION_IF_NULL(csr_tensor->GetIndices());
    MS_EXCEPTION_IF_NULL(csr_tensor->GetValues());
    (void)values->emplace_back(csr_tensor->GetIndptr());
    (void)values->emplace_back(csr_tensor->GetIndices());
    (void)values->emplace_back(csr_tensor->GetValues());
    (void)std::transform(csr_tensor->shape().begin(), csr_tensor->shape().end(), std::back_inserter(*values),
                         [](int64_t n) { return std::make_shared<Int64Imm>(n); });
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    MS_EXCEPTION_IF_NULL(coo_tensor->GetIndices());
    MS_EXCEPTION_IF_NULL(coo_tensor->GetValues());
    (void)values->emplace_back(coo_tensor->GetIndices());
    (void)values->emplace_back(coo_tensor->GetValues());
    (void)std::transform(coo_tensor->shape().begin(), coo_tensor->shape().end(), std::back_inserter(*values),
                         [](int64_t n) { return std::make_shared<Int64Imm>(n); });
  } else {
    (void)values->emplace_back(value);
  }
}

// The device address of input ref node may be modified by input tensor, so need update the device address of ref node.
void UpdateDeviceAddressByRefInputNode(const std::vector<KernelGraphPtr> &graphs,
                                       const std::set<AnfNode *> &modified_input_nodes) {
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    // The DeviceAddress of the graph parameter has been updated for GE mode, but need update for kbk sub graph execute
    // mode.
    if ((graph->is_graph_run_mode() || graph->is_any_type_input()) && !EnableKbkSubGraphExecute()) {
      continue;
    }

    for (auto &iter : graph->GetRefMap()) {
      auto &output_pair = iter.first;
      auto &input_pair = iter.second;
      MS_EXCEPTION_IF_NULL(output_pair.first);
      MS_EXCEPTION_IF_NULL(input_pair.first);
      if (modified_input_nodes.count(input_pair.first.get()) == 0) {
        continue;
      }

      MS_LOG(INFO) << "Update the ptr of ref node: " << output_pair.first->fullname_with_scope()
                   << " by the modified ref input parameter: " << input_pair.first->fullname_with_scope();
      auto ref_node_output_addr = AnfAlgo::GetMutableOutputAddr(output_pair.first, output_pair.second, false);
      MS_EXCEPTION_IF_NULL(ref_node_output_addr);
      const auto &front_input_node = AnfAlgo::FetchFrontNodeByBackendNode(input_pair.first, *graph);
      auto input_kernel_tensor =
        DeviceTensorStore::GetInstance().Fetch(front_input_node.get(), ref_node_output_addr->GetDeviceType());
      // Maybe subgraphs share the same backend input parameter, so fetch device tensor store by front node of this
      // subgraph maybe nullptr and use the output addr of input parameter directly.
      if (input_kernel_tensor == nullptr) {
        input_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_pair.first, input_pair.second, false);
      }
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      const auto &input_addr = input_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(input_addr);
      MS_EXCEPTION_IF_CHECK_FAIL((ref_node_output_addr->GetDeviceType() == input_addr->GetDeviceType()),
                                 "The device type of ref node is not equal.");
    }
  }
}

void UpdateDataNodeDeviceAddressSize(const AnfNodePtr &input_node, const TensorPtr &input_tensor,
                                     const kernel::KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const device::DeviceAddressPtr &device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  }
  auto device_shape = trans::TransShapeToDevice(
    input_tensor->shape(), kernel::GetFormatFromEnumToStr(kernel_tensor->format()), input_node, 0, output_type_id);
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto device_address_size = type_size * SizeOf(device_shape);
  MS_LOG(INFO) << "Size of device_address is updated from " << device_address->GetSize() << " to "
               << device_address_size;
  device_address->SetSize(device_address_size);
}

bool TftGetUceFlag() {
  static auto callback = GET_COMMON_CALLBACK(TftGetUceFlag, bool);
  return callback != nullptr && callback();
}

bool TftIsRebootNode() {
  static auto callback = GET_COMMON_CALLBACK(TftIsRebootNode, bool);
  return callback != nullptr && callback();
}

bool TftIsArf() {
  static auto callback = GET_COMMON_CALLBACK(TftIsArf, bool);
  return callback != nullptr && callback();
}

void TftClearErrorType() {
  static auto callback = GET_COMMON_CALLBACK(TftClearErrorType, void);
  if (callback != nullptr) {
    callback();
  }
}

const ValuePtr &TftGetConstant(const AnfNodePtr &node) {
  static auto callback = GET_COMMON_CALLBACK(TftGetConstant, const ValuePtr &, const AnfNodePtr &);
  MS_EXCEPTION_IF_NULL(callback);
  return callback(node);
}
}  // namespace

std::atomic<size_t> DataPrepareActor::execution_count_ = 0;
mindspore::HashMap<const DataPrepareActor *, mindspore::HashSet<const tensor::Tensor *>>
  DataPrepareActor::tensors_need_reprepare_ = {};
mindspore::HashMap<const tensor::Tensor *, mindspore::HashSet<const DataPrepareActor *>>
  DataPrepareActor::tensor_with_graphs_ = {};

void DataPrepareActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  strategy_ = graph_compiler_info_->strategy_;
  if (graph_compiler_info_->graphs_.size() != graph_compiler_info_->device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  size_t host_data_size = 0;
  if (host_data_source_actor_ != nullptr) {
    host_data_size = host_data_source_actor_->data_nodes().size();
  }
  has_parameter_input_ = graph_compiler_info_->inputs_num_ > host_data_size;
  MS_LOG(INFO) << graph_compiler_info_->name_
               << " has the parameter input num: " << graph_compiler_info_->inputs_num_ - host_data_size;

  for (const auto &graph : graph_compiler_info_->graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_dynamic_shape()) {
      has_dynamic_shape_ = true;
      break;
    }
  }
  tensors_need_reprepare_[this] = {};

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  is_enable_infer_boost_ = ms_context->IsEnableInferBoost();
  enable_trace_memory_ = EnableTraceMemory();
}

void DataPrepareActor::UpdateDynamicShapeAndSize(const AnfNodePtr &input_node, const TensorPtr &input_tensor) const {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_tensor == nullptr || IsEmptySequenceTensor(input_tensor.get())) {
    return;
  }
  if (!input_node->isa<Parameter>()) {
    return;
  }
  auto input_param = input_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(input_param);
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (!input_param->has_dynamic_shape() && !IsDynamic(kernel_tensor->GetShapeVector())) {
    return;
  }

  // Update shape.
  MS_LOG(DEBUG) << "Update dynamic shape for parameter:" << input_param->DebugString();
  const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    output_kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    return;
  }
  output_kernel_tensor->SetShape(input_tensor->base_shape_ptr());

  // Update size.
  auto device_format = kernel_tensor->format();
  static const std::set<Format> kNormalFormat = {
    Format::DEFAULT_FORMAT, Format::ND, Format::NCHW, Format::NHWC, Format::HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->DataNBytes();
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set device address:" << device_address << " size from:" << device_address->GetSize()
      << " to:" << tensor_data_size;
    device_address->SetSize(tensor_data_size);
  } else {
    MS_LOG(DEBUG) << "Update data node device address size";
    // Size of 5D format device_address is larger than tensor_data_size.
    UpdateDataNodeDeviceAddressSize(input_node, input_tensor, kernel_tensor);
  }
}

void DataPrepareActor::UpdateDeviceAddressForDataNode(const AnfNodePtr &input_node, const TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(input_node);
  // Assign tensor address to input data node and set `ref_count` to `SIZE_MAX` for avoiding clean.
  (void)address_modified_input_nodes_.insert(input_node.get());
  auto tensor_address = input_tensor->device_address();
  if (tensor_address == nullptr) {
    return;
  }

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->set_new_ref_count(SIZE_MAX);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (tensor_address == device_address) {
    tensor_address->SetNodeIndex(input_node, 0);
    return;
  }

  // If tensor address and device address are different (heterogeneous scenarios), or device address is persisted
  // Update device address data in data source actor process.
  if (kernel_tensor->is_ptr_persisted() || (tensor_address->GetDeviceType() != device_address->GetDeviceType()) ||
      (!AnfAlgo::IsEquivalentFormat(input_tensor->format(), kernel_tensor->format())) ||
      (input_tensor->data_type() != kernel_tensor->dtype_id())) {
    MS_LOG(DEBUG) << "Cannot update address of " << input_node->DebugString();
    return;
  }

  AnfAlgo::SetOutputAddr(tensor_address, input_tensor->storage_info(), 0, input_node);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Update device address of " << input_node->DebugString() << " to "
                                               << tensor_address.get() << " ptr:" << tensor_address->GetPtr();
  tensor_address->SetNodeIndex(input_node, 0);
  auto ref_iter = ref_kernel_tensors_.find({input_node, 0});
  if (ref_iter == ref_kernel_tensors_.end()) {
    return;
  }
  for (const auto &ref_kernel_tensor : ref_iter->second) {
    if (ref_kernel_tensor == nullptr) {
      continue;
    }
    ref_kernel_tensor->set_pointer_ref_count(kernel_tensor.get());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set pointer ref count:" << tensor_address->device_pointer() << " from node:" << input_node->DebugString()
      << " kernel tensor:" << kernel_tensor->ToString() << " to kernel tensor:" << ref_kernel_tensor->ToString();
  }
}

void DataPrepareActor::RecordGraphInputsForInputOptimize(const VectorRef &args) {
  MSTX_RANGE_GUARD("RecordGraphInputsForInputOptimize", nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, "RecordGraphInputsForInputOptimize",
                            true);
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  if (!EnableKbkSubGraphExecute()) {
    return;
  }
  if (inference_input_indexes_.empty()) {
    size_t non_weight_parameter_num = graph_parameter_store->GetNonWeightParameterNum();
    size_t current_data_num = 0;
    for (size_t i = 0; i < graph_compiler_info_->origin_parameters_order_.size(); ++i) {
      if (current_data_num == non_weight_parameter_num) {
        break;
      }
      // The input data is front of the parameter weight.
      if (graph_parameter_store->GetPositionWeight(i)) {
        MS_LOG(DEBUG) << "Skip the prepare host data for parameter: "
                      << graph_compiler_info_->origin_parameters_order_[i]->fullname_with_scope();
        continue;
      }
      const auto &origin_parameter = graph_compiler_info_->origin_parameters_order_[i];
      MS_EXCEPTION_IF_NULL(origin_parameter);
      if (i >= args.size()) {
        MS_LOG(DEBUG) << "Arg index out of args range, index is " << i << " and args size is " << args.size();
        continue;
      }
      inference_input_indexes_.emplace_back(i);
      inference_parameters_.emplace_back(origin_parameter);
      ++current_data_num;
    }
  }
  for (size_t i = 0; i < inference_input_indexes_.size(); ++i) {
    auto index = inference_input_indexes_[i];
    std::vector<tensor::TensorPtr> flatten_tensors;
    AnfAlgo::FlattenInputArg(args[index], inference_parameters_[i], &flatten_tensors);
    // Push flatten tensors into store buffers.
    graph_parameter_store->FillBuffer(index, flatten_tensors);
  }
  // Convert all non-contiguous and non-tuple inputs in the current network into contiguous ones.
  graph_parameter_store->ConvertNormalInputContiguous(inference_input_indexes_);
  auto isDyn = graph_parameter_store->RecordGraphInputsAndIsDyn(graph_compiler_info_, inference_input_indexes_,
                                                                inference_parameters_);
  if (GraphCaptureManager ::GetInstance().GetEnableGraphCapture()) {
    GraphCaptureManager::GetInstance().SetShapeKey();
  }
  MS_LOG(INFO) << "Prepare for actor set: " << graph_compiler_info_->name_ << ", convert static shape: " << (!isDyn)
               << ", dynamic shape: " << has_dynamic_shape_ << ", enable trace memory: " << enable_trace_memory_
               << ", enable parallel dispatch: " << EnableParallelDispatchKernel()
               << ", graph phase: " << graph_compiler_info_->graph_phase_;
  if (has_dynamic_shape_) {
    const auto &phase = graph_compiler_info_->graph_phase_;
    bool is_increment_graph = (phase.find("increment") != std::string::npos);
    if (GraphCaptureManager ::GetInstance().GetEnableGraphCapture()) {
      GraphCaptureManager::GetInstance().SetIncrementGraph(is_increment_graph);
    }
    if (is_increment_graph) {
      ActorDispatcher::set_enable_static_shape(!isDyn);
      if (enable_trace_memory_) {
        if (has_continuous_memory()) {
          MS_LOG(EXCEPTION)
            << "Can not support continuous memory allocate in dynamic shape graph when enable trace memory.";
        }
        if (!ActorDispatcher::enable_static_shape()) {
          ActorDispatcher::set_enable_trace_dynamic_memory(true);
        } else {
          ActorDispatcher::set_enable_use_trace_memory(true);
          ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_actor_set(EnableParallelDispatchKernel());
          if (ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set()) {
            MS_LOG(INFO) << "Enable parallel dispatch kernel for current actor set: " << graph_compiler_info_->name_
                         << ", graph phase: " << graph_compiler_info_->graph_phase_;
          }
        }
      }
    }
  }

  if (EnableRuntimeNewPipeline() && ActorDispatcher::enable_static_shape() &&
      ActorDispatcher::enable_runtime_multi_pipeline()) {
    // Not need infer and resize queue work in spin state when convert dynamic shape to static shape.
    RuntimePipeline::GetInstance().infer_queue()->Pause();
    RuntimePipeline::GetInstance().resize_queue()->Pause();
    ActorDispatcher::set_enable_runtime_multi_pipeline(false);
  }
}

void DataPrepareActor::PrepareDataBeforeInputOptimize(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                      const VectorRef &args, OpContext<KernelTensor> *const context,
                                                      uint64_t start_time) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  graph_parameter_store->ResetPrepareState();
  // Only prepare weight for first step, because the weight memory is not released.
  // Allocate memory for weights in actor would have memory fragmentation.
  if (first_step_ || TftGetUceFlag()) {
    PrepareDataForDeviceTensorStore(input_tensors, args, context);
  }
  first_step_ = false;

  if (is_enable_infer_boost_) {
    RecordGraphInputsForInputOptimize(args);
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendDebugReq(context);
    return;
  }

  if (profiler_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendProfilerReq(context);
    return;
  }

  PROFILER_END(start_time, runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kPreLaunch, GetAID().Name(),
               false);
  PostRun(context);
}

void DataPrepareActor::PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
                                   OpContext<KernelTensor> *const context, GraphExecutionStrategy real_strategy) {
  MS_EXCEPTION_IF_NULL(context);
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  try {
    // Preprocess before prepare data for data prepare actor.
    PreprocessBeforePrepareData();
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
  }

  MS_LOG(DEBUG) << "Data prepare actor(" << GetAID().Name() << ") prepares data.";
  if (enable_input_optimize_) {
    PrepareDataBeforeInputOptimize(input_tensors, args, context, start_time);
    return;
  }

  real_strategy_ = real_strategy;
  try {
    bool not_empty_input = !input_tensors.empty() || !args.empty();
    if (first_step_ || TftGetUceFlag() || (enable_prepare_case() && not_empty_input)) {
      PrepareDataForDeviceTensorStore(input_tensors, args, context);
    }
    PrepareDataForHostTensorQueue(input_tensors, args, context);
    tensors_need_reprepare_[this].clear();
  } catch (const std::exception &e) {
    std::string error_info = e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
  }

  first_step_ = false;
  if (IsRunningFailed(context)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  if (!address_modified_input_nodes_.empty()) {
    UpdateDeviceAddressByRefInputNode(graph_compiler_info_->graphs_, address_modified_input_nodes_);
    address_modified_input_nodes_.clear();
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendDebugReq(context);
    return;
  }

  if (profiler_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendProfilerReq(context);
    return;
  }

  PROFILER_END(start_time, runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kPreLaunch, GetAID().Name(),
               false);

  PostRun(context);
}

void DataPrepareActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugOnStepBegin, graph_compiler_info_->graphs_,
                            graph_compiler_info_->origin_parameters_order_, graph_compiler_info_->device_contexts_,
                            context, &GetAID());
  OnDebugFinish(context);
}

void DataPrepareActor::SendProfilerReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  ActorDispatcher::SendSync(*profiler_aid_, &ProfilerActor::ProfilerOnStepBegin, graph_compiler_info_->graphs_,
                            graph_compiler_info_->origin_parameters_order_, graph_compiler_info_->device_contexts_,
                            context, &GetAID());
  OnDebugFinish(context);
}

void DataPrepareActor::OnDebugFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  PostRun(context);
}

void DataPrepareActor::RecordTensorsNeedReprepare(tensor::Tensor *tensor) {
  tensor_with_graphs_[tensor].insert(this);
  if (tensor != nullptr && tensor->is_parameter()) {
    auto callback = [](const tensor::Tensor *tensor) {
      const auto &graph_iter = tensor_with_graphs_.find(tensor);
      if (graph_iter != tensor_with_graphs_.end()) {
        const auto &data_prepare_actors = graph_iter->second;
        for (const auto &data_prepare_actor : data_prepare_actors) {
          const auto &iter = tensors_need_reprepare_.find(data_prepare_actor);
          if (iter != tensors_need_reprepare_.end()) {
            tensors_need_reprepare_[data_prepare_actor].insert(tensor);
          }
        }
      }
    };
    tensor->set_update_value_callback(callback);
  }

  if (tensor != nullptr && !tensors_need_reprepare_[this].empty() && tensor->is_parameter()) {
    auto erased_num = tensors_need_reprepare_[this].erase(tensor);
    MS_LOG(DEBUG) << "Erase " << erased_num << " tensor which is reprepared.";
  }
}

TensorPtr DataPrepareActor::FetchInputTensor(const std::vector<TensorPtr> &tensors, size_t tensor_index,
                                             const VectorRef &args, const KernelWithIndex &front_node) {
  if (!tensors.empty()) {
    MS_EXCEPTION_IF_CHECK_FAIL((tensor_index < tensors.size()), "The tensor index is out of range.");
    return tensors[tensor_index];
  }

  MS_EXCEPTION_IF_NULL(front_node.first);
  const auto &iter = std::find(graph_compiler_info_->origin_parameters_order_.begin(),
                               graph_compiler_info_->origin_parameters_order_.end(), front_node.first);
  if (iter == graph_compiler_info_->origin_parameters_order_.end()) {
    MS_LOG(INFO) << "Not origin parameter:  " << front_node.first->fullname_with_scope();
    return nullptr;
  }
  auto arg_index = iter - graph_compiler_info_->origin_parameters_order_.begin();
  auto tensor = FetchInputTensorByArg(args, arg_index, front_node);
  // The tensor needs to be updated if modified.
  RecordTensorsNeedReprepare(tensor.get());

  // The tensor needs to be converted to contiguous before being given to the actors.
  // After the view feature is supported in the graph mode, the following code will be deleted.
  DeviceAddressUtils::ConvertContiguousTensorSync(tensor);
  tensor_with_graphs_[tensor.get()].insert(this);
  if (tensor != nullptr && tensor->is_parameter()) {
    auto callback = [](const tensor::Tensor *tensor) {
      const auto &graph_iter = tensor_with_graphs_.find(tensor);
      if (graph_iter != tensor_with_graphs_.end()) {
        const auto &data_prepare_actors = graph_iter->second;
        for (const auto &data_prepare_actor : data_prepare_actors) {
          const auto &iter = tensors_need_reprepare_.find(data_prepare_actor);
          if (iter != tensors_need_reprepare_.end()) {
            tensors_need_reprepare_[data_prepare_actor].insert(tensor);
          }
        }
      }
    };
    tensor->set_update_value_callback(callback);
  }

  if (tensor != nullptr && !tensors_need_reprepare_[this].empty() && tensor->is_parameter()) {
    auto erased_num = tensors_need_reprepare_[this].erase(tensor.get());
    MS_LOG(DEBUG) << "Erase " << erased_num << " tensor which is reprepared.";
  }
  return tensor;
}

void DataPrepareActor::PrepareDataForDeviceTensorStore(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                       const VectorRef &args, OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Prepare store data, input tensor size: " << input_tensors.size() << ", arg size: " << args.size();
  MSTX_RANGE_GUARD("PrepareStoreData", nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, "PrepareStoreData", true);
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  const auto &parser = graph_compiler_info_->control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  for (size_t i = 0; i < graph_compiler_info_->graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info_->graphs_[i];
    const auto &device_context = graph_compiler_info_->device_contexts_[i];

    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(DEBUG) << "prepare data for graph:" << graph->ToString();
    // Prepare the data of device tensor store(value nodes of graph).
    for (const auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (AnfAlgo::OutputAddrExist(value_node, 0)) {
        const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
        MS_EXCEPTION_IF_NULL(front_node);
        MS_LOG(DEBUG) << "Prepare data for value node:" << value_node->fullname_with_scope()
                      << ", debug name:" << value_node->DebugString() << ", front node:" << front_node->DebugString()
                      << " for graph:" << graph->ToString();
        const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
        const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        // If front_node has more than one device tensor, it means the node may used in multi graphs.
        // so we will clear the deviceaddress flag of ignore.
        if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr) && kernel_tensors.size() > 1) {
          kernel_tensor->ClearFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
        }
        // If node address has flag ignore, we will not prepare device data for it.
        if (!TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
          PrepareDataForValueNode(value_node, front_node, device_context, context);
        }
      }
    }

    // Uce error restart do not need to prepare weights.
    if (TftGetUceFlag()) {
      continue;
    }

    // Prepare the data of device tensor store(weights of graph).
    const auto &input_nodes = graph->input_nodes();
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);
      const auto &real_device_context = device::FetchRealDeviceContext(input_node, device_context);
      MS_EXCEPTION_IF_NULL(real_device_context);
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
      MS_LOG(DEBUG) << "Backend input node:" << input_node->DebugString()
                    << " front node:" << (front_node == nullptr ? "null" : front_node->DebugString())
                    << " backend is weight:" << IsPersistentDeviceTensor(input_node)
                    << " front is weight:" << parser->IsRootGraphPersistentDeviceTensor(front_node);
      if (IsPersistentDeviceTensor(input_node) && parser->IsRootGraphPersistentDeviceTensor(front_node)) {
        if (enable_input_optimize_) {
          PrepareWeightForInputOptimize(std::make_pair(front_node, 0), context);
          continue;
        }
        std::vector<TensorPtr> graph_tensors = input_tensors.empty() ? std::vector<TensorPtr>() : input_tensors[i];
        TensorPtr input_tensor = FetchInputTensor(graph_tensors, j, args, {front_node, 0});
        PrepareDataForWeightNode(input_node, front_node, input_tensor, real_device_context, context);
      }
    }
  }

  if (TftGetUceFlag()) {
    MS_LOG(INFO) << "Clear UCE state.";
    TftClearErrorType();
  }

  std::vector<TensorPtr> control_input = input_tensors.empty() ? std::vector<TensorPtr>() : input_tensors.back();
  PrepareDeviceTensorStoreForControlNode(parser, control_input, args, context);
  RaiseARFError(args);
}

void DataPrepareActor::RaiseARFError(const VectorRef &args) {
  if (TftIsRebootNode() && !TftIsArf() && !args.empty()) {
    MS_LOG(EXCEPTION) << "ARF FINISH , do next step repair weight.";
  }
}

void DataPrepareActor::PrepareDataForHostTensorQueue(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                     const VectorRef &args, OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Prepare host data, input tensor size: " << input_tensors.size() << ", arg size: " << args.size();
  MSTX_RANGE_GUARD("PrepareHostData", nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, "PrepareHostData", true);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  if ((host_data_source_actor_ == nullptr) || (host_tensor_queue_ == nullptr)) {
    return;
  }

  if (input_tensors.empty()) {
    PrepareDataForHostTensorQueueNew(args, context);
    return;
  }

  // Fill host tensors.
  std::vector<TensorPtr> host_tensors;
  host_tensors.resize(host_data_source_actor_->data_nodes().size());
  for (size_t i = 0; i < graph_compiler_info_->graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info_->graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);

    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    if (input_nodes.size() != tensors.size()) {
      std::string error_info = "Invalid tensor size:" + std::to_string(tensors.size()) +
                               " and input node size:" + std::to_string(input_nodes.size()) +
                               " for kernel graph:" + graph->ToString();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsHostQueueDSActor(input_node, graph, graph_compiler_info_->origin_parameters_order_, strategy_) ||
          input_tensor == nullptr) {
        continue;
      }

      auto tensor_position = host_data_source_actor_->FetchNodePosition({input_node, 0});
      if (tensor_position >= host_tensors.size()) {
        std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
      }
      MS_LOG(DEBUG) << "Set tensor position:" << tensor_position << " for input data.";
      host_tensors[tensor_position] = input_tensor;

      // Synchronize dynamic shape info of the input tensor to the parameter node of graph.
      if (graph->is_dynamic_shape()) {
        UpdateDynamicShapeAndSize(input_node, input_tensor);
      }

      UpdateDeviceAddressForDataNode(input_node, input_tensor);
    }
  }

  PrepareHostTensorQueueForControlNode(input_tensors.back(), &host_tensors, context);

  host_tensor_queue_->Push(host_tensors);
}

void DataPrepareActor::RecordGraphInputs(const std::vector<TensorPtr> &host_tensors,
                                         const std::vector<size_t> &host_param_indexes) {
  auto &llm_manager = LLMManager::GetInstance();
  for (size_t i = 0; i < host_tensors.size(); ++i) {
    auto host_tensor = host_tensors[i];
    auto param_index = host_param_indexes[i];
    const auto &origin_parameter = graph_compiler_info_->origin_parameters_order_[param_index];
    // host_tensor must not be nullptr
    llm_manager.add_graph_input(origin_parameter->fullname_with_scope(), host_tensor);
    MS_LOG(DEBUG) << "Add input tensor data for input parameter: " << origin_parameter->fullname_with_scope();
  }
}

void DataPrepareActor::RecordInputAndConvertStatic(const std::vector<TensorPtr> &host_tensors,
                                                   const std::vector<size_t> &host_param_indexes, bool isDyn) {
  if (is_enable_infer_boost_ && EnableKbkSubGraphExecute()) {
    RecordGraphInputs(host_tensors, host_param_indexes);
    MS_LOG(INFO) << "[Run into old runtime procedure] Prepare for actor set: " << graph_compiler_info_->name_
                 << ", convert static shape: " << (!isDyn) << ", dynamic shape: " << has_dynamic_shape_
                 << ", enable trace memory: " << enable_trace_memory_
                 << ", enable parallel dispatch: " << EnableParallelDispatchKernel()
                 << ", graph phase: " << graph_compiler_info_->graph_phase_;
    if (has_dynamic_shape_) {
      const auto &phase = graph_compiler_info_->graph_phase_;
      bool is_increment_graph = (phase.find("increment") != std::string::npos);
      if (is_increment_graph) {
        ActorDispatcher::set_enable_static_shape(!isDyn);
        if (enable_trace_memory_) {
          if (has_continuous_memory()) {
            MS_LOG(EXCEPTION)
              << "Can not support continuous memory allocate in dynamic shape graph when enable trace memory.";
          }
          if (!ActorDispatcher::enable_static_shape()) {
            ActorDispatcher::set_enable_trace_dynamic_memory(true);
          } else {
            ActorDispatcher::set_enable_use_trace_memory(true);
            ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_actor_set(EnableParallelDispatchKernel());
            if (ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set()) {
              MS_LOG(INFO) << "Enable parallel dispatch kernel for current actor set: " << graph_compiler_info_->name_
                           << ", graph phase: " << graph_compiler_info_->graph_phase_;
            }
          }
        }
      }
    }

    if (EnableRuntimeNewPipeline() && ActorDispatcher::enable_static_shape() &&
        ActorDispatcher::enable_runtime_multi_pipeline()) {
      // Not need infer and resize queue work in spin state when convert dynamic shape to static shape.
      RuntimePipeline::GetInstance().infer_queue()->Pause();
      RuntimePipeline::GetInstance().resize_queue()->Pause();
      ActorDispatcher::set_enable_runtime_multi_pipeline(false);
    }
  }
}

void DataPrepareActor::PrepareDataForHostTensorQueueNew(const VectorRef &args, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  size_t host_data_size = host_data_source_actor_->data_nodes().size();
  size_t current_data_num = 0;
  std::vector<TensorPtr> host_tensors;
  std::vector<size_t> host_param_indexes;
  host_tensors.resize(host_data_size);
  host_tensors_.resize(host_data_size);
  host_param_indexes.resize(host_data_size);
  bool isDyn = false;
  // Fill host tensors.
  for (size_t i = 0; i < graph_compiler_info_->origin_parameters_order_.size(); ++i) {
    if (current_data_num == host_data_size) {
      break;
    }
    const auto &origin_parameter = graph_compiler_info_->origin_parameters_order_[i];
    MS_EXCEPTION_IF_NULL(origin_parameter);
    // The input data is front of the parameter weight.
    if (common::AnfAlgo::IsParameterWeight(origin_parameter->cast<ParameterPtr>())) {
      MS_LOG(DEBUG) << "Skip the prepare host data for parameter: " << origin_parameter->fullname_with_scope();
      continue;
    }

    auto iter = graph_compiler_info_->origin_parameters_to_backend_parameters_.find(origin_parameter);
    if (iter == graph_compiler_info_->origin_parameters_to_backend_parameters_.end()) {
      MS_LOG(DEBUG) << "Not find the parameter in the origin parameters: " << origin_parameter->fullname_with_scope();
      continue;
    }

    for (auto origin_to_backend_pair : iter->second) {
      auto input_tensor = FetchInputTensorByArg(args, i, origin_to_backend_pair.first);
      if (input_tensor == nullptr) {
        MS_LOG(ERROR) << "The input tensor is nullptr for arg index: " << i
                      << ", parameter: " << origin_parameter->fullname_with_scope();
        continue;
      }
      // Single ops(run in pynative mode) output to net(context is graph mode) input.
      auto tensor_position = host_data_source_actor_->FetchNodePosition(origin_to_backend_pair.second);
      if (tensor_position >= host_tensors.size()) {
        std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
      }
      if (host_tensors[tensor_position] != nullptr) {
        continue;
      }
      input_tensor->set_name(origin_parameter->fullname_with_scope());
      MS_LOG(INFO) << "Set host tensor position:" << tensor_position
                   << " for input parameter:" << origin_parameter->fullname_with_scope();

      if (!isDyn) {
        if (host_tensors_[tensor_position] != input_tensor->shape() || input_tensor->shape().empty()) {
          isDyn = true;
        }
      }
      host_tensors_[tensor_position] = input_tensor->shape();
      host_tensors[tensor_position] = input_tensor;
      host_param_indexes[tensor_position] = i;
      ++current_data_num;

      UpdateDynamicShapeAndSize(origin_to_backend_pair.second.first, input_tensor);

      // Avoid the device `ptr_` being hold by the input tensor and the output tensor, the input tensor address cannot
      // be directly set to the input control node, which may be a passthrough node. The device 'ptr_' is re-malloced
      // and device to device copy by input tensor address in data source process.
      if (origin_to_backend_pair.first.first != origin_to_backend_pair.second.first) {
        UpdateDeviceAddressForDataNode(origin_to_backend_pair.second.first, input_tensor);
      }
    }
  }

  RecordInputAndConvertStatic(host_tensors, host_param_indexes, isDyn);
  host_tensor_queue_->Push(host_tensors);
}

//  The branch processing of PrepareDataForValueNode that value type is tensor.
void DataPrepareActor::PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                                     const AnfNodePtr &front_node, const DeviceContext *device_context,
                                                     OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);

  auto tensor = node_value->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);

  if (!(first_step_ || TftGetUceFlag())) {
    return;
  }

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->IsPtrValid()) {
    CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
    if (TftGetUceFlag()) {
      SyncTensorData(tensor, kernel_tensor, node, device_context, context, real_strategy_);
    }
    return;
  }

  SyncTensorData(tensor, kernel_tensor, node, device_context, context, real_strategy_);
  tensor->set_device_address(device_tensor);
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Prepare device data for value node: " << node->DebugString() << ", output index: " << 0
    << " device address:" << device_tensor << " ptr:" << device_tensor->GetPtr();
  CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
}

void DataPrepareActor::PrepareDataForControlValueNode(const KernelWithIndex &node_with_index,
                                                      const DeviceContext *device_context,
                                                      OpContext<KernelTensor> *const context,
                                                      const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  MS_EXCEPTION_IF_NULL(parser);
  if (!node_with_index.first->isa<ValueNode>()) {
    return;
  }

  const auto &node = node_with_index.first->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  size_t index = node_with_index.second;
  MS_LOG(DEBUG) << "Prepare data for control value node:" << node->DebugString() << " index:" << index;
  auto node_value = node->value();
  if (common::AnfAlgo::IsDynamicSequence(node)) {
    auto tensor = common::AnfAlgo::SequenceToTensor(node_value);
    parser->AddControlNodeTensor(tensor);
    node_value = tensor;
    AnfAlgo::UpdateValueNodeShape(node);
  }
  MS_EXCEPTION_IF_NULL(node_value);
  std::vector<ValuePtr> values;
  ValueTupleToValue(node_value, &values);

  if (node_with_index.second >= values.size()) {
    MS_LOG(INFO) << "Invalid index:" << node_with_index.second << " for node:" << node->DebugString();
    return;
  }
  const auto &value = values[index];
  MS_EXCEPTION_IF_NULL(value);
  TensorPtr tensor = nullptr;
  if (value->isa<StringImm>()) {
    PrepareDataForStringValue(node, index, node, device_context, context);
    return;
  } else if (!value->isa<tensor::Tensor>()) {
    tensor = parser->CreateTensorForValue(value);
  } else {
    tensor = value->cast<tensor::TensorPtr>();
  }

  MS_EXCEPTION_IF_NULL(tensor);
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }

  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForControlValueNode", node, graph_str, memory::mem_pool::MemType::kConstantValue,
                kernel_tensor);
  if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
  static std::string name = "Alloc memory";
  kernel_tensor->IncreaseNewRefCount(name);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PrepareDataForControlValueNode",
                                                 device::GetDeviceNameByType(device_tensor->GetDeviceType()),
                                                 device_tensor->GetPtr(), kernel_tensor->dtype_id(),
                                                 kernel_tensor->GetShapeVector(), kernel_tensor->tensor_storage_info());
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    auto output_address = reinterpret_cast<uintptr_t>(device_tensor.get());
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: PrepareDataForControlValueNode, device address class ptr: "
                    << output_address << ", node: " << node->fullname_with_scope() << ", graph: " << graph_str
                    << ", device address size: " << device_tensor->GetSize()
                    << ", device address addr: " << device_tensor->GetPtr();
  }

  if (device_tensor->GetSize() == 0) {
    MS_LOG(INFO) << "Empty tuple sync";
    return;
  }
  if (!device_context->device_res_manager_->SyncAllStreams() ||
      !SyncCopy(kernel_tensor.get(), tensor.get(), kDefaultStreamIndex)) {
    std::string error_info = "Sync host to device failed for node:" + node->DebugString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  tensor->set_device_address(device_tensor);
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
}

void DataPrepareActor::PrepareDataForStringValue(const ValueNodePtr &node, size_t index, const AnfNodePtr &front_node,
                                                 const DeviceContext *device_context,
                                                 OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<StringImm>(node)) {
    return;
  }
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // Copy data from value to device.
  auto copy_to_device = [&node_value, &kernel_tensor, this, &node, &context]() {
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    const auto &device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    // account '\0' to string size, keep consistent with method `CreateDeviceAddressForScalarAndString` defined in
    // `device_address_utils.cc`
    auto string_tensor =
      tensor::from_buffer(kObjectTypeString, shape, const_cast<void *>(kernel_tensor->GetValuePtr()), tensor_size);
    const auto &host_device_address = (dynamic_cast<device::DeviceAddress *>(string_tensor->device_address().get()));
    MS_EXCEPTION_IF_NULL(host_device_address);
    host_device_address->SetSize(tensor_size + 1);
    MS_LOG(DEBUG) << "Sync string to device for string:" << node_value->ToString() << " size:" << tensor_size;
    if (!SyncAllStreamForDeviceAddress(device_tensor, string_tensor->device_address()) ||
        !SyncCopy(kernel_tensor.get(), string_tensor.get(), kDefaultStreamIndex)) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  };

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    if (first_step_ || TftGetUceFlag()) {
      CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
    }
    if (TftGetUceFlag()) {
      copy_to_device();
    }
    return;
  }
  MS_LOG(INFO) << "Prepare device data for value node: " << node->DebugString();

  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForStringValue", node, graph_str, memory::mem_pool::MemType::kConstantValue, kernel_tensor);
  if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
  static std::string name = "Alloc memory";
  kernel_tensor->IncreaseNewRefCount(name);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PrepareDataForStringValue",
                                                 device::GetDeviceNameByType(device_tensor->GetDeviceType()),
                                                 device_tensor->GetPtr(), kernel_tensor->dtype_id(),
                                                 kernel_tensor->GetShapeVector(), kernel_tensor->tensor_storage_info());
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    auto output_address = reinterpret_cast<uintptr_t>(device_tensor.get());
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: PrepareDataForValueNode, device address class ptr: "
                    << output_address << ", device address size: " << device_tensor->GetSize()
                    << ", node: " << node->fullname_with_scope() << ", graph: " << graph_str
                    << ", device address addr: " << device_tensor->GetPtr();
  }

  copy_to_device();
  CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
}

void DataPrepareActor::PrepareDataForSequenceAndScalarValue(const ValueNodePtr &node, size_t index,
                                                            const AnfNodePtr &front_node,
                                                            const DeviceContext *device_context,
                                                            OpContext<KernelTensor> *const context) const {
  if (!(first_step_ || TftGetUceFlag())) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  if ((!node_value->isa<ValueSequence>()) && (!node_value->isa<Scalar>())) {
    return;
  }

  if (node_value->isa<ValueSequence>() && node_value->cast<ValueSequencePtr>()->size() == 0) {
    return;
  }

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto copy_to_device = [&kernel_tensor, &node, this, &context]() {
    const auto &device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    auto tensor = tensor::from_buffer(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(),
                                      const_cast<void *>(kernel_tensor->GetValuePtr()), kernel_tensor->size());
    if (!SyncAllStreamForDeviceAddress(device_tensor, tensor->device_address()) ||
        !SyncCopy(kernel_tensor.get(), tensor.get(), kDefaultStreamIndex)) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  };
  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
    if (TftGetUceFlag()) {
      copy_to_device();
    }
    return;
  }

  MS_LOG(DEBUG) << "Prepare device data for value node: " << node->DebugString();
  // 1. Allocate device memory for value node.
  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForSequenceAndScalarValue", node, graph_str, memory::mem_pool::MemType::kConstantValue,
                kernel_tensor);
  if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
  static std::string name = "Alloc memory";
  kernel_tensor->IncreaseNewRefCount(name);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, "PrepareDataForSequenceAndScalarValue",
                                                 device::GetDeviceNameByType(device_tensor->GetDeviceType()),
                                                 device_tensor->GetPtr(), kernel_tensor->dtype_id(),
                                                 kernel_tensor->GetShapeVector(), kernel_tensor->tensor_storage_info());
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    auto output_address = reinterpret_cast<uintptr_t>(device_tensor.get());
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: PrepareDataForValueNode, device address class ptr: "
                    << output_address << ", device address size: " << device_tensor->GetSize()
                    << ", node: " << node->fullname_with_scope() << ", graph: " << graph_str
                    << ", device address addr: " << device_tensor->GetPtr();
  }

  // 2. Sync copy data from host to device.
  copy_to_device();

  // 3. Handle heterogeneous scene.
  CopyDataFromDeviceTensorStore(front_node, node, kernel_tensor, device_context, context);
}

// Prepare the device data for persistent device tensor of value node.
void DataPrepareActor::PrepareDataForValueNode(const ValueNodePtr &node, const AnfNodePtr &front_node,
                                               const DeviceContext *device_context,
                                               OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  ValuePtr node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  if (TftGetUceFlag() && node_value->isa<tensor::Tensor>()) {
    // In UCE scenario, the constants value in device may be corrupted, so here restore from host backup values
    node_value = TftGetConstant(node);
  }
  MS_LOG(DEBUG) << "Prepare data for value node:" << node->DebugString() << " node addr:" << node.get()
                << " front node:" << front_node->DebugString() << " front node addr:" << front_node.get();
  if (node_value->isa<tensor::Tensor>()) {
    PrepareDataForValueNodeTensor(node, node_value, front_node, device_context, context);
  } else if (node_value->isa<ValueSequence>() || node_value->isa<Scalar>()) {
    PrepareDataForSequenceAndScalarValue(node, 0, front_node, device_context, context);
  } else if (node_value->isa<StringImm>()) {
    PrepareDataForStringValue(node, 0, front_node, device_context, context);
  } else if (node_value->isa<None>() || node_value->isa<Type>()) {
    MS_LOG(DEBUG) << "No need to prepare data for None or type value node:" << node->DebugString();
  } else {
    MS_LOG(WARNING) << "Not support the value type: " << node->fullname_with_scope();
  }
}

void DataPrepareActor::CopyDataFromDeviceTensorStore(const AnfNodePtr &front_node, const AnfNodePtr &backend_node,
                                                     const KernelTensorPtr &host_kernel_tensor,
                                                     const DeviceContext *device_context,
                                                     OpContext<KernelTensor> *context) const {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(host_kernel_tensor);
  const auto &host_tensor_address = host_kernel_tensor->device_address();
  const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  MS_LOG(DEBUG) << "Front node: " << front_node->DebugString() << ", node addr: " << front_node.get()
                << ", kernel tensor num is: " << kernel_tensors.size();
  for (auto &another_kernel_tensor : kernel_tensors) {
    if (another_kernel_tensor == nullptr) {
      MS_LOG(DEBUG) << "Current kernel tensor is null.";
      continue;
    }
    auto &another_device_tensor = another_kernel_tensor->device_address();
    if (another_device_tensor == host_tensor_address) {
      MS_LOG(DEBUG) << "Current kernel tensor: " << another_kernel_tensor->ToString() << " is continued.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(another_device_tensor);
    MS_LOG(DEBUG) << "Current kernel tensor: " << another_kernel_tensor->ToString();
    auto another_device_name = another_device_tensor->GetDeviceType();
    const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {another_device_name, device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(another_device_context);
    bool need_alloc_memory = (another_device_tensor->GetPtr() == nullptr);
    auto graph_str = (backend_node->func_graph() == nullptr) ? "" : backend_node->func_graph()->ToString();
    if (need_alloc_memory) {
      auto mem_type =
        backend_node->isa<ValueNode>() ? device::tracker::MemType::kConstantValue : memory::mem_pool::MemType::kWeight;
      UpdateTracker("CopyDataFromDeviceTensorStore", backend_node, graph_str, mem_type, another_kernel_tensor);
      if ((!another_device_context->device_res_manager_->AllocateMemory(another_device_tensor.get(),
                                                                        kDefaultStreamIndex))) {
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *another_device_context,
                                                    backend_node->fullname_with_scope(),
                                                    another_device_tensor->GetSize());
      }
      static std::string name = "Alloc memory";
      another_kernel_tensor->IncreaseNewRefCount(name);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        MarkTensorAsOutput, "CopyDataFromDeviceTensorStore",
        device::GetDeviceNameByType(another_device_tensor->GetDeviceType()), another_device_tensor->GetPtr(),
        another_kernel_tensor->dtype_id(), another_kernel_tensor->GetShapeVector(),
        another_kernel_tensor->tensor_storage_info());
    }
    if (memory::mem_pool::IsNeedProfilieMemoryLog() && need_alloc_memory) {
      auto output_address = reinterpret_cast<uintptr_t>(another_device_tensor.get());
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: CopyDataFromDeviceTensorStore, device address class ptr: "
                      << output_address << ", device address size: " << another_device_tensor->GetSize()
                      << ", device address addr: " << another_device_tensor->GetPtr()
                      << ", node: " << backend_node->fullname_with_scope() << ", graph: " << graph_str
                      << ", frontnode: " << (front_node == nullptr ? "null" : front_node->DebugString());
    }

    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device name:" << another_device_name << " from device address:" << host_tensor_address
                 << " to:" << another_device_tensor;
    auto skip_h2d = TftIsRebootNode();
    if (!skip_h2d && (!SyncAllStreamForDeviceAddress(another_device_tensor, host_tensor_address) ||
                      !SyncCopy(another_kernel_tensor.get(), host_kernel_tensor.get(), kDefaultStreamIndex))) {
      std::string error_info = "Sync data error.";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  }
}

// Prepare the device data for persistent device tensor of weight node from host tensor.
void DataPrepareActor::PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node,
                                                const TensorPtr &tensor, const DeviceContext *device_context,
                                                OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(context);
  auto param_node = backend_node->cast<ParameterPtr>();
  if (param_node != nullptr) {
    auto param_info = param_node->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    bool used = !param_info->ignore_device_addr();
    if (!used) {
      MS_LOG(WARNING) << backend_node->DebugString()
                      << " the Parameter is never used by real kernel in graphs, skip to allocate.";
      return;
    }
  }
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "Host tensor is empty for node:" << backend_node->DebugString();
    return;
  }

  auto node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(backend_node, 0, false);
  MS_EXCEPTION_IF_NULL(node_kernel_tensor);
  auto device_tensor = node_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto host_tensor_address = tensor->device_address();
  auto host_tensor_storage_info = tensor->storage_info();
  Format host_format = tensor->format();
  if (host_tensor_address == nullptr) {
    // Tensor with initializer but didn't init_data yet.
    auto empty_tensor = tensor::from_spec(tensor->data_type(), tensor->shape(), device::DeviceType::kCPU);
    tensor->set_device_address(empty_tensor->device_address());
    tensor->set_storage_info(nullptr);
    host_tensor_address = empty_tensor->device_address();
    host_tensor_storage_info = nullptr;
    MS_EXCEPTION_IF_NULL(host_tensor_address);
    MS_LOG(DEBUG) << "Create device address:" << host_tensor_address->ToString() << " for host tensor:" << tensor.get()
                  << " for backend node:" << backend_node->DebugString() << " front node:" << front_node->DebugString();
  }
  auto host_kernel_tensor = node_kernel_tensor->CloneKernelTensor();
  host_kernel_tensor->set_device_address(host_tensor_address);
  host_kernel_tensor->set_tensor_storage_info(host_tensor_storage_info);
  MS_LOG(DEBUG) << "Create kernel tensor:" << host_kernel_tensor->ToString();
  auto skip_h2d = TftIsRebootNode();
  // Use the device address of host tensor to set device tensor.
  bool is_need_sync = false;
  if (host_tensor_address != device_tensor) {
    if (host_tensor_address->GetDeviceType() != device_context->GetDeviceType()) {
      if (device_tensor->GetDeviceType() != device_context->GetDeviceType()) {
        const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
          {backend_node, 0}, nullptr, device_tensor->GetSize(),
          kernel::GetFormatFromEnumToStr(node_kernel_tensor->format()), node_kernel_tensor->dtype_id(),
          node_kernel_tensor->GetShapeVector(),
          device::GetDeviceNameByType(device_context->device_context_key().device_type_),
          device_context->device_context_key().device_id_);
        kernel_tensor->set_stream_id(device_tensor->stream_id());
        host_kernel_tensor = kernel_tensor;
        host_tensor_address = host_kernel_tensor->device_address();
        host_tensor_storage_info = host_kernel_tensor->tensor_storage_info();
        host_format = host_kernel_tensor->format();
        MS_EXCEPTION_IF_NULL(host_tensor_address);
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor:" << kernel_tensor->ToString();
        host_tensor_address->set_from_persistent_mem(tensor->is_parameter());
      } else {
        host_tensor_address = device_tensor;
        host_format = node_kernel_tensor->format();
        host_tensor_storage_info = node_kernel_tensor->tensor_storage_info();
      }
      is_need_sync = true;
    }
    MS_EXCEPTION_IF_NULL(host_tensor_address);

    if (host_tensor_address->GetDeviceType() != device_tensor->GetDeviceType()) {
      MS_LOG(INFO) << "The device type is not equal, host tensor type:" << host_tensor_address->GetDeviceType()
                   << ", device tensor type:" << device_tensor->GetDeviceType();
      // The fake heterogeneous scenario.
      if (DeviceTensorStore::GetInstance().Fetch(front_node.get()).size() == 1) {
        MS_LOG(EXCEPTION) << "Deprecated code is called. Execution aborted.";
      }
    } else if (host_tensor_address != device_tensor) {
      // In the scenario of training + inference , the device address of the weight node can not be changed when
      // multi-graphs sink mode is set.
      if (node_kernel_tensor->is_ptr_persisted() ||
          !AnfAlgo::IsEquivalentFormat(host_format, node_kernel_tensor->format())) {
        if ((device_tensor->GetPtr() == nullptr) &&
            (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex))) {
          SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context,
                                                      backend_node->fullname_with_scope(), device_tensor->GetSize());
        }
        static std::string name = "Alloc memory";
        host_kernel_tensor->IncreaseNewRefCount(name);
        if (!skip_h2d && (!SyncAllStreamForDeviceAddress(device_tensor, host_tensor_address) ||
                          !SyncCopy(node_kernel_tensor.get(), tensor.get(), kDefaultStreamIndex))) {
          std::string error_info = "Sync data error.";
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
        }
        host_tensor_address = device_tensor;
        host_tensor_storage_info = node_kernel_tensor->tensor_storage_info();
      } else {
        (void)address_modified_input_nodes_.insert(backend_node.get());
        AnfAlgo::SetOutputAddr(host_tensor_address, host_tensor_storage_info, 0, backend_node);
        node_kernel_tensor->set_format(host_format);
      }
    }
  }
  // Maybe the same host_tensor_address corresponds to the different front_node in shared weight scene,
  // so need update the device tensor store always.
  MS_EXCEPTION_IF_NULL(host_tensor_address);
  host_tensor_address->SetNodeIndex(backend_node, 0);
  host_kernel_tensor->set_device_address(host_tensor_address);
  host_kernel_tensor->set_tensor_storage_info(host_tensor_storage_info);
  DeviceTensorStore::GetInstance().Insert(front_node.get(), host_kernel_tensor);

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (is_need_sync || (!host_tensor_address->IsPtrValid())) {
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->DebugString()
                 << ", device type:" << host_tensor_address->GetDeviceType();
    SyncTensorData(tensor, host_kernel_tensor, backend_node, device_context, context, real_strategy_,
                   TftIsRebootNode());
  }
  tensor->set_device_address(host_tensor_address);
  tensor->set_storage_info(host_tensor_storage_info);

  // Allocate another device memory and copy data from host tensor to another device(if exist).
  CopyDataFromDeviceTensorStore(front_node, backend_node, host_kernel_tensor, device_context, context);
}

void DataPrepareActor::PrepareDeviceTensorStoreForControlNode(const ControlNodeParserPtr &control_node_parser,
                                                              const std::vector<TensorPtr> &tensors,
                                                              const VectorRef &args,
                                                              OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(control_node_parser);
  if (!control_node_parser->IsInited()) {
    return;
  }

  for (const auto &value_node_with_context : control_node_parser->front_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node_with_context.first.first);
    if (value_node_with_context.first.first->kernel_info() != nullptr &&
        AnfAlgo::OutputAddrExist(value_node_with_context.first.first, 0)) {
      PrepareDataForControlValueNode(value_node_with_context.first, value_node_with_context.second, context,
                                     control_node_parser);
    }
  }

  const auto &control_node_parameters = control_node_parser->control_node_parameters();
  if (!tensors.empty() && control_node_parameters.size() != tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid tensor size.");
  }
  for (size_t i = 0; i < control_node_parameters.size(); ++i) {
    auto &front_parameter = control_node_parameters[i].first;
    MS_EXCEPTION_IF_NULL(front_parameter);
    if (!control_node_parser->IsRootGraphPersistentDeviceTensor(front_parameter)) {
      continue;
    }

    if (enable_input_optimize_) {
      PrepareWeightForInputOptimize(control_node_parameters[i], context);
      continue;
    }

    TensorPtr tensor = FetchInputTensor(tensors, i, args, control_node_parameters[i]);
    if (tensor == nullptr) {
      continue;
    }

    auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_parameter.get());
    if (kernel_tensors.empty()) {
      MS_LOG(WARNING) << "Failed to get device tensor for front node:" << front_parameter->DebugString();
      continue;
    }
    auto kernel_tensor = kernel_tensors[0];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
    auto host_tensor_address = tensor->device_address();
    if ((kernel_tensor->device_address() == host_tensor_address) || (kernel_tensor->device_address()->IsPtrValid())) {
      continue;
    }

    auto node = (kernel_tensor->device_address()->GetNodeIndex()).first;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Prepare device data for weight node by root graph parameter:"
                 << front_parameter->fullname_with_scope() << ", backend node:" << node->DebugString()
                 << ", device type:" << kernel_tensor->device_address()->GetDeviceType();
    if (host_tensor_address == nullptr) {
      tensor->set_device_address(kernel_tensor->device_address());
      tensor->set_storage_info(kernel_tensor->tensor_storage_info());
      auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {kernel_tensor->device_address()->GetDeviceType(), kernel_tensor->device_address()->device_id()});
      SyncTensorData(tensor, kernel_tensor, node, device_context, context, GraphExecutionStrategy::kPipeline,
                     TftIsRebootNode());
    } else {
      if (host_tensor_address->GetSize() != kernel_tensor->device_address()->GetSize()) {
        MS_LOG(WARNING) << "Please check the size of parameter:" << front_parameter->fullname_with_scope()
                        << ", host tensor size:" << host_tensor_address->GetSize()
                        << ", device tensor size:" << kernel_tensor->device_address()->GetSize();
      }
      host_tensor_address->SetNodeIndex(node, 0);
      kernel_tensor->set_device_address(host_tensor_address);
      kernel_tensor->set_tensor_storage_info(tensor->storage_info());
      DeviceTensorStore::GetInstance().Remove(front_parameter.get());
      DeviceTensorStore::GetInstance().Insert(front_parameter.get(), kernel_tensor);
    }
  }
}

void DataPrepareActor::PrepareHostTensorQueueForControlNode(const std::vector<TensorPtr> &tensors,
                                                            std::vector<TensorPtr> *const host_tensors,
                                                            OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  MS_EXCEPTION_IF_NULL(graph_compiler_info_->control_node_parser_);
  MS_EXCEPTION_IF_NULL(host_data_source_actor_);
  MS_EXCEPTION_IF_NULL(host_tensors);

  const auto &control_node_parameters = graph_compiler_info_->control_node_parser_->control_node_parameters();
  for (size_t i = 0; i < control_node_parameters.size(); ++i) {
    const auto &input_node = control_node_parameters[i].first;
    const auto &input_tensor = tensors[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      continue;
    }

    if (find(graph_compiler_info_->origin_parameters_order_.begin(),
             graph_compiler_info_->origin_parameters_order_.end(),
             input_node) == graph_compiler_info_->origin_parameters_order_.end()) {
      continue;
    }

    auto tensor_position = host_data_source_actor_->FetchNodePosition(control_node_parameters[i]);
    if (tensor_position >= host_tensors->size()) {
      std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
    if ((*host_tensors)[tensor_position] != nullptr) {
      continue;
    }
    MS_LOG(DEBUG) << "Set tensor position:" << tensor_position << " for input data.";
    (*host_tensors)[tensor_position] = input_tensor;

    UpdateDynamicShapeAndSize(input_node, input_tensor);
    // Avoid the device `ptr_` being hold by the input tensor and the output tensor, the input tensor address cannot
    // be directly set to the input control node, which may be a passthrough node. The device 'ptr_' is re-malloced
    // and device to device copy by input tensor address in data source process.
  }
}

void DataPrepareActor::PrepareWeightForInputOptimize(const KernelWithIndex &node_with_index,
                                                     OpContext<KernelTensor> *const context) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  const auto &front_node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(front_node);
  auto outer_idx = graph_parameter_store->GetFrontNodeToIndex(front_node.get());
  (void)FetchParameter(std::make_pair(node_with_index, outer_idx), GetAID());
  // Record the update tensors for reprepare.
  // This will be removed after input optimization performance improved.
  if (is_enable_infer_boost_) {
    auto tensor = graph_parameter_store->FetchTensor(outer_idx, node_with_index);
    RecordTensorsNeedReprepare(tensor);
  }
}

void DataPrepareActor::PreprocessBeforePrepareData() const {
  // Try to defrag memory.
  auto defrag_memory_step_freq = GetDefragMemoryStepFreq();
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  if (!runtime_conf_instance->GetEnableKernelLaunchCapture() && ++execution_count_ % defrag_memory_step_freq == 0) {
    std::set<const DeviceContext *> defrag_memory_contexts;
    for (auto &device_context : graph_compiler_info_->device_contexts_) {
      MS_EXCEPTION_IF_NULL(device_context);
      if ((defrag_memory_contexts.count(device_context) == 0)) {
        device_context->device_res_manager_->DefragMemory();
      }
      (void)defrag_memory_contexts.insert(device_context);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
