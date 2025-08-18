/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "backend/ge_backend/runtime/actor/data_prepare_actor.h"

#include <algorithm>
#include <set>
#include <memory>
#include <vector>
#include <string>

#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/loop_count_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "backend/ge_backend/runtime/actor/profiler_actor.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "ir/tensor_new.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "utils/llm_manager.h"
#include "utils/convert_utils.h"
#include "include/backend/common/ms_device_shape_transfer.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "tools/profiler/mstx/mstx_impl.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
void UpdateTracker(const std::string &task_name, const AnfNodePtr &node, const std::string &graph_str,
                   memory::mem_pool::MemType mem_type, const DeviceTensorPtr &device_tensor) {
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node->fullname_with_scope(), graph_str);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, task_name, mem_type, device_tensor->GetSize(),
                                                 device_tensor.get());
}

void SyncTensorData(const TensorPtr &host_tensor, const KernelTensorPtr &kernel_tensor, const AnfNodePtr &node,
                    OpContext<KernelTensor> *const context, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(context);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  bool need_alloc_memory = (device_tensor->GetPtr() == nullptr);
  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  auto mem_type =
    node->isa<ValueNode>() ? memory::mem_pool::MemType::kConstantValue : memory::mem_pool::MemType::kWeight;
  UpdateRefCount(kernel_tensor, true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (need_alloc_memory) {
    UpdateTracker("SyncTensorData", node, graph_str, mem_type, device_tensor);
    if (!host_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, *context, node->fullname_with_scope(),
                                                  device_tensor->GetSize());
    }
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
  // Copy data from host tensor to device.
  auto host_tensor_size = LongToSize(host_tensor->DataNBytes());
  auto host_tensor_type = host_tensor->data_type();
  if (!host_context->device_res_manager_->SyncAllStreams() ||
      !SyncCopy(kernel_tensor.get(), host_tensor.get(), kDefaultStreamIndex)) {
    std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", host tensor size: " + std::to_string(host_tensor_size) +
                             ", host tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device tensor size: " + std::to_string(device_tensor->GetSize());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy, (*context), error_info);
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

void UpdateDataNodeDeviceAddressSize(const AnfNodePtr &input_node, const TensorPtr &input_tensor,
                                     const kernel::KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_address = kernel_tensor->device_address();
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
}  // namespace

std::atomic<size_t> DataPrepareActor::execution_count_ = 0;
mindspore::HashMap<const DataPrepareActor *, mindspore::HashSet<const tensor::Tensor *>>
  DataPrepareActor::tensors_need_reprepare_ = {};
mindspore::HashMap<const tensor::Tensor *, mindspore::HashSet<const DataPrepareActor *>>
  DataPrepareActor::tensor_with_graphs_ = {};

void DataPrepareActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  strategy_ = graph_compiler_info_->strategy_;

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
  auto device_format = kernel::GetFormatFromEnumToStr(kernel_tensor->format());
  static const std::set<std::string> kNormalFormat = {
    kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->DataNBytes();
    MS_LOG(DEBUG) << "Set device address:" << device_address << " size from:" << device_address->GetSize()
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

  auto tensor_address = input_tensor->device_address();
  if (tensor_address == nullptr) {
    return;
  }

  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (tensor_address == device_address) {
    tensor_address->SetNodeIndex(input_node, 0);
    kernel_tensor->set_original_ref_count(SIZE_MAX);
    kernel_tensor->ResetRefCount();
    return;
  }

  // If tensor address and device address are different (heterogeneous scenarios), or device address is persisted
  // Update device address data in data source actor process.
  if (kernel_tensor->is_ptr_persisted() ||
      (tensor_address->GetDeviceType() != device_address->GetDeviceType() ||
       (!AnfAlgo::IsEquivalentFormat(input_tensor->format(), kernel_tensor->format())) ||
       input_tensor->data_type() != kernel_tensor->dtype_id())) {
    MS_LOG(DEBUG) << "Cannot update address of " << input_node->DebugString();
    return;
  }

  // Assign tensor address to input data node and set `ref_count` to `SIZE_MAX` for avoiding clean.
  (void)address_modified_input_nodes_.insert(input_node.get());
  device_address->set_device_pointer(tensor_address->device_pointer());
  MS_LOG(DEBUG) << "Update device address of " << input_node->DebugString() << " to " << tensor_address.get()
                << " ptr:" << tensor_address->GetPtr();
  device_address->SetNodeIndex(input_node, 0);
  kernel_tensor->set_original_ref_count(SIZE_MAX);
  kernel_tensor->ResetRefCount();
  auto ref_iter = ref_kernel_tensors_.find({input_node, 0});
  if (ref_iter == ref_kernel_tensors_.end()) {
    return;
  }
  for (const auto &ref_kernel_tensor : ref_iter->second) {
    if (ref_kernel_tensor == nullptr) {
      continue;
    }
    ref_kernel_tensor->set_pointer_ref_count(kernel_tensor.get());
    MS_LOG(DEBUG) << "Set pointer ref count:" << tensor_address->device_pointer()
                  << " from node:" << input_node->DebugString() << " kernel tensor:" << kernel_tensor->ToString()
                  << " to device address:" << ref_kernel_tensor->ToString();
  }
}

void DataPrepareActor::PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
                                   OpContext<KernelTensor> *const context, GraphExecutionStrategy real_strategy) {
  MS_EXCEPTION_IF_NULL(context);
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  MS_LOG(DEBUG) << "Data prepare actor(" << GetAID().Name() << ") prepares data.";
  real_strategy_ = real_strategy;
  try {
    bool not_empty_input = !input_tensors.empty() || !args.empty();
    if (first_step_ || (enable_prepare_case() && not_empty_input)) {
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

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name(), false);

  PostRun(context);
}

void DataPrepareActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugOnStepBegin, graph_compiler_info_->graphs_,
                            graph_compiler_info_->origin_parameters_order_, context, &GetAID());
  OnDebugFinish(context);
}

void DataPrepareActor::SendProfilerReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  ActorDispatcher::SendSync(*profiler_aid_, &ProfilerActor::ProfilerOnStepBegin, graph_compiler_info_->graphs_,
                            graph_compiler_info_->origin_parameters_order_, context, &GetAID());
  OnDebugFinish(context);
}

void DataPrepareActor::OnDebugFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  PostRun(context);
}

TensorPtr DataPrepareActor::FetchInputTensor(const std::vector<TensorPtr> &tensors, size_t tensor_index,
                                             const VectorRef &args, const KernelWithIndex &front_node) {
  if (!tensors.empty()) {
    MS_EXCEPTION_IF_CHECK_FAIL((tensor_index < tensors.size()), "The tensor index is out of range.");
    auto tensor = tensors[tensor_index];
    // The tensor needs to be converted to contiguous before being given to the actors.
    // After the view feature is supported in the graph mode, the following code will be deleted.
    if (!DeviceAddressUtils::IsContiguousTensor(tensor)) {
      MS_LOG(EXCEPTION) << "The ge backend only support contiguous inputs, please check.";
    }
    return tensor;
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

  // The tensor needs to be converted to contiguous before being given to the actors.
  // After the view feature is supported in the graph mode, the following code will be deleted.
  if (!DeviceAddressUtils::IsContiguousTensor(tensor)) {
    MS_LOG(EXCEPTION) << "The ge backend only support contiguous inputs, please check.";
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
    const auto &graph_executor = graph_compiler_info_->graph_executor_;
    // alloc graph fixed memory
    if (graph_executor != nullptr) {
      graph_executor->AllocGEFixMemory();
    }
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
          PrepareDataForValueNode(value_node, front_node, context);
        }
      }
    }

    // Prepare the data of device tensor store(weights of graph).
    const auto &input_nodes = graph->input_nodes();
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
      MS_LOG(DEBUG) << "Backend input node:" << input_node->DebugString()
                    << " front node:" << (front_node == nullptr ? "null" : front_node->DebugString())
                    << " backend is weight:" << IsPersistentDeviceTensor(input_node)
                    << " front is weight:" << parser->IsRootGraphPersistentDeviceTensor(front_node);
      if (IsPersistentDeviceTensor(input_node) && parser->IsRootGraphPersistentDeviceTensor(front_node)) {
        std::vector<TensorPtr> graph_tensors = input_tensors.empty() ? std::vector<TensorPtr>() : input_tensors[i];
        TensorPtr input_tensor = FetchInputTensor(graph_tensors, j, args, {front_node, 0});
        PrepareDataForWeightNode(input_node, front_node, input_tensor, context);
      }
    }
  }

  std::vector<TensorPtr> control_input = input_tensors.empty() ? std::vector<TensorPtr>() : input_tensors.back();
  PrepareDeviceTensorStoreForControlNode(parser, control_input, args, context);
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
  }
}

void DataPrepareActor::PrepareDataForHostTensorQueueNew(const VectorRef &args, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  size_t host_data_size = host_data_source_actor_->data_nodes().size();
  size_t current_data_num = 0;
  std::vector<TensorPtr> host_tensors;
  host_tensors.resize(host_data_size);
  host_tensors_.resize(host_data_size);
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

      host_tensors_[tensor_position] = input_tensor->shape();
      host_tensors[tensor_position] = input_tensor;
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
  host_tensor_queue_->Push(host_tensors);
}

//  The branch processing of PrepareDataForValueNode that value type is tensor.
void DataPrepareActor::PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                                     const AnfNodePtr &front_node,
                                                     OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(context);

  auto tensor = node_value->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);

  if (!first_step_) {
    return;
  }

  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->IsPtrValid()) {
    return;
  }

  SyncTensorData(tensor, kernel_tensor, node, context, real_strategy_);
  tensor->set_device_address(device_tensor);
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  UpdateRefCount(kernel_tensor, true);
  MS_LOG(DEBUG) << "Prepare device data for value node: " << node->DebugString() << ", output index: " << 0
                << " kernel tensor:" << kernel_tensor->ToString();
}

void DataPrepareActor::PrepareDataForControlValueNode(const KernelWithIndex &node_with_index,
                                                      OpContext<KernelTensor> *const context,
                                                      const ControlNodeParserPtr &parser) const {
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
    PrepareDataForStringValue(node, index, node, context);
    return;
  } else if (!value->isa<tensor::Tensor>()) {
    tensor = parser->CreateTensorForValue(value);
  } else {
    tensor = value->cast<tensor::TensorPtr>();
  }

  MS_EXCEPTION_IF_NULL(tensor);
  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }
  if (tensor->data_type() != kernel_tensor->dtype_id() || tensor->shape() != kernel_tensor->GetShapeVector() ||
      tensor->format() != kernel_tensor->format()) {
    MS_LOG(INFO) << "Not same info for tensor:" << tensor->ToString() << " type:" << TypeIdLabel(tensor->data_type())
                 << " shape:" << tensor->shape() << " format:" << tensor->format()
                 << " and kernel tensor:" << kernel_tensor->ToString();
  }
  tensor->set_device_address(device_tensor);
  tensor->set_storage_info(kernel_tensor->tensor_storage_info());
  UpdateRefCount(kernel_tensor, true);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForControlValueNode", node, graph_str, memory::mem_pool::MemType::kConstantValue,
                device_tensor);
  if (!host_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
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
  if (!host_context->device_res_manager_->SyncAllStreams() ||
      !SyncCopy(kernel_tensor.get(), tensor.get(), kDefaultStreamIndex)) {
    std::string error_info = "Sync host to device failed for node:" + node->DebugString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
}

void DataPrepareActor::PrepareDataForStringValue(const ValueNodePtr &node, size_t index, const AnfNodePtr &front_node,
                                                 OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<StringImm>(node)) {
    return;
  }
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // Copy data from value to device.
  auto copy_to_device = [&node_value, &device_tensor, this, &node, &context, index]() {
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    // account '\0' to string size, keep consistent with method `CreateDeviceAddressForScalarAndString` defined in
    // `device_address_utils.cc`
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto string_tensor =
      tensor::from_buffer(kObjectTypeString, shape, const_cast<void *>(kernel_tensor->GetValuePtr()), tensor_size);
    const auto &host_device_address = (dynamic_cast<device::DeviceAddress *>(string_tensor->device_address().get()));
    MS_EXCEPTION_IF_NULL(host_device_address);
    host_device_address->SetSize(tensor_size + 1);
    MS_LOG(DEBUG) << "Sync string to device for string:" << node_value->ToString() << " size:" << tensor_size;
    device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    if (!host_context->device_res_manager_->SyncAllStreams() ||
        !SyncCopy(kernel_tensor.get(), string_tensor.get(), kDefaultStreamIndex)) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  };

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }
  MS_LOG(INFO) << "Prepare device data for value node: " << node->DebugString();

  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForStringValue", node, graph_str, memory::mem_pool::MemType::kConstantValue, device_tensor);
  UpdateRefCount(kernel_tensor, true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
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
}

void DataPrepareActor::PrepareDataForSequenceAndScalarValue(const ValueNodePtr &node, size_t index,
                                                            const AnfNodePtr &front_node,
                                                            OpContext<KernelTensor> *const context) const {
  if (!first_step_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  if ((!node_value->isa<ValueSequence>()) && (!node_value->isa<Scalar>())) {
    return;
  }

  if (node_value->isa<ValueSequence>() && node_value->cast<ValueSequencePtr>()->size() == 0) {
    return;
  }

  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto copy_to_device = [&device_tensor, &node, this, &context, index]() {
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto tensor = tensor::from_buffer(kernel_tensor->dtype_id(), kernel_tensor->GetShapeVector(),
                                      const_cast<void *>(kernel_tensor->GetValuePtr()), kernel_tensor->size());
    device::DeviceContextKey host_key = {device_tensor->GetDeviceType(), device_tensor->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);

    if (!host_context->device_res_manager_->SyncAllStreams() ||
        !SyncCopy(kernel_tensor.get(), tensor.get(), kDefaultStreamIndex)) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  };
  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }

  UpdateRefCount(kernel_tensor, true);
  MS_LOG(DEBUG) << "Prepare device data for value node: " << node->DebugString();
  // 1. Allocate device memory for value node.
  auto graph_str = (node->func_graph() == nullptr) ? "" : node->func_graph()->ToString();
  UpdateTracker("PrepareDataForSequenceAndScalarValue", node, graph_str, memory::mem_pool::MemType::kConstantValue,
                device_tensor);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  if (!host_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }
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
}

// Prepare the device data for persistent device tensor of value node.
void DataPrepareActor::PrepareDataForValueNode(const ValueNodePtr &node, const AnfNodePtr &front_node,
                                               OpContext<KernelTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  MS_LOG(DEBUG) << "Prepare data for value node:" << node->DebugString() << " front node:" << front_node->DebugString();
  if (node_value->isa<tensor::Tensor>()) {
    PrepareDataForValueNodeTensor(node, node_value, front_node, context);
  } else if (node_value->isa<ValueSequence>() || node_value->isa<Scalar>()) {
    PrepareDataForSequenceAndScalarValue(node, 0, front_node, context);
  } else if (node_value->isa<StringImm>()) {
    PrepareDataForStringValue(node, 0, front_node, context);
  } else if (node_value->isa<None>() || node_value->isa<Type>()) {
    MS_LOG(DEBUG) << "No need to prepare data for None or type value node:" << node->DebugString();
  } else {
    MS_LOG(WARNING) << "Not support the value type: " << node->fullname_with_scope();
  }
}

// Prepare the device data for persistent device tensor of weight node from host tensor.
void DataPrepareActor::PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node,
                                                const TensorPtr &tensor, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(context);
  auto param_node = backend_node->cast<ParameterPtr>();
  if (param_node != nullptr) {
    auto param_info = param_node->param_info();
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
  // Host kernel tensor replace the input tensor data.
  auto host_kernel_tensor = node_kernel_tensor->CloneKernelTensor();
  host_kernel_tensor->set_device_address(host_tensor_address);
  host_kernel_tensor->set_format(tensor->format());
  host_kernel_tensor->set_tensor_storage_info(host_tensor_storage_info);
  // Use the device address of host tensor to set device tensor.
  bool is_need_sync = false;

  if (host_tensor_address != device_tensor) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (host_tensor_address->GetDeviceType() != device::GetDeviceTypeByName(device_name)) {
      if (device_tensor->GetDeviceType() != device::GetDeviceTypeByName(device_name)) {
        MS_LOG(EXCEPTION) << "Invalid device tensor type" << device_tensor->GetDeviceType()
                          << " for input node:" << backend_node->DebugString()
                          << " for ge backend device type:" << device::GetDeviceTypeByName(device_name);
      }
      host_tensor_address = device_tensor;
      host_kernel_tensor->set_format(node_kernel_tensor->format());
      host_tensor_storage_info = node_kernel_tensor->tensor_storage_info();
      is_need_sync = true;
      UpdateRefCount(host_kernel_tensor, true);
    }
  }
  // Maybe the same host_tensor_address corresponds to the different front_node in shared weight scene,
  // so need update the device tensor store always.
  MS_EXCEPTION_IF_NULL(host_tensor_address);
  host_tensor_address->SetNodeIndex(backend_node, 0);
  host_kernel_tensor->set_device_address(host_tensor_address);
  host_kernel_tensor->set_tensor_storage_info(host_tensor_storage_info);
  UpdateRefCount(host_kernel_tensor, true);
  DeviceTensorStore::GetInstance().Insert(front_node.get(), host_kernel_tensor);

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  if (is_need_sync || (!host_tensor_address->IsPtrValid())) {
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->DebugString()
                 << ", device type:" << host_tensor_address->GetDeviceType();
    SyncTensorData(tensor, host_kernel_tensor, backend_node, context, real_strategy_);
  }
  tensor->set_device_address(host_tensor_address);
  tensor->set_format(host_kernel_tensor->format());
  tensor->set_storage_info(host_tensor_storage_info);
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
    MS_EXCEPTION_IF_NULL(value_node_with_context.first);
    if (value_node_with_context.first->kernel_info() != nullptr &&
        AnfAlgo::OutputAddrExist(value_node_with_context.first, 0)) {
      PrepareDataForControlValueNode(value_node_with_context, context, control_node_parser);
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

    TensorPtr tensor = FetchInputTensor(tensors, i, args, control_node_parameters[i]);
    if (tensor == nullptr) {
      continue;
    }

    auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_parameter.get());
    if (kernel_tensors.empty()) {
      MS_LOG(WARNING) << "Failed to get device tensor for front node:" << front_parameter->DebugString();
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_tensors[0]);
    MS_EXCEPTION_IF_NULL(kernel_tensors[0]->device_address());
    auto host_tensor_address = tensor->device_address();
    if (host_tensor_address == nullptr || (kernel_tensors[0]->device_address() == host_tensor_address) ||
        (kernel_tensors[0]->IsPtrValid())) {
      continue;
    }

    auto node = (kernel_tensors[0]->device_address()->GetNodeIndex()).first;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Prepare device data for weight node by root graph parameter:"
                 << front_parameter->fullname_with_scope() << ", backend node:" << node->DebugString()
                 << ", device type:" << kernel_tensors[0]->GetDeviceType();
    if (host_tensor_address->GetDeviceType() != kernel_tensors[0]->device_address()->GetDeviceType()) {
      SyncTensorData(tensor, kernel_tensors[0], node, context, GraphExecutionStrategy::kPipeline);
      tensor->set_device_address(kernel_tensors[0]->device_address());
      tensor->set_format(kernel_tensors[0]->format());
      tensor->set_storage_info(kernel_tensors[0]->tensor_storage_info());
    } else {
      if (host_tensor_address->GetSize() != kernel_tensors[0]->GetSize()) {
        MS_LOG(WARNING) << "Please check the size of parameter:" << front_parameter->fullname_with_scope()
                        << ", host tensor size:" << host_tensor_address->GetSize()
                        << ", device tensor size:" << kernel_tensors[0]->GetSize();
      }
      host_tensor_address->SetNodeIndex(node, 0);
      UpdateRefCount(kernel_tensors[0], true);
      kernel_tensors[0]->set_device_address(host_tensor_address);
      kernel_tensors[0]->set_format(tensor->format());
      kernel_tensors[0]->set_tensor_storage_info(tensor->storage_info());
      DeviceTensorStore::GetInstance().Remove(front_parameter.get());
      DeviceTensorStore::GetInstance().Insert(front_parameter.get(), kernel_tensors[0]);
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
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
