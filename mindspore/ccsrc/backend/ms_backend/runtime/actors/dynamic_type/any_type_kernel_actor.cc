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

#include "backend/ms_backend/runtime/actors/dynamic_type/any_type_kernel_actor.h"
#include <set>
#include <unordered_map>
#include <functional>
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "primitive/framework_ops.h"
#include "include/utils/pipeline/fallback.h"
#include "ir/graph_utils.h"
#include "include/backend/common/kernel_graph/py_execute_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "runtime/hardware_abstract/utils.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace runtime {
namespace {
using AddressPtr = kernel::AddressPtr;
using PyExecuteOutputUserData = kernel::PyExecuteOutputUserData;
}  // namespace

std::mutex AnyTypeKernelActor::instance_lock_;

AnyTypeKernelActor::AnyTypeKernelActor(const std::string &name, const KernelGraphPtr &graph,
                                       const DeviceContext *device_context, const AID &memory_manager_aid,
                                       const AID *debug_aid, const AID *recorder_aid, KernelTransformType type)
    : SuperKernelActor(name, graph, "", device_context, memory_manager_aid, debug_aid, recorder_aid, type) {
  enable_kbk_sub_graph_execute_ = true;
}

void AnyTypeKernelActor::FetchInputDeviceTensor(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return;
  }
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    MS_EXCEPTION_IF_NULL(input_data->data_);
    size_t index = IntToSize(input_data->index_);
    if (index >= input_kernel_tensors_.size()) {
      std::string error_info = "Invalid graph input index:" + std::to_string(index) +
                               " total:" + std::to_string(input_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[index] = input_data->data_;
  }

  for (auto &device_tensor_store_key : extern_device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid device context for any type actor:" + GetAID().Name());
    }
    auto kernel_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceType());
    if (kernel_tensor == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, device_tensor_store_key.second)
        << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
        << " index:" << device_tensor_store_key.first << " device type:" << device_contexts_[0]->GetDeviceType();
      continue;
    }
    if (device_tensor_store_key.first >= input_kernel_tensors_.size()) {
      std::string error_info =
        "Invalid graph input device tensor store index:" + std::to_string(device_tensor_store_key.first) +
        " total:" + std::to_string(input_kernel_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[device_tensor_store_key.first] = kernel_tensor;
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Fetch device tensor store:" << kernel_tensor->ToString()
      << " by key:" << device_tensor_store_key.second->DebugString() << " index:" << device_tensor_store_key.first
      << " for actor:" << GetAID();
  }
  for (const auto &parameter_index : extern_parameter_indexs_) {
    // Collect the input kernel tensor.
    auto kernel_tensor = FetchParameter(parameter_index.second, GetAID());
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    input_kernel_tensors_[parameter_index.first] = kernel_tensor;
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Fetch parameter store:" << kernel_tensor->ToString() << " by outer index:" << parameter_index.second.second
      << " inner index:" << parameter_index.second.first.second << " for actor:" << GetAID();
  }
}
namespace {
GraphSegmentPtr BuildSegmentByGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> nodes;
  std::vector<AnfNodePtr> all_nodes = TopoSort(graph->get_return());
  for (const auto &node : all_nodes) {
    if (node == nullptr || (!node->isa<CNode>()) || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    MS_LOG(DEBUG) << "build new segment node:" << node->DebugString();
    nodes.emplace_back(node);
  }
  return std::make_shared<GraphSegment>(nodes, false);
}

std::string GenerateIDForGraph(const std::vector<KernelTensorPtr> &kernel_tensors, const std::vector<size_t> &indexes) {
  std::string id;
  auto get_shape_and_type_string = [&id](const ShapeVector &shape_vector, TypeId type_id) {
    id += "shape_";
    (void)std::for_each(shape_vector.begin(), shape_vector.end(), [&id](int64_t shape) {
      id += std::to_string(shape);
      id += "_";
    });
    id = id + "type_" + std::to_string(type_id) + "_";
  };
  for (const auto &index : indexes) {
    if (index >= kernel_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid parameter index:" << index << " for device tensor num:" << kernel_tensors.size();
    }
    id = id + "index_" + std::to_string(index) + "_";
    const auto &kernel_tensor = kernel_tensors[index];
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      MS_LOG(EXCEPTION) << "Empty device tensor index:" << index;
    }
    if (kernel_tensor->user_data() == nullptr) {
      get_shape_and_type_string(kernel_tensor->GetShapeVector(), kernel_tensor->dtype_id());
      continue;
    }

    const auto &user_data_obj =
      kernel_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    if (user_data_obj == nullptr) {
      MS_LOG(ERROR) << "Failed to get user data from input index:" << index
                    << " kernel tensor:" << kernel_tensor->ToString();
      return "FAILED";
    }
    const auto &obj = user_data_obj->obj;
    py::gil_scoped_acquire gil_acquire;
    const auto &abstract = pyexecute::GenerateAbstractFromPyObject(obj);
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractSequence>()) {
      auto sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      id = id + "Tuple_" + std::to_string(sequence_abs->size()) + "_";
    } else if (abstract->isa<abstract::AbstractScalar>()) {
      id = id + "Scalar_";
    } else if (abstract->isa<abstract::AbstractTensor>()) {
      id = id + "Tensor_";
    }
    get_shape_and_type_string(kernel_tensor->GetShapeVector(), kernel_tensor->dtype_id());
  }
  return id;
}

void InferParameterAbstractForModelGraph(const KernelGraphPtr &graph,
                                         const std::vector<KernelTensorPtr> &kernel_tensors,
                                         const std::vector<size_t> &indexes) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t index : indexes) {
    if (index >= kernel_tensors.size() || index >= graph->input_nodes().size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for input kernel tensor size:" << kernel_tensors.size()
                        << " for graph:" << graph->ToString();
    }
    const auto &kernel_tensor = kernel_tensors[index];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
    auto input_node = graph->input_nodes()[index];
    MS_EXCEPTION_IF_NULL(input_node);
    abstract::AbstractBasePtr abstract;
    if (kernel_tensor->user_data() != nullptr &&
        kernel_tensor->user_data()->has(kernel::PyExecuteOutputUserData::key)) {
      MS_LOG(DEBUG) << "User data:" << kernel_tensor->user_data()
                    << " in device address:" << kernel_tensor->device_address()
                    << " for input:" << input_node->DebugString();
      const auto &user_data_obj =
        kernel_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
      MS_EXCEPTION_IF_NULL(user_data_obj);
      const auto &obj = user_data_obj->obj;
      py::gil_scoped_acquire gil_acquire;
      abstract = pyexecute::GenerateAbstractFromPyObject(obj);
    } else {
      TypePtr type = kernel_tensor->GetType();
      BaseShapePtr shape = kernel_tensor->GetShape();
      MS_EXCEPTION_IF_NULL(type);
      MS_EXCEPTION_IF_NULL(shape);
      if (type->isa<Tuple>() && shape->isa<abstract::TupleShape>()) {
        const auto &tuple_type = type->cast<TuplePtr>();
        const auto &tuple_shape = shape->cast<abstract::TupleShapePtr>();
        MS_EXCEPTION_IF_NULL(tuple_shape);
        MS_EXCEPTION_IF_NULL(tuple_type);
        if (tuple_type->dynamic_len() && tuple_type->dynamic_element_type() != nullptr && tuple_shape->size() > 0) {
          TypePtrList typle_list(tuple_shape->size(), tuple_type->dynamic_element_type());
          type = std::make_shared<Tuple>(typle_list);
          MS_LOG(DEBUG) << "Replace type from:" << kernel_tensor->GetType() << " to:" << type->ToString();
        }
      }
      abstract = abstract::MakeAbstract(shape, type);
    }
    MS_EXCEPTION_IF_NULL(abstract);
    MS_LOG(DEBUG) << "Infer parameter by abstract:" << abstract->ToString();
    if (!abstract->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Set abstract:" << abstract->ToString() << " for input node:" << input_node->DebugString()
        << " kernel tensor:" << kernel_tensor->ToString();
      input_node->set_abstract(abstract);
      continue;
    }
    MS_LOG(DEBUG) << "Sequence abstract:" << abstract->ToString();
    auto new_abstract = abstract->Clone();
    MS_EXCEPTION_IF_NULL(new_abstract);
    auto seq_abstract = new_abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abstract);
    seq_abstract->set_dynamic_len(true);
    // Dynamic len element is used to check if the sequence is dynamic len.
    if (!seq_abstract->elements().empty() && seq_abstract->elements()[0] != nullptr) {
      seq_abstract->set_dynamic_len_element_abs(seq_abstract->elements()[0]->Clone());
    }
    MS_EXCEPTION_IF_NULL(kernel_tensor->device_address());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set abstract:" << seq_abstract->ToString() << " for input node:" << input_node->DebugString()
      << " kernel tensor:" << kernel_tensor->ToString();
    input_node->set_abstract(seq_abstract);
  }
}
}  // namespace
namespace {
void ClearAttrForGraph(const KernelGraphPtr &graph, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node_pair : graph->front_backend_anf_map()) {
    MS_EXCEPTION_IF_NULL(node_pair.second);
    if (!node_pair.second->isa<CNode>()) {
      continue;
    }
    MS_LOG(DEBUG) << "Check for node:" << node_pair.second->DebugString() << " attr name:" << attr_name;
    const auto &cnode = node_pair.second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
      MS_LOG(DEBUG) << "Erase flag for node:" << node_pair.second->DebugString() << " attr name:" << attr_name;
      common::AnfAlgo::EraseNodeAttr(attr_name, cnode);
    }
  }
}

void PrepareValueNode(const AnfNodePtr &node, KernelTensor *kernel_tensor) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (!node->isa<ValueNode>()) {
    return;
  }
  const auto &value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &node_value = value_node->value();
  if (node_value != nullptr && (node_value->isa<None>() || node_value->isa<Type>())) {
    MS_LOG(DEBUG) << "No need to prepare data for None or type value node:" << node->DebugString();
    return;
  }
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->GetDeviceType(), device_tensor->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  if (device_tensor->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get())) {
      MS_LOG(EXCEPTION) << "Failed to allocate memory for device tensor store:" << device_tensor;
    }
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Device address:" << device_tensor << " allocate ptr:" << device_tensor->GetPtr()
      << " for value node:" << node->DebugString();
  } else {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Device address:" << device_tensor << " already has ptr:" << device_tensor->GetPtr()
      << " for value node:" << node->DebugString();
  }
  device_context->device_res_manager_->SyncAllStreams();
  if (!device_context->device_res_manager_->Copy(device_tensor->GetMutablePtr(), kernel_tensor->GetValuePtr(),
                                                 kernel_tensor->size(), device::CopyType::kH2D, kDefaultStreamIndex)) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Failed to sync data for value node:" << node->DebugString();
  }

  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Device address:" << device_tensor << " ptr:" << device_tensor->GetPtr()
    << " for value node:" << node->DebugString();
}

void PersisitValueNode(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Value node size:" << graph->graph_value_nodes().size() << " for graph:" << graph->ToString();
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
      MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString()
                   << " for graph:" << graph->ToString();
      continue;
    }
    auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(old_kernel_tensor);
    auto device_tensor = old_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    MS_EXCEPTION_IF_NULL(front_node);
    device_tensor->SetNodeIndex(value_node, 0);
    DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), old_kernel_tensor);
    PrepareValueNode(value_node, old_kernel_tensor.get());
    MS_LOG(DEBUG) << "Add device tensor store:" << device_tensor << " node:" << front_node->DebugString()
                  << " graph:" << graph->ToString();

    // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
    if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceType()) == nullptr) {
      MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                   << ", type:" << device_context->GetDeviceType() << " dtype:" << old_kernel_tensor->dtype_id()
                   << " current device address:" << device_tensor << " in value node:" << value_node->DebugString();

      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {value_node, 0}, nullptr, device_tensor->GetSize(), kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()),
        old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(),
        device::GetDeviceNameByType(device_context->device_context_key().device_type_),
        device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(device_tensor->stream_id());
      auto other_type_device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(other_type_device_tensor);
      other_type_device_tensor->SetNodeIndex(value_node, 0);
      other_type_device_tensor->set_from_persistent_mem(true);
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
      DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), kernel_tensor);
      MS_LOG(DEBUG) << "Add device tensor store:" << other_type_device_tensor << " node:" << front_node->DebugString()
                    << " graph:" << graph->ToString();
      PrepareValueNode(value_node, kernel_tensor.get());
    }
  }
  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
    if (real_device_context == device_context) {
      continue;
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Handle device tensor store for backoff kernel:"
                                         << kernel->fullname_with_scope() << " in graph:" << graph->ToString();
    MS_EXCEPTION_IF_NULL(real_device_context);
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      if (input_node == nullptr || !input_node->isa<ValueNode>()) {
        continue;
      }
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
      MS_EXCEPTION_IF_NULL(front_node);
      if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), real_device_context->GetDeviceType()) != nullptr ||
          !AnfAlgo::OutputAddrExist(input_node, 0, false)) {
        MS_LOG(DEBUG) << "Failed to get device tensor in value node:" << input_node->DebugString()
                      << " has device address:" << AnfAlgo::OutputAddrExist(input_node, 0, false)
                      << " real device type:" << real_device_context->GetDeviceType() << " fetch device address:"
                      << DeviceTensorStore::GetInstance().Fetch(front_node.get(), real_device_context->GetDeviceType());
        continue;
      }
      auto node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(node_kernel_tensor);
      auto device_tensor = node_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->DebugString()
                   << ", type:" << real_device_context->GetDeviceType()
                   << " node device tensor:" << device_tensor->ToString();
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {input_node, 0}, nullptr, device_tensor->GetSize(),
        kernel::GetFormatFromEnumToStr(node_kernel_tensor->format()), node_kernel_tensor->dtype_id(),
        node_kernel_tensor->GetShapeVector(),
        device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
        real_device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(device_tensor->stream_id());
      auto other_type_device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(other_type_device_tensor);
      other_type_device_tensor->SetNodeIndex(input_node, 0);
      other_type_device_tensor->set_from_persistent_mem(true);
      MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString()
                    << " for value node:" << front_node->DebugString();
      DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), kernel_tensor);
      MS_LOG(DEBUG) << "Add device tensor store:" << kernel_tensor << " node:" << front_node->DebugString()
                    << " graph:" << graph->ToString() << ", kernel tensor: " << kernel_tensor;
      PrepareValueNode(input_node, kernel_tensor.get());
    }
  }
}
}  // namespace

KernelGraphPtr AnyTypeKernelActor::CompileRealKernelGraph(OpContext<KernelTensor> *const context) {
  try {
    std::lock_guard<std::mutex> lock(instance_lock_);
    InferParameterAbstractForModelGraph(model_graph_, input_kernel_tensors_, any_type_parameter_indexes_);
    ClearAttrForGraph(model_graph_, kAttrInputIsDynamicShape);
    ClearAttrForGraph(model_graph_, kAttrOutputIsDynamicShape);
    model_graph_->InferType();
    const auto &return_node = model_graph_->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    if (!return_node->isa<CNode>()) {
      MS_LOG_WITH_NODE(EXCEPTION, return_node)
        << "Invalid return node:" << return_node->DebugString() << " for graph:" << model_graph_->ToString();
    }
    const auto &return_cnode = return_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(return_cnode);
    if (return_cnode->size() <= 1) {
      MS_LOG_WITH_NODE(EXCEPTION, return_cnode)
        << "Invalid return cnode:" << return_cnode->DebugString() << " for graph:" << model_graph_->ToString();
    }
    if (device_contexts().empty() || device_contexts()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device context for actor:" << GetAID();
    }
    AnfNodePtrList inputs{};
    AnfNodePtrList outputs{return_cnode->input(1)};
    auto io_nodes = std::make_pair(inputs, outputs);
    device_contexts()[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    auto new_graph = compile_func_(BuildSegmentByGraph(model_graph_), io_nodes, device_contexts()[0]);
    MS_EXCEPTION_IF_NULL(new_graph);
    MS_LOG(DEBUG) << "Compile graph:" << new_graph->ToString() << " for:" << model_graph_->ToString();
    for (const auto &node_pair : new_graph->front_backend_anf_map()) {
      MS_EXCEPTION_IF_NULL(node_pair.first);
      if (!node_pair.first->isa<CNode>()) {
        continue;
      }
      MS_LOG(DEBUG) << "Check for node:" << node_pair.first->DebugString();
      const auto &cnode = node_pair.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->HasAttr(kAttrReplaceRealKernelInBackend)) {
        MS_LOG(DEBUG) << "Erase flag for node:" << node_pair.first->DebugString();
        cnode->EraseAttr(kAttrReplaceRealKernelInBackend);
      }
    }
    PersisitValueNode(new_graph, device_contexts()[0]);
    return new_graph;
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    MS_LOG(ERROR) << "Failed to compile graph:" << model_graph_->ToString() << " in actor:" << GetAID()
                  << " error info:" << e.what();
    return nullptr;
  }
}

void ReorderInputDeviceTensor(std::vector<KernelTensorPtr> *kernel_tensors, const KernelGraphPtr &model_graph,
                              const KernelGraphPtr &real_graph, OpContext<KernelTensor> *const context,
                              const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(kernel_tensors);
  MS_EXCEPTION_IF_NULL(model_graph);
  MS_EXCEPTION_IF_NULL(real_graph);
  MS_LOG(DEBUG) << "Reorder input for model graph:" << model_graph->ToString()
                << " real graph:" << real_graph->ToString() << " in actor:" << actor_name;
  std::vector<KernelTensorPtr> reorder_kernel_tensors(kernel_tensors->size(), nullptr);
  if (kernel_tensors->size() != model_graph->input_nodes().size() ||
      model_graph->input_nodes().size() != real_graph->input_nodes().size()) {
    std::stringstream ofs;
    ofs << "Invalid input kernel tensor size:" << kernel_tensors->size() << " model graph:" << model_graph->ToString()
        << " parameter size:" << model_graph->input_nodes().size() << " real graph" << real_graph->ToString()
        << " parameter size:" << real_graph->input_nodes().size() << " for actor:" << actor_name;
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), ofs.str());
  }
  size_t input_size = kernel_tensors->size();
  std::unordered_map<AnfNodePtr, KernelTensorPtr> node_to_kernel_tensors;
  for (size_t i = 0; i < input_size; ++i) {
    node_to_kernel_tensors[model_graph->input_nodes()[i]] = (*kernel_tensors)[i];
  }
  for (size_t i = 0; i < input_size; ++i) {
    MS_EXCEPTION_IF_NULL(real_graph->input_nodes()[i]);
    const auto &front_node = real_graph->GetFrontAnfByBackendAnf(real_graph->input_nodes()[i]);
    if (front_node == nullptr || node_to_kernel_tensors.find(front_node) == node_to_kernel_tensors.end()) {
      std::stringstream ofs;
      ofs << "Failed to get front parameter for backend parameter:" << real_graph->input_nodes()[i]->DebugString()
          << " index:" << i << " in graph:" << real_graph->ToString() << " model graph:" << model_graph->ToString()
          << " for actor:" << actor_name;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), ofs.str());
    }
    reorder_kernel_tensors[i] = node_to_kernel_tensors[front_node];
  }
  (*kernel_tensors).swap(reorder_kernel_tensors);
}

void AnyTypeKernelActor::PrepareRunContext(OpContext<KernelTensor> *const context) {
  const auto &data_type = GenerateIDForGraph(input_kernel_tensors_, any_type_parameter_indexes_);
  if (data_type == "FAILED") {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  GetAID().Name() + " failed to generate id.");
  }
  if (data_type == current_data_type_) {
    ReorderInputDeviceTensor(&input_kernel_tensors_, model_graph_, graph_, context, GetAID().Name());
    return;
  }
  current_data_type_ = data_type;
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    const auto &new_graph = CompileRealKernelGraph(context);
    if (new_graph == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    GetAID().Name() + " compile graph failed.");
    }
    if (new_graph->input_nodes().size() != model_graph_->input_nodes().size()) {
      MS_LOG(EXCEPTION) << "Invalid input node num:" << new_graph->input_nodes().size()
                        << " in graph:" << new_graph->ToString() << " for model graph:" << model_graph_->ToString()
                        << " input num:" << model_graph_->input_nodes().size() << " for actor:" << GetAID();
    }
    real_graphs_[current_data_type_] = new_graph;
  }
  MS_EXCEPTION_IF_NULL(real_graphs_[current_data_type_]);
  graph_ = real_graphs_[current_data_type_];
  MS_LOG(DEBUG) << "Set graph:" << graph_->ToString() << " for actor:" << GetAID();
  ClearElements(context);
  MS_LOG(DEBUG) << "Clear element end and start build and link for actor:" << GetAID();
  BuildAndLinkKernelActors();
  return;
}

void AnyTypeKernelActor::ClearElements(OpContext<KernelTensor> *const context) {
  memory_alloc_list_.clear();
  kernel_actors_insert_event_.clear();
  param_node_to_input_idx_.clear();
  cnode_to_kernel_actor_.clear();
  kernel_actors_.clear();
  kernel_input_to_graph_input_indices_.clear();
  kernel_input_to_actor_output_indices_.clear();
  MS_EXCEPTION_IF_NULL(model_graph_);
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &model_output = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(model_graph_->output());
  const auto &real_output = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(graph_->output());
  if (model_output.size() != real_output.size()) {
    MS_LOG(EXCEPTION) << "Invalid model output size:" << model_output.size()
                      << " and real output size:" << real_output.size() << " for actor:" << GetAID();
  }
  std::map<KernelWithIndex, KernelWithIndex> model_to_real_outputs;
  for (size_t i = 0; i < model_output.size(); ++i) {
    MS_EXCEPTION_IF_NULL(model_output[i].first);
    MS_EXCEPTION_IF_NULL(real_output[i].first);
    model_to_real_outputs[model_output[i]] = real_output[i];
    model_to_real_outputs[common::AnfAlgo::VisitKernelWithReturnType(model_output[i].first, model_output[i].second)] =
      real_output[i];
  }
  if (model_output_data_nodes_.size() != model_output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output data node size:" << model_output_data_nodes_.size()
                      << " and arrow size:" << model_output_data_arrows_.size() << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "node size:" << model_output_data_nodes_.size() << " for actor:" << GetAID();
  for (size_t i = 0; i < model_output_data_nodes_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(model_output_data_nodes_[i]);
    MS_EXCEPTION_IF_NULL(model_output_data_arrows_[i]);
    const auto &iter =
      model_to_real_outputs.find({model_output_data_nodes_[i], model_output_data_arrows_[i]->from_output_index_});
    if (iter == model_to_real_outputs.end()) {
      MS_LOG(EXCEPTION) << "Failed to get real output by model output:" << model_output_data_nodes_[i]->DebugString()
                        << " model graph:" << model_graph_->ToString() << " real graph:" << graph_->ToString();
    }
    output_data_nodes_[i] = iter->second.first;
    output_data_arrows_[i]->from_output_index_ = SizeToInt(iter->second.second);
    MS_LOG(DEBUG) << "Update output node:" << iter->second.first->fullname_with_scope()
                  << " index:" << iter->second.second << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "Reorder Input device tensor";
  if (graph_->input_nodes().size() != model_graph_->input_nodes().size()) {
    MS_LOG(EXCEPTION) << "Invalid model input size:" << model_graph_->input_nodes().size()
                      << " and real input size:" << graph_->input_nodes().size() << " for actor:" << GetAID();
  }
  std::vector<KernelTensorPtr> new_input_kernel_tensors;
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &backend_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(backend_node);
    const auto &front_node = graph_->GetFrontAnfByBackendAnf(backend_node);
    if (front_node == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        GraphExecutionStrategy::kPipeline, (*context),
        "Failed to get front node by backend node:" + backend_node->DebugString() +
          " in real graph:" + graph_->ToString());
    }
    MS_LOG(DEBUG) << "output data front node:" << front_node->DebugString();
    const auto &front_parameters = model_graph_->input_nodes();
    const auto &iter = find(front_parameters.begin(), front_parameters.end(), front_node);
    if (iter == front_parameters.end()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        GraphExecutionStrategy::kPipeline, (*context),
        "Failed to find index by backend parameter:" + backend_node->DebugString() +
          " front node:" + front_node->DebugString());
    }
    size_t position = LongToSize(iter - front_parameters.begin());
    if (position >= input_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid input position:" + std::to_string(position) +
                                                      " total size:" + std::to_string(input_kernel_tensors_.size()) +
                                                      " for actor:" + GetAID().Name());
    }
    new_input_kernel_tensors.emplace_back(input_kernel_tensors_[position]);
    MS_LOG(DEBUG) << "Add model index:" << position << " real input index:" << i
                  << " kernel tensor:" << input_kernel_tensors_[position] << " for actor:" << GetAID();
  }
  input_kernel_tensors_.swap(new_input_kernel_tensors);
  MS_LOG(DEBUG) << "Clear element end";
}

void AnyTypeKernelActor::Run(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(model_graph_);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Any type kernel actor:" << GetAID() << " run.";
  FetchInputDeviceTensor(context);
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  PrepareRunContext(context);
  try {
    if (!LaunchAllKernels(context, false)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Launch kernels by execution order failed for graph: " +
                                                      graph_->ToString() + " error info:" + context->error_info_);
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    MS_LOG(WARNING) << "Catch error:" << e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), e.what());
  }
  EraseInput(context);
  PostRun(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Any type kernel actor:" << GetAID() << " end run.";
}

void AnyTypeKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph());
  MS_LOG(DEBUG) << "actor:" << GetAID() << " init";
  SuperKernelActor::Init();
  memory_alloc_list_.clear();
  for (size_t i = 0; i < graph()->input_nodes().size(); ++i) {
    const auto &input = graph()->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &abs = input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractAny>()) {
      any_type_parameter_indexes_.emplace_back(i);
      MS_LOG(DEBUG) << "Add any type parameter index:" << i << " by parameter:" << input->DebugString()
                    << " for actor:" << GetAID();
    }
  }
  model_graph_ = graph();
  model_output_data_nodes_ = output_data_nodes_;
  for (const auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    model_output_data_arrows_.emplace_back(
      std::make_shared<DataArrow>(data_arrow->from_output_index_, data_arrow->to_op_id_, data_arrow->to_input_index_));
  }
  extern_device_tensor_store_keys_.swap(device_tensor_store_keys_);
  extern_parameter_indexs_.swap(parameter_indexs_);
}

void AnyTypeKernelActor::UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                                          const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);
  if (!output_node->isa<CNode>()) {
    return;
  }
  const auto &node_with_index = common::AnfAlgo::VisitKernelWithReturnType(output_node, data_arrow->from_output_index_);
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  if (!AnfAlgo::OutputAddrExist(node_with_index.first, node_with_index.second, false)) {
    std::stringstream error_info;
    error_info << "Failed to get output device address for:" << node_with_index.first->DebugString()
               << " index:" << node_with_index.second << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  if (node_with_index.first->isa<Parameter>()) {
    auto iter = std::find(graph_->input_nodes().begin(), graph_->input_nodes().end(), node_with_index.first);
    if (iter != graph_->input_nodes().end()) {
      output_data->data_ = input_kernel_tensors_[iter - graph_->input_nodes().begin()];
      output_data->data_->IncreaseNewRefCount();
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Set output address:" << output_data->data_ << " to output data, output node:" << output_node->DebugString()
        << " output index:" << data_arrow->from_output_index_ << " real node:" << node_with_index.first->DebugString()
        << " index:" << node_with_index.second << " in actor:" << GetAID();
      return;
    }
  }
  output_data->data_ = AnfAlgo::GetOutputKernelTensor(node_with_index.first, node_with_index.second, false);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Set output address:" << output_data->data_ << " to output data, output node:" << output_node->DebugString()
    << " output index:" << data_arrow->from_output_index_ << " real node:" << node_with_index.first->DebugString()
    << " index:" << node_with_index.second << " in actor:" << GetAID();
}
}  // namespace runtime
}  // namespace mindspore
