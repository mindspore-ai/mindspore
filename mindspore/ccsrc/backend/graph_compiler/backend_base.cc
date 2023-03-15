/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/graph_compiler/backend_base.h"

#include <algorithm>
#include <vector>
#include <map>

#include "backend/graph_compiler/transform.h"
#include "ir/anf.h"
#include "utils/log_adapter.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/pynative/graph_adapter.h"
#include "distributed/recovery/recovery_context.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/common/utils/callbacks.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/ps_context.h"
#endif

namespace mindspore {
namespace compile {
bool Backend::GetCond(const BaseRef &c, bool *value) {
  mindspore::ScopedLongRunning long_running;
  return BaseRefToBool(c, value);
}
bool Backend::GetIndex(const BaseRef &c, int64_t *value) { return BaseRefToInt(utils::cast<ValuePtr>(c), value); }

Backend::Backend(const std::string &name) : name_(name), is_multi_graph_sink_(false) {
  MS_LOG(DEBUG) << "Select backend:" << name;
  convert_fn_ = MsVmConvert;
}

namespace {
using Tensor = tensor::Tensor;
bool CheckValidTensorTuple(const std::vector<ValuePtr> &values) {
  if (values.empty() || values[0] == nullptr || (!values[0]->isa<tensor::Tensor>())) {
    return false;
  }
  const auto &const_tensor = values[0]->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(const_tensor);
  const auto &const_shape = const_tensor->shape();
  const auto &const_type_id = const_tensor->data_type();
  size_t const_size = const_tensor->Size();
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] == nullptr || (!values[i]->isa<Tensor>())) {
      MS_LOG(ERROR) << "Invalid value:" << (values[i] == nullptr ? "nullptr" : values[i]->ToString()) << " index:" << i
                    << " in value tuple";
      return false;
    }
    const auto &tensor = values[i]->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &shape = tensor->shape();
    const auto &type_id = tensor->data_type();
    size_t size = tensor->Size();
    if (shape != const_shape || type_id != const_type_id || size != const_size) {
      return false;
    }
  }
  return true;
}

// Return a new tensor with type like single_value.
void SetScalarToTensor(const std::vector<ValuePtr> &values, const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &tensor_type_id = tensor->data_type();
  const auto dst_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(dst_ptr);
  MS_LOG(DEBUG) << "Set scalar tuple to tensor, dst size:" << tensor->data().nbytes();
  for (size_t i = 0; i < values.size(); ++i) {
    // Check mem size.
    if (SizeToLong(abstract::TypeIdSize(tensor_type_id) * (i + 1)) > tensor->data().nbytes()) {
      MS_LOG(EXCEPTION) << "Value size:" << values.size() << " type:" << tensor_type_id
                        << " out of range:" << tensor->data().nbytes();
    }
    const auto &value = values[i];
    MS_EXCEPTION_IF_NULL(value);
    // Check value type.
    if (value->type()->type_id() != tensor_type_id) {
      MS_LOG(EXCEPTION) << "Invalid value type:" << value->type()->type_id() << " for value:" << value->ToString()
                        << " dst type:" << tensor_type_id;
    }
    if (tensor_type_id == TypeId::kNumberTypeInt8) {
      (reinterpret_cast<int8_t *>(dst_ptr))[i] = GetValue<int8_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt16) {
      (reinterpret_cast<int16_t *>(dst_ptr))[i] = GetValue<int16_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt32 || tensor_type_id == kNumberTypeInt) {
      (reinterpret_cast<int32_t *>(dst_ptr))[i] = GetValue<int32_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeInt64) {
      (reinterpret_cast<int64_t *>(dst_ptr))[i] = GetValue<int64_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeBool) {
      (reinterpret_cast<bool *>(dst_ptr))[i] = GetValue<bool>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeFloat32 || tensor_type_id == TypeId::kNumberTypeFloat) {
      (reinterpret_cast<float *>(dst_ptr))[i] = GetValue<float>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeFloat64) {
      (reinterpret_cast<double *>(dst_ptr))[i] = GetValue<double>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt8) {
      (reinterpret_cast<uint8_t *>(dst_ptr))[i] = GetValue<uint8_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt16) {
      (reinterpret_cast<uint16_t *>(dst_ptr))[i] = GetValue<uint16_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt || tensor_type_id == TypeId::kNumberTypeUInt32) {
      (reinterpret_cast<uint32_t *>(dst_ptr))[i] = GetValue<uint32_t>(value);
    } else if (tensor_type_id == TypeId::kNumberTypeUInt64) {
      (reinterpret_cast<uint64_t *>(dst_ptr))[i] = GetValue<uint64_t>(value);
    } else {
      MS_LOG(EXCEPTION) << "Invalid tuple type:" << tensor_type_id << " for scalar to tensor.";
    }
  }
}

// In dynamic sequence, since the number of members is not determined in compile time, the entire sequence needs
// to be placed in single tensor, and the shape of the tuple needs to be recorded in the tensor, so that the shape
// of the tensor can be accurately restored during the dynamic shape derivation process in runtime.
TensorPtr SequenceToTensor(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Invalid sequence value:" << value->ToString();
  }

  const auto &sequence_value = value->cast<ValueSequencePtr>();
  const auto &values = sequence_value->value();
  if (values.empty()) {
    auto tensor = std::make_shared<tensor::Tensor>();
    abstract::BaseShapePtr base_shape = nullptr;
    if (value->isa<ValueTuple>()) {
      base_shape = std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList());
    } else {
      base_shape = std::make_shared<abstract::ListShape>(abstract::BaseShapePtrList());
    }
    tensor->set_base_shape(base_shape);
    return tensor;
  }
  if (values[0] == nullptr || ((!values[0]->isa<Scalar>()) && (!values[0]->isa<Tensor>()))) {
    MS_LOG(WARNING) << "Empty sequence in sequence value:" << value->ToString();
    return std::make_shared<tensor::Tensor>();
  }

  ShapeVector shape_vector{SizeToLong(values.size())};
  if (values[0]->isa<Tensor>()) {
    MS_LOG(DEBUG) << "Check dynamic tuple tensor";
    if (!CheckValidTensorTuple(values)) {
      MS_LOG(EXCEPTION) << "Invalid dynamic sequence tuple:" << value->ToString();
    }
    const auto &tensor = values[0]->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t size = tensor->Size();
    const auto &type_id = tensor->data_type();
    auto single_shape_vector = tensor->shape();
    const auto &single_shape = std::make_shared<abstract::Shape>(single_shape_vector);
    shape_vector.insert(shape_vector.end(), single_shape_vector.begin(), single_shape_vector.end());
    const auto &shape = std::make_shared<abstract::Shape>(shape_vector);
    auto new_tensor = std::make_shared<tensor::Tensor>(type_id, shape_vector);
    MS_EXCEPTION_IF_NULL(new_tensor);
    const auto dst_ptr = new_tensor->data_c();
    MS_EXCEPTION_IF_NULL(dst_ptr);
    MS_LOG(DEBUG) << "Copy start, dst size:" << new_tensor->data().nbytes();
    for (size_t i = 0; i < values.size(); ++i) {
      const auto &sub_value = values[i];
      MS_EXCEPTION_IF_NULL(sub_value);
      const auto &src_tensor = sub_value->cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(src_tensor);
      MS_EXCEPTION_IF_NULL(src_tensor->data_c());
      auto ret = memcpy_s((reinterpret_cast<char *>(dst_ptr)) + i * size,
                          static_cast<size_t>(new_tensor->data().nbytes()), src_tensor->data_c(), size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret;
      }
    }
    const auto &element_shapes = std::vector<abstract::BaseShapePtr>(values.size(), single_shape);
    new_tensor->set_base_shape(std::make_shared<abstract::TupleShape>(element_shapes));
    MS_LOG(DEBUG) << "merge tensor from:" << value->ToString() << " to:" << new_tensor->ToString() << " tensor addr"
                  << new_tensor;
    return new_tensor;
  }

  // Create the tensor.
  auto tensor = std::make_shared<tensor::Tensor>(values[0]->type()->type_id(), shape_vector);
  MS_EXCEPTION_IF_NULL(tensor);
  SetScalarToTensor(values, tensor);
  // Build the tuple shape and set into tensor.
  const auto &element_shape = std::make_shared<abstract::Shape>(ShapeVector({1}));
  const auto &element_shapes = std::vector<abstract::BaseShapePtr>(values.size(), element_shape);
  tensor->set_base_shape(std::make_shared<abstract::TupleShape>(element_shapes));
  return tensor;
}
}  // namespace

void PushInputTensor(const BaseRef &arg, std::vector<tensor::TensorPtr> *inputs, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(inputs);
  if (node != nullptr && node->abstract() != nullptr && common::AnfAlgo::IsDynamicSequence(node)) {
    MS_LOG(DEBUG) << "node:" << node->fullname_with_scope() << " abs:" << node->abstract()->ToString();
    if (!utils::isa<ValuePtr>(arg)) {
      MS_LOG(EXCEPTION) << "Invalid input for dynamic sequence node:" << node->DebugString();
    }
    auto value = utils::cast<ValuePtr>(arg);
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION) << "Invalid value:" << value->ToString()
                        << " for dynamic sequence node:" << node->DebugString();
    }
    const auto &tensor = SequenceToTensor(value);
    inputs->push_back(tensor);
    return;
  }

  if (utils::isa<tensor::TensorPtr>(arg)) {
    auto value = utils::cast<tensor::TensorPtr>(arg);
    inputs->push_back(value);
  } else if (utils::isa<ValuePtr>(arg)) {
    auto value = utils::cast<ValuePtr>(arg);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      auto value_tuple = value->cast<ValueSequencePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      auto tuple_value = value_tuple->value();
      (void)std::transform(tuple_value.begin(), tuple_value.end(), std::back_inserter(*inputs),
                           [](const ValuePtr &v) { return v->cast<tensor::TensorPtr>(); });
    } else if (value->isa<Scalar>()) {
      tensor::TensorPtr scalar_tensor = ScalarToTensor(value->cast<ScalarPtr>());
      inputs->push_back(scalar_tensor);
    } else if (value->isa<Monad>()) {
      // If value is a monad, replace it with an unused tensor.
      inputs->push_back(std::make_shared<tensor::Tensor>(int64_t(0), kBool));
    } else {
      inputs->push_back(value->cast<tensor::TensorPtr>());
    }
  } else if (utils::isa<PyObjectRef>(arg)) {
    auto value = utils::cast<PyObjectRef>(arg).object_;
    inputs->push_back(py::cast<tensor::TensorPtr>(value));
  } else if (utils::isa<VectorRefPtr>(arg)) {
    const auto &args_new = utils::cast<VectorRef>(arg);
    for (const auto &v : args_new) {
      PushInputTensor(v, inputs);
    }
  } else {
    MS_LOG(WARNING) << "Invalid input type.";
  }
}

namespace {
void FlattenValue(const BaseRef &arg, ValuePtrList *flatted_value) {
  MS_EXCEPTION_IF_NULL(flatted_value);
  if (utils::isa<ValueSequencePtr>(arg)) {
    auto value_sequence = utils::cast<ValueSequencePtr>(arg);
    MS_EXCEPTION_IF_NULL(value_sequence);
    auto sequence_value = value_sequence->value();
    for (auto &value : sequence_value) {
      MS_EXCEPTION_IF_NULL(value);
      if (value->isa<tensor::Tensor>()) {
        (void)flatted_value->emplace_back(value);
      } else if (value->isa<Scalar>()) {
        auto scalar = value->cast<ScalarPtr>();
        MS_EXCEPTION_IF_NULL(scalar);
        (void)flatted_value->emplace_back(ScalarToTensor(scalar));
      } else {
        FlattenValue(value, flatted_value);
      }
    }
  } else if (utils::isa<ValueDictionaryPtr>(arg)) {
    auto value_dict = utils::cast<ValueDictionaryPtr>(arg);
    MS_EXCEPTION_IF_NULL(value_dict);
    auto dict_value = value_dict->value();
    for (auto &iter : dict_value) {
      auto value = iter.second;
      MS_EXCEPTION_IF_NULL(value);
      if (value->isa<tensor::Tensor>()) {
        (void)flatted_value->emplace_back(value);
      } else if (value->isa<Scalar>()) {
        auto scalar = value->cast<ScalarPtr>();
        MS_EXCEPTION_IF_NULL(scalar);
        (void)flatted_value->emplace_back(ScalarToTensor(scalar));
      } else {
        FlattenValue(value, flatted_value);
      }
    }
  } else if (utils::isa<tensor::COOTensorPtr>(arg)) {
    auto coo_tensor = utils::cast<tensor::COOTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(coo_tensor);
    for (size_t i = 0; i < coo_tensor->GetTensorLength(); ++i) {
      (void)flatted_value->emplace_back(coo_tensor->GetTensorAt(i));
    }
  } else if (utils::isa<tensor::CSRTensorPtr>(arg)) {
    auto csr_tensor = utils::cast<tensor::CSRTensorPtr>(arg);
    MS_EXCEPTION_IF_NULL(csr_tensor);
    for (size_t i = 0; i < csr_tensor->GetTensorLength(); ++i) {
      (void)flatted_value->emplace_back(csr_tensor->GetTensorAt(i));
    }
  } else {
    MS_LOG(EXCEPTION) << "The value input to flatten should only contains be sequence or dictionary, but it is "
                      << arg.ToString();
  }
}

// Insert the front_node related tensor in the input_tensor.
void PushTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  if (iter == parameters.end()) {
    (void)((*input_tensors).emplace_back(nullptr));
    return;
  }
  auto position = iter - parameters.begin();

  // If the node is dynamic sequence all the element in tuple should be placed in single tensor.
  PushInputTensor(args[position], input_tensors, front_node);
}

void PushTupleTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                     size_t index, std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  const size_t position = iter - parameters.begin();
  // If the parameter is not found in the parameters of the root graph, it means that it is the input of the subgraph,
  // and there is no need to input a tensor.
  if (position >= args.size()) {
    MS_LOG(DEBUG) << "Position out of args range, position value is " << position << " and args size is " << args.size()
                  << ".";
    (void)input_tensors->emplace_back(nullptr);
    return;
  }
  ValuePtrList flatted_value_tuple_value;
  FlattenValue(args[position], &flatted_value_tuple_value);
  if (index >= flatted_value_tuple_value.size()) {
    MS_LOG(EXCEPTION) << "Index out of flatted_value_tuple_value range, index value is " << index
                      << " and flatted_value_tuple_value size is " << flatted_value_tuple_value.size() << ".";
  }
  auto input = flatted_value_tuple_value[index];
  MS_EXCEPTION_IF_NULL(input);
  auto tensor_input = input->cast<tensor::TensorPtr>();
  input_tensors->push_back(tensor_input);
}
}  // namespace

std::vector<std::vector<tensor::TensorPtr>> GetRunGraphInputs(const GraphCompilerInfo &graph_compiler_info,
                                                              const VectorRef &args) {
  const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;
  std::vector<std::vector<tensor::TensorPtr>> input_tensor_lists;
  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    std::vector<tensor::TensorPtr> input_tensors;
    MS_EXCEPTION_IF_NULL(kernel_graph);
    for (const auto &input_node : kernel_graph->input_nodes()) {
      auto element_pair = kernel_graph->GetElementInTupleBackendFrontIndexMap(input_node);
      if (element_pair.first) {
        PushTupleTensor(args, origin_parameters, element_pair.first, element_pair.second, &input_tensors);
      } else {
        const auto &front_node = kernel_graph->GetFrontAnfByBackendAnf(input_node);
        PushTensor(args, origin_parameters, front_node, &input_tensors);
      }
    }
    (void)input_tensor_lists.emplace_back(input_tensors);
  }

  // Input tensors of the control node.
  std::vector<tensor::TensorPtr> input_tensors;
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  // Get inputs of control node which come from the host actor.
  const auto &control_node_parameters = graph_compiler_info.control_node_parser_->control_node_parameters();
  for (const auto &parameter_with_index : control_node_parameters) {
    const auto &parameter = parameter_with_index.first;
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abs = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractSequence>() && (!common::AnfAlgo::IsDynamicSequence(parameter))) {
      MS_LOG(DEBUG) << "Fetch input tensor for tuple parameter:" << parameter->DebugString() << " in control flow.";
      PushTupleTensor(args, origin_parameters, parameter, parameter_with_index.second, &input_tensors);
    } else {
      PushTensor(args, origin_parameters, parameter, &input_tensors);
    }
  }
  (void)input_tensor_lists.emplace_back(input_tensors);

  return input_tensor_lists;
}

MindRTBackendBase::MindRTBackendBase(const std::string &backend_name, const std::string &device_name,
                                     uint32_t device_id)
    : Backend(backend_name), device_name_(device_name), device_id_(device_id) {
  root_graph_ = nullptr;
  auto ms_context = MsContext::GetInstance();
  const bool pynative_mode = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  auto &cut_list = pynative_mode ? GetControlOps() : GetMsNonlinearOps();

  graph_partition_ = std::make_shared<GraphPartition>(cut_list, backend_name);
  graph_compiler_ = std::make_shared<GraphCompiler>();

  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name, device_id});
  device_context->Initialize();
  device_id_ = device_context->device_context_key().device_id_;
#ifdef ENABLE_DEBUGGER
  SetDebuggerInit();
#endif
  runtime::GraphScheduler::GetInstance().Initialize();
}

void MindRTBackendBase::ProcessNotSupportCnode(const FuncGraphPtr &func_graph,
                                               const mindspore::device::DeviceType &old_target,
                                               const mindspore::device::DeviceType &new_target) const {
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if (!common::AnfAlgo::HasNodeAttr(kAttrNotSupportOpForDevice, cnode)) {
      continue;
    }

    auto not_support_device = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrNotSupportOpForDevice);
    if (device::GetDeviceTypeByName(not_support_device) != old_target) {
      continue;
    }

    common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(device::GetDeviceNameByType(new_target)), node);
  }
}

namespace {
int64_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(output_index_value_node);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto idx = value->isa<Int64Imm>() ? GetValue<int64_t>(value) : GetValue<int>(value);
  return idx;
}

KernelWithIndex VisitRealNodeWithNestLevel(const AnfNodePtr &anf_node, size_t index, size_t *nest_level) {
  if (!anf_node->isa<CNode>()) {
    return {anf_node, index};
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(cnode) == prim::kTupleGetItem) {
    (*nest_level)++;
    auto real_node_with_index = VisitRealNodeWithNestLevel(common::AnfAlgo::GetTupleGetItemRealInput(cnode),
                                                           common::AnfAlgo::GetTupleGetItemOutIndex(cnode), nest_level);
    auto real_node = real_node_with_index.first;
    auto real_index = real_node_with_index.second;
    if (real_node->isa<CNode>() && common::AnfAlgo::GetCNodeName(real_node) == prim::kMakeTuple) {
      (*nest_level)--;
      auto make_tuple = real_node->cast<CNodePtr>();
      return VisitRealNodeWithNestLevel(make_tuple->input(real_index + 1), index, nest_level);
    }
    return real_node_with_index;
  }
  return common::AnfAlgo::VisitKernelWithReturnType(anf_node, index, false,
                                                    {prim::kPrimMakeTuple, prim::kPrimTupleGetItem});
}

bool NeedConvertToRealTupleGetItem(const CNodePtr &cnode) {
  if (common::AnfAlgo::GetCNodeName(cnode) != prim::kTupleGetItem || cnode->inputs().size() != kTupleGetItemInputSize) {
    return false;
  }
  if (!cnode->input(kInputNodeOutputIndexInTupleGetItem)->isa<ValueNode>() || GetTupleGetItemOutIndex(cnode) < 0) {
    return true;
  }
  size_t nest_level = 0;
  const size_t nest_limit = 1;
  auto real_node = VisitRealNodeWithNestLevel(cnode, 0, &nest_level);
  if (!common::AnfAlgo::IsCallNode(real_node.first) && AnfUtils::IsRealCNodeKernel(real_node.first) &&
      nest_level > nest_limit) {
    return true;
  }
  return false;
}
}  // namespace

const ActorInfo &MindRTBackendBase::CompileGraphs(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Status record: start compile function graph: " << func_graph->ToString();
  PROF_START(compile_func_graph);

  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  UnifyMindIR(root_graph);
  root_graph_ = root_graph;

  // Register a summary callback function, which is called in the final stages of summary.
  graph_compiler_->RegisterSummaryCallBackFunc(callbacks::SummarySaveCallback);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  ms_execution_mode_ = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  func_graph->set_flag(kFlagPyNativeRunInGraph, ms_execution_mode_ == kPynativeMode);

  // Compile root graph.
  graph_id_to_device_context_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  control_nodes_.clear();

  auto jit_level = common::AnfAlgo::GetJitLevel(func_graph);
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_}, jit_level);
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  bool all_support = device_context->PartitionGraph(func_graph);
  if (all_support) {
    auto run_mode = device_context->GetRunMode(func_graph);
    if (run_mode == device::RunMode::kGraphMode && pynative::GraphAdapter::PyNativeEnableTaskSink(func_graph)) {
      auto graph_id = graph_compiler_->CompileWholeGraphForGraphRunMode(func_graph, device_context);
      graph_id_to_device_context_[graph_id] = device_context;
    } else {
      CompileSubGraph(func_graph, device::RunMode::kKernelMode);
    }
  } else {
    ProcessNotSupportCnode(func_graph, device_context->GetDeviceType(), mindspore::device::DeviceType::kCPU);
    CompileSubGraph(func_graph);
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  if ((ms_execution_mode_ == kGraphMode ||
       (ms_execution_mode_ == kPynativeMode && jit_level == "O3" && context_ptr->backend_policy() == "ge")) &&
      ((!graph_compiler_info->graphs_.empty()) || graph_compiler_info->control_nodes_.size() > 1)) {
    // Transform graph to actor DAG, and schedule the actor DAG.
    ParseControlNodes(*graph_compiler_info);
    const auto &actor_set = runtime::GraphScheduler::GetInstance().Transform(*graph_compiler_info);
    runtime::GraphScheduler::GetInstance().Schedule(actor_set);
  }
  const ActorInfo &actor_info = graph_compiler_info->name_;
  (void)actor_to_graph_compiler_info_.emplace(graph_compiler_info->name_, std::move(graph_compiler_info));
  PROF_END(compile_func_graph);

  MS_LOG(INFO) << "Status record: end compile function graph: " << func_graph->ToString()
               << ", produce actor: " << actor_info;
  return actor_info;
}

void MindRTBackendBase::UnifyMindIR(const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(root_graph->manager());
  const std::map<std::string, std::string> kOpListToTupleNames = {{prim::kMakeListNew, prim::kMakeTuple},
                                                                  {prim::kListGetItem, prim::kTupleGetItem},
                                                                  {prim::kListSetItem, prim::kTupleSetItem}};

  FuncGraphSet graphs = root_graph->manager()->func_graphs();
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    auto nodes = TopoSort(graph->get_return());
    for (const auto &node : nodes) {
      if (node == nullptr || (!node->isa<CNode>())) {
        continue;
      }
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      // List name --> tuple name.
      auto iter = kOpListToTupleNames.find(common::AnfAlgo::GetCNodeName(cnode));
      if (iter != kOpListToTupleNames.end()) {
        common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
        cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(iter->second)));
        // Reset full scope name.
        cnode->set_fullname_with_scope("");
        MS_LOG(INFO) << "Rename op from " << iter->first << " to " << iter->second << " for op "
                     << cnode->fullname_with_scope() << ", debug name:" << cnode->DebugString();
      }

      // TupleGetItem --> RealTupleGetItem.
      if (NeedConvertToRealTupleGetItem(cnode)) {
        common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
        cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(prim::kRealTupleGetItem)));
        // Reset full scope name.
        cnode->set_fullname_with_scope("");
        MS_LOG(INFO) << "Rename op from TupleGetItem to RealTupleGetItem for op " << cnode->fullname_with_scope()
                     << ", debug name:" << cnode->DebugString();
      }

      // MakeTuple --> RealMakeTuple
      if (common::AnfAlgo::GetCNodeName(cnode) == prim::kMakeTuple && common::AnfAlgo::IsDynamicSequence(cnode)) {
        common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
        cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(prim::kRealMakeTuple)));
        // Reset full scope name.
        cnode->set_fullname_with_scope("");
        MS_LOG(INFO) << "Rename op from MakeTuple to RealMakeTuple for op " << cnode->fullname_with_scope()
                     << ", debug name:" << cnode->DebugString();
      }
    }
  }
}

void MindRTBackendBase::CompileSubGraph(const FuncGraphPtr &func_graph, device::RunMode run_mode) {
  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  CompileGraph(root_graph, run_mode);

  MS_EXCEPTION_IF_NULL(root_graph->manager());
  FuncGraphSet sub_graphs = root_graph->manager()->func_graphs();
  for (const auto &sub_graph : sub_graphs) {
    if (sub_graph != func_graph && sub_graph != nullptr) {
      CompileGraph(sub_graph, run_mode);
    }
  }
}

void MindRTBackendBase::CompileGraph(const FuncGraphPtr &func_graph, device::RunMode run_mode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_partition_);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  bool contain_multi_target = false;
  // Split graph to segments.
  const auto &segments = graph_partition_->Partition(func_graph, &contain_multi_target);
  MS_LOG(INFO) << "Compile graph: " << func_graph->ToString() << ", Split segments size:" << segments.size();

  // Foreach the segments to compile graph.
  for (const auto &segment : segments) {
    CompileGraph(segment, run_mode);
  }
}

void MindRTBackendBase::CompileGraph(const GraphSegmentPtr &segment, device::RunMode run_mode) {
  MS_EXCEPTION_IF_NULL(segment);
  // Compile the normal nodes, which doesn't contain the cut node.
  if (segment->nodes_.size() == 0) {
    MS_LOG(EXCEPTION) << "The segments size is 0.";
  }
  if (!segment->is_cut_) {
    MS_EXCEPTION_IF_NULL(segment->nodes_[0]);
    MS_LOG(INFO) << "Compile normal segment, the first node: " << segment->nodes_[0]->DebugString();

    // Get the device context.
    const auto &cur_device_name = GetCNodeTarget(segment->nodes_[0]);
    const auto &device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({cur_device_name, device_id_});
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->Initialize();

    // Transform nodes to inputs and outputs.
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = TransformSegmentToAnfGraph(segment->nodes_);

    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    GraphId graph_id;
    if (root_graph_->has_flag(kFlagEnableRunGraphBySingleOp)) {
      graph_id = graph_compiler_->CompileDynamicGraph(segment, outputs, device_context);
    } else {
      graph_id =
        graph_compiler_->CompileGraph(segment, outputs, device_context, run_mode, ms_execution_mode_ == kPynativeMode);
      if (graph_compiler_->Fetch(graph_id)->has_flag(kFlagEnableRunGraphBySingleOp)) {
        root_graph_->set_flag(kFlagEnableRunGraphBySingleOp, true);
      }
    }

    graph_id_to_device_context_[graph_id] = device_context;

    const auto &func_graph = segment->nodes_[0]->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph_to_kernel_graph_ids_.find(func_graph) == func_graph_to_kernel_graph_ids_.end()) {
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>{graph_id});
    } else {
      (void)func_graph_to_kernel_graph_ids_[func_graph].back().emplace_back(graph_id);
    }
  } else {
    // Compile the cut node.
    auto cut_node = segment->nodes_[0];
    MS_EXCEPTION_IF_NULL(cut_node);
    MS_LOG(INFO) << "Compile cut segment, the cut node: " << cut_node->DebugString();
    control_nodes_.push_back(cut_node);
    if (common::AnfAlgo::IsCallNode(cut_node) || common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitchLayer)) {
      const auto &func_graph = cut_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>());
    }
  }
}

namespace {
void TensorValueToVector(const ValuePtr &value, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(outputs);
  if (value->isa<ValueSequence>()) {
    auto value_tuple = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<tensor::Tensor>()) {
        auto tensor = element->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        outputs->emplace_back(tensor);
      } else if (element->isa<Scalar>()) {
        auto scalar = element->cast<ScalarPtr>();
        MS_EXCEPTION_IF_NULL(scalar);
        outputs->emplace_back(ScalarToTensor(scalar));
      } else if (element->isa<ValueSequence>()) {
        VectorRef tuple;
        TensorValueToVector(element, &tuple);
        outputs->emplace_back(tuple);
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    outputs->emplace_back(tensor);
  } else if (value->isa<Scalar>()) {
    auto scalar = value->cast<ScalarPtr>();
    MS_EXCEPTION_IF_NULL(scalar);
    outputs->emplace_back(ScalarToTensor(scalar));
  }
}

bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &graph_output, const VectorRef &args, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(graph_output);
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_output->isa<ValueNode>()) {
    MS_LOG(INFO) << "Graph's output is a constant. No need to execute.";
    VectorRef output_tmp;
    ValuePtr value = GetValueNode(graph_output);
    TensorValueToVector(value, &output_tmp);
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      outputs->emplace_back(output_tmp);
    } else if (value->isa<tensor::Tensor>() || value->isa<Scalar>()) {
      *outputs = output_tmp;
    } else {
      MS_LOG(INFO) << "Graph output is empty!";
    }
    return true;
  }

  if (graph_output->isa<Parameter>()) {
    MS_LOG(INFO) << "Graph's output is a parameter. If all params are inputs, no need to execute.";
    // Find the right parameter as ret_val.
    auto func_graph = graph_output->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params = func_graph->parameters();
    if (args.size() != params.size()) {
      MS_LOG(EXCEPTION) << "Input size " << args.size() << " not equal to graph input size " << params.size();
    }

    auto it = std::find(params.begin(), params.end(), graph_output);
    if (it == params.end()) {
      MS_EXCEPTION(UnknownError) << "When graph output is Parameter, it should be found in graph parameters";
    }
    size_t index = it - params.cbegin();
    if (index >= args.size()) {
      MS_EXCEPTION(UnknownError) << "Index " << index << " equal or larger than args size " << args.size();
    }

    outputs->emplace_back(args[index]);
    return true;
  }
  return false;
}
}  // namespace

void MindRTBackendBase::ConstructOutputs(runtime::ActorSet *actor_set, VectorRef *outputs,
                                         const FuncGraphPtr &root_graph) {
  bool need_contruct_output = !(distributed::recovery::RecoveryContext::GetInstance()->enable_recovery() &&
                                distributed::recovery::RecoveryContext::GetInstance()->need_reset());
  bool is_embedding_cache_server = false;
#if defined(__linux__) && defined(WITH_BACKEND)
  is_embedding_cache_server = ps::PSContext::instance()->cache_enable() && ps::PSContext::instance()->is_server();
#endif
  if (need_contruct_output) {
    // Update device address for output node of graph.
    // Summary processing will use the output device address, so must be after the summary processing.
    if (!is_embedding_cache_server) {
      actor_set->output_actor_->UpdateOutputDeviceAddress();
    }

    // Fetch outputs.
    MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
    auto &output_tensors = actor_set->output_actor_->outputs();
    if (!output_tensors.empty()) {
      size_t output_position = 0;
      std::vector<tensor::TensorPtr> tuple_tensors;
      ConstructOutputs(root_graph->output(), output_tensors, &output_position, outputs, &tuple_tensors);

      // The tensor may be repeated, so it needs to be set null last.
      for (auto &tuple_tensor : tuple_tensors) {
        MS_EXCEPTION_IF_NULL(tuple_tensor);
        tuple_tensor->set_device_address(nullptr);
      }
    }
  }
}

void MindRTBackendBase::RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(root_graph_);
  if (IsGraphOutputValueNodeOrParameter(root_graph_->output(), args, outputs)) {
    return;
  }

  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY)) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return;
  }

  // Open abstract_lock for dynamic_shape
  AnfUtils::OpenAbstractLock();

  MS_LOG(INFO) << "Status record: start run actor: " << actor_info;
  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(actor_info);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the graph compiler info.";
  }
  MS_EXCEPTION_IF_NULL(graph_iter->second);
  const auto &graph_compiler_info = *(graph_iter->second);
  // For pynative and graph mix execution.
  WaitTaskFinish();

  // Run in the pynative mode.
  MS_EXCEPTION_IF_NULL(outputs);
  // There will be more than one kernel graph in heterogeneous scenario in a ms function of PyNative Mode.
  if (ms_execution_mode_ == kPynativeMode) {
    RunGraphByCondition(actor_info, graph_compiler_info, args, outputs);
    return;
  }

  auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
  // Release python gil.
  mindspore::ScopedLongRunning long_running;
  // Run actor DAG.
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(actor_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors);

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  ConstructOutputs(actor_set, outputs, root_graph_);

  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << actor_info;
}

BaseRef MindRTBackendBase::ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                                     const std::vector<tensor::TensorPtr> &output_tensors,
                                                     size_t *output_position,
                                                     std::vector<tensor::TensorPtr> *tuple_tensors) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);

  size_t outputs_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
  if (*output_position + outputs_num > output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position << " need:" << outputs_num
                      << " total:" << output_tensors.size();
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    (*output_position)++;
    return output_tensors[(*output_position) - 1];
  }

  VectorRef outputs;
  const auto &tuple_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  // Dynamic len tuple.
  if (tuple_abstract->dynamic_len()) {
    auto &output_tensor = output_tensors[*output_position];
    MS_EXCEPTION_IF_NULL(output_tensor);
    auto &tensor_shape = output_tensor->base_shape_ptr();
    // Restore the tuple output by the tensor of tuple.
    if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
      ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &outputs,
                                   tuple_tensors);
      (*output_position)++;
      return outputs;
    }
  }

  const auto &sub_abstracts = tuple_abstract->elements();
  for (const auto &sub_abstract : sub_abstracts) {
    MS_EXCEPTION_IF_NULL(sub_abstract);
    outputs.emplace_back(ConstructOutputByAbstract(sub_abstract, output_tensors, output_position, tuple_tensors));
  }
  return outputs;
}

void MindRTBackendBase::ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor,
                                                     const abstract::SequenceShapePtr &tensor_shape, VectorRef *outputs,
                                                     std::vector<tensor::TensorPtr> *tuple_tensors) {
  MS_EXCEPTION_IF_NULL(output_tensor);
  MS_EXCEPTION_IF_NULL(tensor_shape);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tuple_tensors);
  // If outputs an empty sequence return an empty sequence value.
  if (tensor_shape->size() == 0) {
    if (tensor_shape->isa<abstract::TupleShape>()) {
      outputs->emplace_back(std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
    } else {
      outputs->emplace_back(std::make_shared<ValueList>(std::vector<ValuePtr>()));
    }
    return;
  }
  // No need split multi tensors when the tuple size is not greater than 1.
  if (tensor_shape->size() <= 1) {
    outputs->emplace_back(output_tensor);
    return;
  }

  auto tensor_type_id = output_tensor->data_type();
  auto device_tensor = std::dynamic_pointer_cast<device::DeviceAddress>(output_tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto tensor_device_ptr = device_tensor->GetMutablePtr();
  auto tensor_device_size = device_tensor->GetSize();
  MS_EXCEPTION_IF_NULL(tensor_device_ptr);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->device_name(), device_tensor->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  // Split the tensor of tuple to tensors.
  tuple_tensors->emplace_back(output_tensor);
  size_t copy_offset_size = 0;
  for (size_t i = 0; i < tensor_shape->size(); ++i) {
    // Create split tensor.
    auto split_tensor_shape = BaseShapeToShape((*tensor_shape)[i]);
    auto split_tensor_size = SizeOf(split_tensor_shape) * GetTypeByte(TypeIdToType(tensor_type_id));
    auto split_tensor = std::make_shared<tensor::Tensor>(tensor_type_id, split_tensor_shape);
    auto split_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(
      nullptr, split_tensor_size, device_tensor->format(), device_tensor->type_id(), split_tensor_shape);

    // Copy data from origin tensor to the split tensor.
    device::DynamicMemAllocatorDebugInfo::SetDebugInfo("Split tuple outputs", device::AllocatorType::kOther);
    if (!device_context->device_res_manager_->AllocateMemory(split_device_tensor.get())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, kernel name: Split tuple outputs, alloc size: "
                        << split_device_tensor->GetSize() << "B.";
    }
    if (copy_offset_size + split_tensor_size > tensor_device_size) {
      MS_LOG(EXCEPTION) << "The copy size is out of range, copy size:" << split_tensor_size
                        << ", copy offset size:" << copy_offset_size << ", total size:" << tensor_device_size;
    }
    if (!split_device_tensor->SyncDeviceToDevice(split_tensor_shape, split_tensor_size, device_tensor->type_id(),
                                                 AddressOffset(tensor_device_ptr, copy_offset_size),
                                                 device_tensor->format())) {
      MS_LOG(EXCEPTION) << "Sync device to device failed, device type:" << split_device_tensor->GetDeviceType()
                        << ", copy size:" << split_tensor_size << ", output node: Split tuple outputs.";
    }
    copy_offset_size += split_tensor_size;

    // Fill the outputs.
    split_tensor->set_device_address(split_device_tensor);
    outputs->emplace_back(split_tensor);
  }
}

namespace {
bool IsEmptySequence(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                     const size_t *const output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(output_position);
  // When the output node is a valuenode, the position may out of range.
  if (*output_position >= output_tensors.size()) {
    return false;
  }

  if (output_node->abstract() == nullptr || (!output_node->abstract()->isa<abstract::AbstractSequence>())) {
    return false;
  }
  const auto &tuple_abs = output_node->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if ((!tuple_abs->dynamic_len()) && tuple_abs->dynamic_len_element_abs() == nullptr) {
    return false;
  }
  const auto &tensor = output_tensors[*output_position];
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    return false;
  }
  const auto &sequence_shape = tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  return sequence_shape->size() == 0;
}
}  // namespace

void MindRTBackendBase::ConstructOutputs(const AnfNodePtr &output_node,
                                         const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                         VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);
  const PrimitiveSet expand_prims{
    prim::kPrimMakeTuple,
    prim::kPrimMakeCSRTensor,
    prim::kPrimMakeCOOTensor,
    prim::kPrimMakeRowTensor,
  };

  // If outputs an empty sequence return an empty sequence value.
  if (IsEmptySequence(output_node, output_tensors, output_position)) {
    if (output_node->abstract()->isa<abstract::AbstractTuple>()) {
      outputs->emplace_back(std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
    } else {
      outputs->emplace_back(std::make_shared<ValueList>(std::vector<ValuePtr>()));
    }
    ++(*output_position);
    return;
  }

  // The MakeTuple/MakeSaprse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(output_node, expand_prims)) {
    auto make_tuple = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    VectorRef make_tuple_output;
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      ConstructOutputs(make_tuple->input(i), output_tensors, output_position, &make_tuple_output, tuple_tensors);
    }
    outputs->emplace_back(std::move(make_tuple_output));
    return;
  }

  // The depend node need get the real node.
  if (common::AnfAlgo::CheckPrimitiveType(output_node, prim::kPrimDepend)) {
    auto depend_node = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
    ConstructOutputs(depend_node->input(kRealInputIndexInDepend), output_tensors, output_position, outputs,
                     tuple_tensors);
    return;
  }

  auto outputs_num = AnfAlgo::GetOutputElementNum(output_node);
  // The value node uses the value to be output, to avoid the host memory of value free due to value node destruction.
  if (output_node->isa<ValueNode>()) {
    auto value = output_node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      outputs->emplace_back(value);
      (*output_position) += CountValueNum(value->cast<ValueSequencePtr>());
    } else if (outputs_num != 0) {
      outputs->emplace_back(value);
      (*output_position) += outputs_num;
    }
    // The empty value node return the empty VectorRef.
    return;
  }

  if (common::AnfAlgo::IsCallNode(output_node)) {
    auto abstract = output_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    outputs->emplace_back(ConstructOutputByAbstract(abstract, output_tensors, output_position, tuple_tensors));
    return;
  }

  auto &output_abstract = output_node->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  // Wrap output to VectorRef if the output is tuple.
  if (output_abstract->isa<abstract::AbstractSequence>()) {
    VectorRef output_tuple;
    for (size_t i = 0; i < outputs_num; ++i) {
      if (*output_position >= output_tensors.size()) {
        MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position;
      }
      auto &output_tensor = output_tensors[*output_position];
      MS_EXCEPTION_IF_NULL(output_tensor);
      auto &tensor_shape = output_tensor->base_shape_ptr();
      // Restore the tuple output by the tensor of tuple.
      if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
        ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &output_tuple,
                                     tuple_tensors);
      } else {
        output_tuple.emplace_back(output_tensor);
      }
      ++(*output_position);
    }
    outputs->emplace_back(std::move(output_tuple));
  } else {
    for (size_t i = 0; i < outputs_num; ++i) {
      if (*output_position >= output_tensors.size()) {
        MS_LOG(EXCEPTION) << "The output position is out of range: " << *output_position;
      }
      outputs->emplace_back(output_tensors[*output_position]);
      ++(*output_position);
    }
  }
}

#ifdef ENABLE_DEBUGGER
void MindRTBackendBase::SetDebuggerInit() const {
  auto debugger_ = Debugger::GetInstance();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  debugger_->Init(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
}
#endif

std::shared_ptr<GraphCompilerInfo> MindRTBackendBase::ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::string name = "kernel_graph";
  size_t graph_index = 0;
  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    (void)graphs.emplace_back(graph_compiler_->Fetch(graph_id_to_context.first));
    (void)device_contexts.emplace_back(graph_id_to_context.second);
    if (graph_index == 0) {
      (void)name.append("_").append(std::to_string(graph_id_to_context.first));
    } else if (graph_index == graph_id_to_device_context_.size() - 1) {
      (void)name.append("-").append(std::to_string(graph_id_to_context.first));
    }
    ++graph_index;
  }

  auto parser = std::make_shared<ControlNodeParser>();

  runtime::KernelMapPosition outputs_order;
  const auto &root_output =
    common::AnfAlgo::VisitKernelWithReturnType(root_graph->output(), 0, false, {prim::kPrimTupleGetItem}).first;
  size_t position = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(root_output);
  size_t outputs_num = outputs.size();
  for (const auto &output : outputs) {
    if (outputs_order.count(output) == 0) {
      outputs_order[output] = {position++};
    } else {
      (void)outputs_order[output].emplace_back(position++);
    }
  }

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  auto strategy = runtime::GraphExecutionStrategy::kPipeline;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) != kOptimizeO0 ||
      context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    strategy = runtime::GraphExecutionStrategy::kPipelineWithExecutionOrder;
  }
  return std::make_shared<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes_,
                                             root_graph->parameters(), parser, outputs_order, outputs_num, name, false,
                                             strategy);
}

void MindRTBackendBase::ParseControlNodes(const GraphCompilerInfo &graph_compile_info) {
  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_ids : func_graph_to_kernel_graph_ids_) {
    const auto &func_graph = func_graph_to_kernel_graph_ids.first;
    for (const auto &sub_kernel_graphs_ids : func_graph_to_kernel_graph_ids.second) {
      std::vector<KernelGraphPtr> kernel_graphs;
      for (const auto &graph_id : sub_kernel_graphs_ids) {
        const auto &kernel_graph = graph_compiler_->Fetch(graph_id);
        MS_EXCEPTION_IF_NULL(kernel_graph);
        (void)kernel_graphs.emplace_back(kernel_graph);
      }
      (void)func_graph_to_kernel_graphs[func_graph].emplace_back(kernel_graphs);
    }
  }

  graph_compile_info.control_node_parser_->Parse(control_nodes_, graph_compile_info.graphs_,
                                                 graph_compile_info.device_contexts_, root_graph_,
                                                 func_graph_to_kernel_graphs);
}
}  // namespace compile
}  // namespace mindspore
