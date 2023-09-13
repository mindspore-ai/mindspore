/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

#include <memory>
#include <stack>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "mindspore/core/ops/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/framework_utils.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "include/common/profiler.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_engine.h"

namespace mindspore {
namespace opt::dynamic_shape {
InfPyHandler cpp_infer_py_handler_{nullptr};
void set_cpp_infer_py_handler(const InfPyHandler &infer_handler) { cpp_infer_py_handler_ = infer_handler; }
namespace {
constexpr int64_t kInvalidShape = -2;

void InferShapeForNopNode(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!common::AnfAlgo::IsNopNode(input_node)) {
    MS_LOG(INFO) << "Input node is not a nop node, no need infer.";
    return;
  }
  if (!common::AnfAlgo::IsNeedSkipNopOpExecution(input_node)) {
    MS_LOG(INFO) << "The Nop node need execution, no need the InferShapeForNopNode.";
    return;
  }
  MS_LOG(INFO) << "Infer shape for nop node.";
  std::stack<AnfNodePtr> nop_road;
  nop_road.push(input_node);

  auto in_node = input_node;
  while (true) {
    auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(in_node, 0);
    in_node = input_node_with_idx.first;
    MS_EXCEPTION_IF_NULL(in_node);
    if (common::AnfAlgo::IsNopNode(in_node)) {
      nop_road.push(in_node);
    } else {
      break;
    }
  }

  while (!nop_road.empty()) {
    auto nop_node = nop_road.top();
    MS_EXCEPTION_IF_NULL(nop_node);
    AnfAlgo::InferShape(nop_node->cast<CNodePtr>());
    nop_road.pop();
  }
}

TypeId GetSequenceType(const abstract::AbstractSequencePtr &seq_abs) {
  MS_EXCEPTION_IF_NULL(seq_abs);
  auto elems = seq_abs->elements();
  MS_EXCEPTION_IF_CHECK_FAIL(elems.size() >= 1, "Element size is less than 1.");
  MS_EXCEPTION_IF_NULL(elems[0]);
  if (!elems[0]->isa<abstract::AbstractScalar>() && !elems[0]->isa<abstract::AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "The 0'th element of sequence must be a scalar, but got:" << seq_abs->ToString();
  }

  auto fixed_type = (elems[0]->isa<abstract::AbstractScalar>()
                       ? elems[0]->BuildType()->type_id()
                       : elems[0]->cast<abstract::AbstractTensorPtr>()->element()->BuildType()->type_id());
  for (size_t i = 1; i < elems.size(); i++) {
    MS_EXCEPTION_IF_NULL(elems[i]);
    if (!elems[i]->isa<abstract::AbstractScalar>() && !elems[i]->isa<abstract::AbstractTensor>()) {
      MS_LOG(EXCEPTION) << "The " << i << "'th element of sequence must be a scalar, but got:" << elems[i]->ToString();
    }
    MS_EXCEPTION_IF_NULL(elems[i]->BuildType());
    auto follow_type = (elems[i]->isa<abstract::AbstractScalar>()
                          ? elems[i]->BuildType()->type_id()
                          : elems[i]->cast<abstract::AbstractTensorPtr>()->element()->BuildType()->type_id());
    if (fixed_type != follow_type) {
      MS_LOG(EXCEPTION) << "Different type found between 0'th element[Type: " << fixed_type << "] and " << i
                        << "'th element[Type: " << follow_type << "]";
    }
  }
  return fixed_type;
}

tensor::TensorPtr CreateTensorMem(const std::pair<AnfNodePtr, size_t> &input_node_with_index) {
  auto real_input = input_node_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input);
  auto real_input_index = input_node_with_index.second;
  auto abs = real_input->abstract();
  MS_EXCEPTION_IF_NULL(abs);

  ShapeVector shape;
  TypeId type;
  if (abs->isa<abstract::AbstractScalar>()) {
    shape = {1};
    MS_EXCEPTION_IF_NULL(abs->BuildType());
    type = abs->BuildType()->type_id();
  } else if (AnfAlgo::IsRealSquenceOutput(real_input)) {
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    auto elem_num = seq_abs->size();
    if (elem_num == 0) {
      MS_LOG(DEBUG) << "Empty sequence for node:" << real_input->fullname_with_scope();
      return std::make_shared<tensor::Tensor>(TypeId::kNumberTypeInt64, ShapeVector({0}));
    }
    type = GetSequenceType(seq_abs);
    shape = {SizeToLong(elem_num)};
  } else if (abs->isa<abstract::AbstractTensor>() || abs->isa<abstract::AbstractSequence>()) {
    shape = trans::GetRuntimePaddingShape(real_input, real_input_index);
    if (real_input->isa<ValueNode>()) {
      // the type of ValueNode in KernelInfo is kTypeUnknown
      type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);
    } else {
      type = AnfAlgo::GetOutputDeviceDataType(real_input, real_input_index);
      if (type == TypeId::kTypeUnknown) {
        type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "For node:" << real_input->fullname_with_scope() << ", abstract(" << abs->ToString()
                      << ") is invalid.";
  }

  MS_LOG(DEBUG) << "Create tensor by node:" << input_node_with_index.first->DebugString()
                << " index:" << input_node_with_index.second << " type:" << type << " shape:" << shape
                << " abstract:" << abs->ToString();
  return std::make_shared<tensor::Tensor>(type, shape);
}

tensor::TensorPtr GetDependValueTensor(const AnfNodePtr &node, size_t i,
                                       const std::pair<AnfNodePtr, size_t> &input_node_with_index, bool skip_nop_node,
                                       void *args) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_node_with_index.first);
  if (IsPrimitiveCNode(node, prim::kPrimPyExecute) && input_node_with_index.first->isa<ValueNode>()) {
    const auto &value_node = input_node_with_index.first->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      return value->cast<tensor::TensorPtr>();
    } else if (value->isa<Scalar>()) {
      return ScalarToTensor(value->cast<ScalarPtr>());
    }
  }
  auto depended_value = CreateTensorMem(input_node_with_index);
  MS_EXCEPTION_IF_NULL(depended_value);
  // First use the data of args.
  if (args != nullptr) {
    auto input_device_address = reinterpret_cast<std::vector<device::DeviceAddress *> *>(args);
    MS_EXCEPTION_IF_NULL(input_device_address);
    if (i < input_device_address->size() && input_device_address->at(i) != nullptr) {
      uint64_t start_time = 0;
      PROFILER_START(start_time);
      auto addr = reinterpret_cast<device::DeviceAddress *>(input_device_address->at(i));
      MS_EXCEPTION_IF_NULL(addr);
      auto node_idx = addr->node_index();
      auto user_data = addr->user_data();
      if (user_data != nullptr && user_data->has(kernel::PyExecuteOutputUserData::key)) {
        auto addr_node = node_idx.first.lock();
        MS_EXCEPTION_IF_NULL(addr_node);
        auto out_addr = AnfAlgo::GetMutableOutputAddr(addr_node, node_idx.second, skip_nop_node);
        depended_value->set_device_address(out_addr, false);
        return depended_value;
      }
      MS_LOG(DEBUG) << "Get depend value tensor for node:" << node->DebugString() << " input index:" << i
                    << " input node:" << input_node_with_index.first->DebugString() << " index"
                    << input_node_with_index.second << " node addr:" << input_node_with_index.first
                    << " device_address:" << input_device_address->at(i)
                    << " type id:" << input_device_address->at(i)->type_id();
      depended_value->data_sync_directly(input_device_address->at(i));
      PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferDataSync,
                   node->fullname_with_scope(), true);
      return depended_value;
    }
    MS_LOG(WARNING) << "There is no valid data for " << i << " input of " << node->DebugString() << ", "
                    << node->fullname_with_scope();
  }

  // Second use the device address of node as fault-tolerant.
  auto output_addr =
    AnfAlgo::GetMutableOutputAddr(input_node_with_index.first, input_node_with_index.second, skip_nop_node);
  MS_EXCEPTION_IF_NULL(output_addr);
  if (output_addr != nullptr && output_addr->IsPtrValid()) {
    // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
    // address size will be wrong in the dynamic shape scenario.
    depended_value->set_device_address(output_addr, false);
    uint64_t start_time = 0;
    PROFILER_START(start_time);
    // PyExecute using the data of user_data instead of address, so don't need to sync data form device./
    if (IsPrimitiveCNode(input_node_with_index.first, prim::kPrimPyExecute)) {
      MS_LOG(DEBUG) << "The input node is " << input_node_with_index.first->ToString()
                    << ", use user data instead of address.";
      return depended_value;
    }
    MS_LOG(DEBUG) << "Get depend value tensor for node:" << node->DebugString() << " input index:" << i
                  << " input node:" << input_node_with_index.first->DebugString() << " index"
                  << input_node_with_index.second << " node addr:" << input_node_with_index.first
                  << " sync for device tensor:" << output_addr;
    depended_value->data_sync();
    PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferDataSync,
                 node->fullname_with_scope(), true);
    return depended_value;
  }

  MS_LOG(EXCEPTION) << "There is no valid data for " << i << " input of " << node->DebugString() << ", "
                    << node->fullname_with_scope();
}

tensor::TensorPtr GetDependValueTensor(const std::vector<device::DeviceAddressPtr> &device_address_list,
                                       const std::vector<tensor::TensorPtr> &input_tensors, size_t index) {
  if (index >= input_tensors.size()) {
    MS_LOG(EXCEPTION) << "Input index: " << index << "is large than the input tensor's size " << input_tensors.size();
  }

  if (input_tensors[index] != nullptr) {
    return input_tensors[index];
  }

  if (index >= device_address_list.size()) {
    MS_LOG(EXCEPTION) << "Input index: " << index << "is large than the input device addresses's size "
                      << device_address_list.size();
  }

  auto output_addr = device_address_list[index];
  if (output_addr != nullptr && output_addr->IsPtrValid()) {
    auto type = output_addr->type_id();
    auto shape = output_addr->host_shape();
    auto tensor = std::make_shared<tensor::Tensor>(type, shape);
    tensor->set_device_address(output_addr, false);
    uint64_t start_time = 0;
    PROFILER_START(start_time);
    tensor->data_sync();
    PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferDataSync,
                 runtime::kDefaultOpName, true);
    return tensor;
  }

  MS_LOG(EXCEPTION) << "There is no valid data for depend value";
}

abstract::AbstractBasePtr MakeNewAbstractByScalar(const tensor::TensorPtr &depended_value) {
  abstract::AbstractBasePtr new_abs;
  MS_EXCEPTION_IF_NULL(depended_value);
  MS_EXCEPTION_IF_NULL(depended_value->Dtype());
  auto type = depended_value->Dtype()->type_id();
  if (type == kNumberTypeInt32) {
    auto tensor_data = reinterpret_cast<int32_t *>(depended_value->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
  } else if (type == kNumberTypeInt64) {
    auto tensor_data = reinterpret_cast<int64_t *>(depended_value->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
  } else if (type == kNumberTypeFloat32) {
    auto tensor_data = reinterpret_cast<float *>(depended_value->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
  } else if (type == kNumberTypeFloat64) {
    auto tensor_data = reinterpret_cast<double *>(depended_value->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
  } else if (type == kNumberTypeBool) {
    auto tensor_data = reinterpret_cast<bool *>(depended_value->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported type: " << type;
  }
  return new_abs;
}

template <typename T>
abstract::AbstractBasePtrList MakeElemsByTensorValue(void *data, size_t size) {
  MS_EXCEPTION_IF_NULL(data);
  T *tensor_data = static_cast<T *>(data);
  AbstractBasePtrList elems;
  for (size_t i = 0; i < size; i++) {
    auto scalar = std::make_shared<abstract::AbstractScalar>(tensor_data[i]);
    (void)elems.emplace_back(scalar);
  }
  return elems;
}

abstract::AbstractBasePtr MakeNewAbstractBySequence(const tensor::TensorPtr &depended_value,
                                                    const abstract::AbstractBasePtr &input_abs) {
  abstract::AbstractBasePtr new_abs;
  MS_EXCEPTION_IF_NULL(depended_value);
  MS_EXCEPTION_IF_NULL(depended_value->Dtype());
  MS_EXCEPTION_IF_NULL(input_abs);
  auto type = depended_value->Dtype()->type_id();
  AbstractBasePtrList elems;
  switch (type) {
    case kNumberTypeInt32: {
      elems = MakeElemsByTensorValue<int32_t>(depended_value->data_c(), depended_value->DataSize());
      break;
    }
    case kNumberTypeInt64: {
      elems = MakeElemsByTensorValue<int64_t>(depended_value->data_c(), depended_value->DataSize());
      break;
    }
    case kNumberTypeFloat32: {
      elems = MakeElemsByTensorValue<float>(depended_value->data_c(), depended_value->DataSize());
      break;
    }
    case kNumberTypeFloat64: {
      elems = MakeElemsByTensorValue<double>(depended_value->data_c(), depended_value->DataSize());
      break;
    }
    case kNumberTypeBool: {
      elems = MakeElemsByTensorValue<bool>(depended_value->data_c(), depended_value->DataSize());
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "Unsupported type: " << type;
    }
  }
  if (input_abs->isa<abstract::AbstractTuple>()) {
    new_abs = std::make_shared<abstract::AbstractTuple>(elems);
  } else if (input_abs->isa<abstract::AbstractList>()) {
    new_abs = std::make_shared<abstract::AbstractList>(elems);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported abstract type:" << input_abs->ToString();
  }
  MS_EXCEPTION_IF_NULL(new_abs);
  new_abs->set_value(depended_value);
  return new_abs;
}

abstract::AbstractBasePtr MakeNewAbstract(const AnfNodePtr &input, const tensor::TensorPtr &depended_value,
                                          const size_t &input_index) {
  MS_EXCEPTION_IF_NULL(input);
  auto abs = input->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  abstract::AbstractBasePtr new_abs;
  if (abs->isa<abstract::AbstractTensor>()) {
    new_abs = abs->Clone();
    MS_EXCEPTION_IF_NULL(new_abs);
    new_abs->set_value(depended_value);
  } else if (abs->isa<abstract::AbstractScalar>()) {
    new_abs = MakeNewAbstractByScalar(depended_value);
  } else if (AnfAlgo::IsRealSquenceOutput(input)) {
    new_abs = MakeNewAbstractBySequence(depended_value, abs);
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto abstract_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abstract_seq);
    MS_EXCEPTION_IF_CHECK_FAIL((input_index < abstract_seq->elements().size()), "Index is out of range.");
    new_abs = abstract_seq->elements()[input_index]->Clone();
    MS_EXCEPTION_IF_NULL(new_abs);
    new_abs->set_value(depended_value);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported abstract type:" << abs->ToString();
  }
  // Set user data for PyExecute infer.
  if (input->has_user_data<kernel::PyExecuteOutputUserData>()) {
    const auto &output_data = input->user_data<kernel::PyExecuteOutputUserData>();
    MS_EXCEPTION_IF_NULL(new_abs);
    new_abs->set_user_data<kernel::PyExecuteOutputUserData>(output_data);
  }
  auto depend_addr = depended_value->device_address();
  if (depend_addr != nullptr) {
    MS_LOG(DEBUG) << "Input node : " << input->DebugString() << ",use user_data instead of device address";
    auto user_data = depend_addr->user_data();
    if (user_data != nullptr) {
      new_abs->set_user_data<kernel::PyExecuteOutputUserData>(
        user_data->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key));
    }
  }
  return new_abs;
}

bool InferShapeForGraphWithSymbolEngine(const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                                        const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto output = func_graph->output();
  auto symbol_engine = GetValue<SymbolEnginePtr>(func_graph->get_attr(kAttrSymbolEngine));
  if (!symbol_engine->Infer(args_spec_list)) {
    MS_LOG(INFO) << "Infer failed by symbol engine. node " << cnode->fullname_with_scope();
    return false;
  }
  auto out_shapes = symbol_engine->QueryShape(output);
  BaseShapePtr abs_shape = nullptr;
  if (out_shapes.size() == 1) {
    abs_shape = std::make_shared<abstract::Shape>(out_shapes[0]);
  } else {
    abstract::BaseShapePtrList shape_list;
    shape_list.reserve(out_shapes.size());
    (void)std::transform(out_shapes.cbegin(), out_shapes.cend(), std::back_insert_iterator(shape_list),
                         [](const ShapeVector &s) { return std::make_shared<abstract::Shape>(s); });
    abs_shape = std::make_shared<abstract::TupleShape>(shape_list);
  }
  auto output_abs = output->abstract();
  output_abs->set_shape(abs_shape);
  cnode->set_abstract(output_abs);
  return true;
}

void InferShapeForGraph(const CNodePtr &cnode, const FuncGraphPtr &func_graph,
                        const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (func_graph->has_attr(kAttrSymbolEngine)) {
    MS_LOG(DEBUG) << "SymbolEngine is found in funcgraph " << func_graph->ToString();
    if (InferShapeForGraphWithSymbolEngine(cnode, func_graph, args_spec_list)) {
      return;
    }
  }
  MS_LOG(DEBUG) << "InferShape by primitive for funcgraph " << func_graph->ToString();
  if (args_spec_list.size() != func_graph->parameters().size()) {
    MS_LOG(EXCEPTION)
      << "The args_spec_list size should be the same as that of func_graph parameters, but get args_spec_list: "
      << args_spec_list.size() << " vs func_graph parameters: " << func_graph->parameters().size();
  }
  for (size_t i = 0; i < args_spec_list.size(); i++) {
    func_graph->parameters()[i]->set_abstract(args_spec_list[i]->Clone());
  }
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>() || !IsValueNode<Primitive>(node->cast<CNodePtr>()->input(0))) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimReturn)) {
      auto cnode_primitive = GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(cnode_primitive);
      auto prim_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(prim_cnode);

      AbstractBasePtrList cnode_args_spec_list;

      for (size_t i = 1; i < prim_cnode->size(); i++) {
        auto input_node = prim_cnode->input(i);
        MS_EXCEPTION_IF_NULL(input_node);
        (void)cnode_args_spec_list.emplace_back(input_node->abstract()->Clone());
      }
      opt::CppInferShape(cnode_primitive, cnode_args_spec_list, prim_cnode);
    } else {
      auto return_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(return_cnode);
      cnode->set_abstract(return_cnode->input(1)->abstract()->Clone());
    }
  }
  return;
}

TypeId GetTypeIDByAbstract(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return TypeId::kTypeUnknown;
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    auto type = abstract->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    return type->type_id();
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &tensor_abs = abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    MS_EXCEPTION_IF_NULL(tensor_abs->element());
    return GetTypeIDByAbstract(tensor_abs->element());
  } else if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &seq_abs = abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->elements().empty() || seq_abs->elements()[0] == nullptr) {
      return TypeId::kTypeUnknown;
    }
    return GetTypeIDByAbstract(seq_abs->elements()[0]);
  }
  MS_LOG(INFO) << "Invalid abstract:" << abstract->ToString();
  return TypeId::kTypeUnknown;
}

void InferShapeForPrimitive(const CNodePtr &cnode, const PrimitivePtr &primitive,
                            const AbstractBasePtrList &args_spec_list, bool has_py_execute_data) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!has_py_execute_data && !IsPrimitiveCNode(cnode, prim::kPrimPyExecute)) {
    // Pynative mode is rely on the origin abstract of cnode, so cannot modify the abstract inplace, clone from old
    // abstract instead.
    opt::CppInferShape(primitive, args_spec_list, cnode);
  } else {
    if (cpp_infer_py_handler_ == nullptr) {
      // If run without Python.
      MS_LOG(WARNING) << "\'cpp_infer_py_handler_\' should not be null.";
      const auto &abs = opt::CppInferShapeAndType(primitive, args_spec_list);
      MS_LOG(DEBUG) << "The abstract of " << cnode->fullname_with_scope() << " changes from " << cnode->abstract()
                    << " to " << abs;
      cnode->set_abstract(abs);
      return;
    }
    const auto &abs = cpp_infer_py_handler_(cnode, primitive, args_spec_list);
    cnode->set_abstract(abs);
    const auto &kernel_info_device = cnode->kernel_info();
    if (kernel_info_device != nullptr) {
      auto kernel_info = static_cast<device::KernelInfo *>(kernel_info_device);
      auto real_type_id = GetTypeIDByAbstract(abs);
      if (kernel_info != nullptr && kernel_info->GetMutableSelectKernelBuildInfo() != nullptr &&
          real_type_id != TypeId::kTypeUnknown) {
        auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
        build_info->SetOutputDeviceType(real_type_id, 0);
        MS_LOG(DEBUG) << "Set output type:" << real_type_id << " for kernel:" << cnode->fullname_with_scope();
      }
    }
  }
}

void InferShape(const CNodePtr &cnode, std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(DEBUG) << "InferShape start, node:" << cnode->fullname_with_scope();
  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(cnode);

  depend_tensor_map->clear();
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  bool skip_nop_node = !context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  bool has_py_execute_data = false;
  kernel::PyExecuteOutputUserDataPtr list_user_data = nullptr;
  std::vector<size_t> list_start_index;
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i, false);
    auto real_input = input_node_with_index.first;
    auto real_input_index = input_node_with_index.second;

    MS_EXCEPTION_IF_NULL(real_input);
    if (skip_nop_node) {
      InferShapeForNopNode(real_input);
    }

    if (depend_list.find(i) != depend_list.end()) {
      auto depended_value = GetDependValueTensor(cnode, i, input_node_with_index, skip_nop_node, args);
      auto ret2 = depend_tensor_map->try_emplace(i, depended_value);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }

      auto updated_abs = MakeNewAbstract(real_input, depended_value, real_input_index);
      MS_EXCEPTION_IF_NULL(updated_abs);
      MS_EXCEPTION_IF_NULL(real_input);
      MS_EXCEPTION_IF_NULL(real_input->abstract());
      if (updated_abs->has_user_data<kernel::PyExecuteOutputUserData>()) {
        has_py_execute_data = true;
        if (IsPrimitiveCNode(real_input, prim::kPrimPyExecute) &&
            real_input->abstract()->isa<abstract::AbstractSequence>()) {
          auto updated_abs_user_data = updated_abs->user_data<kernel::PyExecuteOutputUserData>();
          if (list_user_data == nullptr || list_user_data != updated_abs_user_data) {
            list_start_index.push_back(i);
            list_user_data = updated_abs_user_data;
          }
        }
      }
      (void)args_spec_list.emplace_back(updated_abs);
    } else {
      auto abs = real_input->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      MS_LOG(DEBUG) << "Real input node:" << real_input->DebugString() << " abs:" << abs->ToString()
                    << " index:" << real_input_index;
      if (abs->isa<abstract::AbstractSequence>() && !AnfAlgo::IsRealSquenceOutput(real_input)) {
        auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
        MS_EXCEPTION_IF_NULL(abs_seq);
        MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abs_seq->elements().size()), "Index is out of range.");
        auto abs_index = abs_seq->elements()[real_input_index];
        (void)args_spec_list.emplace_back(abs_index);
      } else {
        (void)args_spec_list.emplace_back(abs);
      }
    }
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  if (auto primitive = GetValueNode<PrimitivePtr>(inputs[0])) {
    MS_EXCEPTION_IF_NULL(primitive);
    (void)primitive->AddAttr(kAttrListStartIndex, MakeValue(list_start_index));
    InferShapeForPrimitive(cnode, primitive, args_spec_list, has_py_execute_data);
  } else if (auto func_graph = GetValueNode<FuncGraphPtr>(inputs[0])) {
    InferShapeForGraph(cnode, func_graph, args_spec_list);
  } else {
    MS_LOG(EXCEPTION) << "The first input of the cnode should be either a primitive or a function graph, but get: "
                      << inputs[0]->fullname_with_scope();
  }
  MS_LOG(DEBUG) << "InferShape end, node:" << cnode->fullname_with_scope();
}

inline bool IsCpuKernelMod(kernel::KernelModType kernel_mod_type) {
  return kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod ||
         kernel_mod_type == kernel::KernelModType::DeprecatedNativeCpuKernelMod;
}
}  // namespace
bool IsRealCNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n)) {
    CNodePtr cnode = utils::cast<CNodePtr>(n);
    return AnfUtils::IsRealKernel(cnode);
  }
  return false;
}

AnfNodePtr GenInferNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto infer_node = AnfUtils::NewInferActorNode([cnode](void *args) { InferOp(cnode, args); }, cnode);
  MS_EXCEPTION_IF_NULL(infer_node);
  infer_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return infer_node;
}

AnfNodePtr GenInitNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  AnfUtils::CustomActorCallback actor_func = [kernel_mod, cnode](void *) {
    auto args = cnode->user_data<kernel::KernelArgs>();
    if (args == nullptr) {
      args = std::make_shared<kernel::KernelArgs>();
    }
    MS_LOG(DEBUG) << "resize for cnode:" << cnode->fullname_with_scope();
    if (kernel_mod->Resize(args->inputs, args->outputs, args->depend_tensor_map) ==
        static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
      MS_LOG(EXCEPTION) << "Node " << cnode->fullname_with_scope() << " Resize failed.";
    }
  };

  auto init_node = AnfUtils::NewInitActorNode(actor_func, cnode);
  MS_EXCEPTION_IF_NULL(init_node);
  init_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return init_node;
}

void InferOp(const CNodePtr &cnode, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  kernel::KernelArgs kernel_args;
  MS_LOG(DEBUG) << "infer shape for node:" << cnode->fullname_with_scope();
  InferShape(cnode, &kernel_args.depend_tensor_map, args);
  auto kernel_mod_type = kernel_mod->GetKernelModType();
  auto update = kernel::AbstractArgsFromCNode(cnode);
  update.depend_tensor_map = std::move(kernel_args.depend_tensor_map);
  kernel::SetInputsByDependMap(update.depend_tensor_map, &update.inputs, IsCpuKernelMod(kernel_mod_type));
  kernel::SetArgsToCNode(cnode, update);
}

void InferShape(std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map,
                const pynative::ExecuteKernelInfo &execute_kernel,
                const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(execute_kernel.kernel_);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(DEBUG) << "InferShape start, node:" << execute_kernel.kernel_->fullname_with_scope();
  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(execute_kernel.kernel_);

  depend_tensor_map->clear();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto primitive = execute_kernel.primitive_;
  auto input_size = execute_kernel.inputs_device_address_.size();
  for (size_t i = 0; i < input_size; i++) {
    auto input_address = execute_kernel.inputs_device_address_[i];
    MS_EXCEPTION_IF_NULL(input_address);
    if (depend_list.find(i) != depend_list.end()) {
      auto depended_value = GetDependValueTensor(execute_kernel.inputs_device_address_, input_tensors, i);
      MS_EXCEPTION_IF_NULL(depended_value);
      auto ret2 = depend_tensor_map->try_emplace(i, depended_value);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }
      (void)args_spec_list.emplace_back(depended_value->ToAbstract());
    } else {
      auto abs =
        std::make_shared<abstract::AbstractTensor>(TypeIdToType(input_address->type_id()), input_address->host_shape());
      (void)args_spec_list.emplace_back(abs);
    }
  }

  CppInferShape(primitive, args_spec_list, execute_kernel.kernel_);
}

void UpdateOutputDeviceShape(const std::vector<device::DeviceAddressPtr> &output_device_address_list,
                             const AbstractBasePtr &abstract) {
  auto output_num = output_device_address_list.size();
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractTuple>()) {
    auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    for (size_t i = 0; i < output_num; ++i) {
      auto real_abs = abstract_tuple->elements()[i];
      MS_EXCEPTION_IF_NULL(real_abs);
      MS_EXCEPTION_IF_NULL(output_device_address_list[i]);
      output_device_address_list[i]->set_host_shape(BaseShapeToShape(real_abs->BuildShape()));
    }
  } else {
    MS_EXCEPTION_IF_NULL(output_device_address_list[0]);
    output_device_address_list[0]->set_host_shape(BaseShapeToShape(abstract->BuildShape()));
  }
}

kernel::KernelArgs GetKernelArgsForNode(const CNodePtr &cnode, const pynative::ExecuteKernelInfo &execute_kernel,
                                        const kernel::KernelArgs &kernel_args) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  UpdateOutputDeviceShape(execute_kernel.outputs_device_address_, cnode->abstract());

  auto kernel_mod_type = kernel_mod->GetKernelModType();
  auto update = kernel::AbstractArgsFromDeviceAddress(kernel_mod, execute_kernel.inputs_device_address_,
                                                      execute_kernel.outputs_device_address_, cnode->abstract());
  update.depend_tensor_map = kernel_args.depend_tensor_map;
  kernel::SetInputsByDependMap(update.depend_tensor_map, &update.inputs, IsCpuKernelMod(kernel_mod_type));
  return update;
}

kernel::KernelArgs InferOp(const CNodePtr &cnode, const pynative::ExecuteKernelInfo &execute_kernel,
                           const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  kernel::KernelArgs kernel_args;
  InferShape(&kernel_args.depend_tensor_map, execute_kernel, input_tensors);

  return GetKernelArgsForNode(cnode, execute_kernel, kernel_args);
}

kernel::KernelArgs SetOpArgs(const CNodePtr &cnode, const pynative::ExecuteKernelInfo &execute_kernel,
                             const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel::KernelArgs kernel_args;
  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(cnode);
  auto input_size = execute_kernel.inputs_device_address_.size();
  for (size_t i = 0; i < input_size; i++) {
    if (depend_list.find(i) != depend_list.end()) {
      auto depended_value = GetDependValueTensor(execute_kernel.inputs_device_address_, input_tensors, i);
      auto ret2 = kernel_args.depend_tensor_map.try_emplace(i, depended_value);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }
    }
  }

  return GetKernelArgsForNode(cnode, execute_kernel, kernel_args);
}

CustomActorNodeManager &CustomActorNodeManager::Instance() {
  static CustomActorNodeManager instance{};
  return instance;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
