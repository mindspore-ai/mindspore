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
#include <algorithm>
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
#include "ops/op_def.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "include/common/profiler.h"
#include "ir/anf.h"
#include "ir/functor.h"
#include "backend/operator/ops_backend_infer_function.h"

namespace mindspore {
namespace opt::dynamic_shape {
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

tensor::TensorPtr CreateTensorFromIndexedNode(const std::pair<AnfNodePtr, size_t> &input_node_with_index) {
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

tensor::TensorPtr CreateTensorMem(const std::pair<AnfNodePtr, size_t> &input_node_with_index, const AnfNodePtr &node,
                                  size_t i, void *args) {
  if (node != nullptr && common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPyExecute)) {
    MS_EXCEPTION_IF_NULL(args);
    auto input_list = reinterpret_cast<std::vector<device::DeviceAddress *> *>(args);
    MS_EXCEPTION_IF_NULL(input_list);
    if (i >= input_list->size() || input_list->at(i) == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get device address by input num:" << i << " for node:" << node->DebugString();
    }
    const auto &device_address = input_list->at(i);
    MS_EXCEPTION_IF_NULL(device_address->kernel_tensor());
    MS_LOG(DEBUG) << "input node:" << input_node_with_index.first->DebugString()
                  << " abstract:" << input_node_with_index.first->abstract()->ToString()
                  << " device address:" << device_address << " type id:" << device_address->kernel_tensor()->dtype_id()
                  << " shape vector:" << device_address->kernel_tensor()->GetShapeVector();
    auto type_id = device_address->kernel_tensor()->dtype_id();
    if (device_address->kernel_tensor()->GetType() != nullptr &&
        ((device_address->kernel_tensor()->GetType()->isa<Tuple>() &&
          device_address->kernel_tensor()->GetType()->cast<TuplePtr>()->size() == 0) ||
         (device_address->kernel_tensor()->GetType()->isa<List>() &&
          device_address->kernel_tensor()->GetType()->cast<ListPtr>()->size() == 0))) {
      type_id = TypeId::kNumberTypeInt64;
    }
    return std::make_shared<tensor::Tensor>(type_id, device_address->kernel_tensor()->GetShapeVector());
  }

  return CreateTensorFromIndexedNode(input_node_with_index);
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
  auto depended_value = CreateTensorMem(input_node_with_index, node, i, args);
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

void InferShapeForPrimitive(const CNodePtr &cnode, const PrimitivePtr &primitive,
                            const AbstractBasePtrList &args_spec_list, bool has_py_execute_data) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!has_py_execute_data && !IsPrimitiveCNode(cnode, prim::kPrimPyExecute)) {
    // Pynative mode is rely on the origin abstract of cnode, so cannot modify the abstract inplace, clone from old
    // abstract instead.
    opt::CppInferShape(primitive, args_spec_list, cnode);
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
  } else {
    MS_LOG(EXCEPTION) << "The first input of the cnode should be either a primitive or a function graph, but get: "
                      << inputs[0]->fullname_with_scope();
  }
  MS_LOG(DEBUG) << "InferShape end, node:" << cnode->fullname_with_scope();
}

inline bool IsCpuKernelMod(kernel::KernelModType kernel_mod_type) {
  return kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod;
}
}  // namespace

BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInferInner,
                                     op_name, false);
  auto shape_optional = abstract::InferShapeByFuncImpl(primitive, input_args, false);
  if (shape_optional.has_value()) {
    return shape_optional.value();
  }

  // The old register map for InferShape will be deleted in the future.
  auto infer_impl = abstract::GetBackendPrimitiveInferImpl(primitive);
  if (infer_impl.has_value()) {
    auto infer = infer_impl.value();
    if (infer.IsImplInferShapeAndType()) {
      return infer.InferShape(primitive, input_args);
    }
  }
  MS_LOG(EXCEPTION) << "The InferShape function of [" << op_name << "] is not defined.";
}

void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                             const std::vector<kernel::KernelTensor *> &output_kernel_tensors) {
  MS_EXCEPTION_IF_NULL(base_shape);
  size_t output_num = output_kernel_tensors.size();
  if (output_num > 1) {
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    const auto &shapes = sequence_shape->shape();
    if (shapes.size() != output_num) {
      MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << output_num
                        << ", but got: " << shapes.size();
    }
    for (size_t i = 0; i < output_num; i++) {
      const auto &kernel_tensor = output_kernel_tensors[i];
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->SetShape(shapes[i]);
    }
  } else if (output_num == 1) {
    const auto &kernel_tensor = output_kernel_tensors[0];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto sequence_shape = base_shape->cast<abstract::SequenceShapePtr>();
    if ((kernel_tensor->type_id() != kObjectTypeTuple && kernel_tensor->type_id() != kObjectTypeList) &&
        sequence_shape != nullptr) {
      // For the operator prototype whose output is of type Tuple, the back-end operator is expanded as Tensors, and for
      // single-output scenarios, the InferShape result is TupleShape, and the back-end needs to expand it to
      // TensorShape. For example, the output of the split operator is only a Tensor scene.
      const auto &shapes = sequence_shape->shape();
      if (shapes.size() != 1) {
        MS_LOG(EXCEPTION) << "Invalid SequenceShape, expected elements number: " << 1 << ", but got: " << shapes.size();
      }

      kernel_tensor->SetShape(shapes[0]);
    } else {
      kernel_tensor->SetShape(base_shape);
    }
  }
}

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
    auto inputs = AnfAlgo::GetOrCreateAllInputKernelTensors(cnode);
    auto outputs = AnfAlgo::GetOrCreateAllOutputKernelTensors(cnode);
    if (kernel_mod->Resize(inputs, outputs) == static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
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

CustomActorNodeManager &CustomActorNodeManager::Instance() {
  static CustomActorNodeManager instance{};
  return instance;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
