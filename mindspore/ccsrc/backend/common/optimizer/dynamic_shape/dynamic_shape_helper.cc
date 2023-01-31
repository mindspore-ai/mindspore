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
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"

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

bool InferShapeForDefiniteOutputNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimShape)) {
    return false;
  }
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  if (input_size != 1) {
    MS_LOG(EXCEPTION) << "Node only has one input: " << cnode->fullname_with_scope();
  }
  auto cur_shape = dynamic_cast<mindspore::abstract::Shape *>(cnode->Shape().get())->shape();
  if (std::any_of(cur_shape.begin(), cur_shape.end(), [](int64_t x) { return x == kInvalidShape; })) {
    return false;
  }
  std::vector<int64_t> output_shape = {static_cast<int64_t>(cur_shape.size())};
  mindspore::abstract::BaseShapePtr shape = std::make_shared<mindspore::abstract::Shape>(output_shape);

  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(cnode.get());
  auto abstract = cnode->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  abstract->set_shape(shape);
  return true;
}

TypeId GetSequenceType(const abstract::AbstractSequencePtr &seq_abs) {
  auto elems = seq_abs->elements();
  if (!elems[0]->isa<abstract::AbstractScalar>()) {
    MS_LOG(EXCEPTION) << "The 0'th element of sequence must be a scalar, but got:" << elems[0]->ToString();
  }

  auto fixed_type = elems[0]->BuildType();
  for (size_t i = 1; i < elems.size(); i++) {
    if (!elems[i]->isa<abstract::AbstractScalar>()) {
      MS_LOG(EXCEPTION) << "The " << i << "'th element of sequence must be a scalar, but got:" << elems[i]->ToString();
    }
    auto follow_type = elems[i]->BuildType();
    if (!(fixed_type == follow_type)) {
      MS_LOG(EXCEPTION) << "Different type found between 0'th element[Type: " << fixed_type->ToString() << "] and " << i
                        << "'th element[Type: " << follow_type->ToString() << "]";
    }
  }
  return fixed_type->type_id();
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
    type = abs->BuildType()->type_id();
  } else if (AnfAlgo::IsRealSquenceOutput(real_input)) {
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    auto elem_num = seq_abs->size();
    if (elem_num == 0) {
      MS_LOG(DEBUG) << "Empty sequence for node:" << real_input->fullname_with_scope();
      return nullptr;
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
    }
  } else {
    MS_LOG(EXCEPTION) << "For node:" << real_input->fullname_with_scope() << ", abstract(" << abs->ToString()
                      << ") is invalid.";
  }

  return std::make_shared<tensor::Tensor>(type, shape);
}

tensor::TensorPtr GetDependValueTensor(const AnfNodePtr &node, size_t i,
                                       const std::pair<AnfNodePtr, size_t> &input_node_with_index, bool skip_nop_node,
                                       void *args) {
  auto real_input = input_node_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input);
  auto real_input_index = input_node_with_index.second;

  auto depended_value = CreateTensorMem(input_node_with_index);
  auto output_addr = AnfAlgo::GetMutableOutputAddr(real_input, real_input_index, skip_nop_node);
  if (output_addr != nullptr && output_addr->IsPtrValid()) {
    // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
    // address size will be wrong in the dynamic shape scenario.
    depended_value->set_device_address(output_addr, false);
    depended_value->data_sync();
  } else {
    // If real_input is parameter and is control flow's output, the device address stored in AnfNode is useless.
    if (args == nullptr) {
      MS_LOG(EXCEPTION) << "Address is nullptr, and no valid address args is passed!";
    }
    auto input_device_address = reinterpret_cast<std::vector<device::DeviceAddress *> *>(args);
    if (i >= input_device_address->size() || input_device_address->at(i) == nullptr) {
      MS_EXCEPTION_IF_NULL(node);
      if (IsPrimitiveCNode(node, prim::kPrimPyExecute)) {
        MS_LOG(INFO) << "There is no valid address for " << i << " input of " << node->DebugString() << ", "
                     << node->fullname_with_scope();
        return depended_value;
      }
      MS_LOG(EXCEPTION) << "There is no valid address for " << i << " input of " << node->DebugString() << ", "
                        << node->fullname_with_scope();
    }

    depended_value->data_sync_directly(input_device_address->at(i));
  }

  return depended_value;
}

abstract::AbstractBasePtr MakeNewAbstract(const AnfNodePtr &input, const tensor::TensorPtr &depended_value,
                                          const size_t &input_index) {
  auto abs = input->abstract();
  abstract::AbstractBasePtr new_abs;
  if (abs->isa<abstract::AbstractTensor>()) {
    new_abs = abs->Clone();
    new_abs->set_value(depended_value);

    // Set user data for PyExecute infer.
    if (input->has_user_data<kernel::PyExecuteOutputData>()) {
      const auto &output_data = input->user_data<kernel::PyExecuteOutputData>();
      new_abs->set_user_data<kernel::PyExecuteOutputData>(output_data);
    }
  } else if (abs->isa<abstract::AbstractScalar>()) {
    auto type = depended_value->Dtype()->type_id();
    if (type == kNumberTypeInt32) {
      auto tensor_data = reinterpret_cast<int32_t *>(depended_value->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
    } else if (type == kNumberTypeInt64) {
      auto tensor_data = reinterpret_cast<int64_t *>(depended_value->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      new_abs = std::make_shared<abstract::AbstractScalar>(*tensor_data);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type: " << type;
    }
  } else if (AnfAlgo::IsRealSquenceOutput(input)) {
    auto type = depended_value->Dtype()->type_id();
    AbstractBasePtrList elems;
    if (type == kNumberTypeInt32) {
      auto tensor_data = reinterpret_cast<int32_t *>(depended_value->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      for (size_t i = 0; i < depended_value->DataSize(); i++) {
        auto scalar = std::make_shared<abstract::AbstractScalar>(tensor_data[i]);
        (void)elems.emplace_back(scalar);
      }
    } else if (type == kNumberTypeInt64) {
      auto tensor_data = reinterpret_cast<int64_t *>(depended_value->data_c());
      MS_EXCEPTION_IF_NULL(tensor_data);
      for (size_t i = 0; i < depended_value->DataSize(); i++) {
        auto scalar = std::make_shared<abstract::AbstractScalar>(tensor_data[i]);
        (void)elems.emplace_back(scalar);
      }
    } else {
      MS_LOG(EXCEPTION) << "Unsupported type:" << type;
    }

    if (abs->isa<abstract::AbstractTuple>()) {
      new_abs = std::make_shared<abstract::AbstractTuple>(elems);
    } else if (abs->isa<abstract::AbstractList>()) {
      new_abs = std::make_shared<abstract::AbstractList>(elems);
    }

    MS_LOG(EXCEPTION) << "Unsupported abstract type:" << abs->ToString();
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto abstract_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abstract_seq);
    MS_EXCEPTION_IF_CHECK_FAIL((input_index < abstract_seq->elements().size()), "Index is out of range.");
    new_abs = abstract_seq->elements()[input_index]->Clone();
    new_abs->set_value(depended_value);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported abstract type:" << abs->ToString();
  }

  return new_abs;
}

void InferShape(const CNodePtr &cnode, std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(DEBUG) << "InferShape start, node:" << cnode->fullname_with_scope();
  std::set<int64_t> depend_list = abstract::GetValueDependArgIndices(cnode);
  auto ret = InferShapeForDefiniteOutputNode(cnode);
  if (ret) {
    return;
  }

  depend_tensor_map->clear();
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  bool skip_nop_node = !context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  bool has_py_execute_data = false;
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
      if (updated_abs->has_user_data<kernel::PyExecuteOutputData>()) {
        has_py_execute_data = true;
      }
      (void)args_spec_list.emplace_back(updated_abs);
    } else {
      auto abs = real_input->abstract();
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

  if (!has_py_execute_data && !IsPrimitiveCNode(cnode, prim::kPrimPyExecute)) {
    // Pynative mode is rely on the origin abstract of cnode, so cannot modify the abstract inplace, clone from old
    // abstract instead.
    opt::CppInferShape(primitive, args_spec_list, cnode);
  } else {
    if (cpp_infer_py_handler_ == nullptr) {
      MS_LOG(EXCEPTION) << "\'cpp_infer_py_handler_\' should not be null.";
    }
    const auto &abs = cpp_infer_py_handler_(cnode, primitive, args_spec_list);
    cnode->set_abstract(abs);
  }
}

inline bool IsDeprecatedCpuOrGpuKernelMod(kernel::KernelModType kernel_mod_type) {
  return kernel_mod_type == kernel::KernelModType::DeprecatedNativeGpuKernelMod ||
         kernel_mod_type == kernel::KernelModType::DeprecatedNativeCpuKernelMod;
}

inline bool IsCpuGpuKernelMod(kernel::KernelModType kernel_mod_type) {
  return kernel_mod_type == kernel::KernelModType::NativeGpuKernelMod ||
         kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod ||
         kernel_mod_type == kernel::KernelModType::DeprecatedNativeGpuKernelMod ||
         kernel_mod_type == kernel::KernelModType::DeprecatedNativeCpuKernelMod;
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
    if (kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
        static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
      MS_LOG(EXCEPTION) << "Node " << cnode->fullname_with_scope() << " Resize failed.";
    }
  };

  auto init_node = AnfUtils::NewInitActorNode(actor_func, cnode);
  init_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return init_node;
}

void InferOp(const CNodePtr &cnode, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  kernel::KernelArgs kernel_args;
  InferShape(cnode, &kernel_args.depend_tensor_map, args);

  if (auto kernel_mod_type = kernel_mod->GetKernelModType(); IsCpuGpuKernelMod(kernel_mod_type)) {
    auto update = kernel::AbstractArgsFromCNode(cnode, IsDeprecatedCpuOrGpuKernelMod(kernel_mod_type));
    update.depend_tensor_map = std::move(kernel_args.depend_tensor_map);
    kernel::SetInputsByDependMap(update.depend_tensor_map, &update.inputs, IsCpuKernelMod(kernel_mod_type));
    kernel::SetArgsToCNode(cnode, update);
  } else {
    kernel::SetArgsToCNode(cnode, kernel_args);
  }
}

CustomActorNodeManager &CustomActorNodeManager::Instance() {
  static CustomActorNodeManager instance{};
  return instance;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
