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

#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

#include <memory>
#include <stack>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_dtype_record.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace opt::dynamic_shape {
namespace {
constexpr int64_t kInvalidShape = -2;

void InferShapeForNopNode(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!common::AnfAlgo::IsNopNode(input_node) || !common::AnfAlgo::IsDynamicShape(input_node)) {
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

tensor::TensorPtr GetDependValueTensor(const AnfNodePtr &node, size_t i,
                                       const std::pair<AnfNodePtr, size_t> &input_node_with_index, bool skip_nop_node,
                                       void *args, bool abstract_in_cache) {
  auto real_input = input_node_with_index.first;
  MS_EXCEPTION_IF_NULL(real_input);
  auto real_input_index = input_node_with_index.second;
  auto shapes = trans::GetRuntimePaddingShape(real_input, real_input_index);
  TypeId host_type;
  if (abstract_in_cache) {
    // for cnode in the cache, we use device type as there is a mismatch
    host_type = AnfAlgo::GetOutputDeviceDataType(real_input, real_input_index);
  } else {
    // for cnode not in the cache, valuenodes and other nodes, we use inferred type
    host_type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);
  }
  auto out_tensor = std::make_shared<tensor::Tensor>(host_type, shapes);

  auto output_addr = AnfAlgo::GetMutableOutputAddr(real_input, real_input_index, skip_nop_node);
  if (output_addr != nullptr && output_addr->GetPtr() != nullptr) {
    // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
    // address size will be wrong in the dynamic shape scenario.
    out_tensor->set_device_address(output_addr, false);
    out_tensor->data_sync();
  } else {
    // If real_input is parameter and is control flow's output, the device address stored in AnfNode is useless.
    if (args == nullptr) {
      MS_LOG(EXCEPTION) << "Address is nullptr, and no valid address args is passed!";
    }
    auto input_device_address = reinterpret_cast<std::vector<device::DeviceAddress *> *>(args);
    if (i >= input_device_address->size() || input_device_address->at(i) == nullptr) {
      MS_EXCEPTION_IF_NULL(node);
      MS_LOG(EXCEPTION) << "There is no valid address for " << i << " input of " << node->fullname_with_scope();
    }

    out_tensor->data_sync_directly(input_device_address->at(i));
  }

  return out_tensor;
}

void InferShape(const CNodePtr &cnode, std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map, void *args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(INFO) << "InferShape start, node:" << cnode->fullname_with_scope();
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
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i, false);
    auto real_input = input_node_with_index.first;
    auto real_input_index = input_node_with_index.second;

    bool abstract_in_cache = DynamicShapeDtypeManager::GetInstance().CheckDeviceType(real_input);
    AbstractBasePtr cached_abstract;
    AbstractBasePtr real_input_abs = real_input->abstract();

    if (abstract_in_cache) {
      auto cached_type_list = DynamicShapeDtypeManager::GetInstance().GetDeviceType(real_input);
      if (real_input_abs->isa<abstract::AbstractTensor>()) {
        auto shape_ptr = real_input_abs->BuildShape();
        cached_abstract = std::make_shared<abstract::AbstractTensor>(cached_type_list[0], shape_ptr);
      } else if (real_input_abs->isa<abstract::AbstractTuple>()) {
        auto abstract_tuple = real_input_abs->cast<abstract::AbstractTuplePtr>();
        MS_EXCEPTION_IF_NULL(abstract_tuple);
        AbstractBasePtrList abstract_list;

        for (size_t output_index = 0; output_index < cached_type_list.size(); ++output_index) {
          auto cur_element = abstract_tuple->elements()[output_index];
          MS_EXCEPTION_IF_NULL(cur_element);
          auto shape_ptr = cur_element->BuildShape();
          auto new_abstract = std::make_shared<abstract::AbstractTensor>(cached_type_list[output_index], shape_ptr);
          abstract_list.push_back(new_abstract);
        }
        cached_abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
      } else {
        MS_LOG(EXCEPTION) << "Output of " << real_input->fullname_with_scope()
                          << " is neither a Tensor nor a Tuple of Tensor, but " << real_input_abs->ToString();
      }
    }
    MS_EXCEPTION_IF_NULL(real_input);
    if (skip_nop_node) {
      InferShapeForNopNode(real_input);
    }
    if (depend_list.find(i) != depend_list.end()) {
      auto out_tensor = GetDependValueTensor(cnode, i, input_node_with_index, skip_nop_node, args, abstract_in_cache);
      auto ret2 = depend_tensor_map->try_emplace(i, out_tensor);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }

      // cppcheck-suppress unreadVariable
      auto lock = AnfUtils::GetAbstractLock(real_input.get());
      AbstractBasePtr real_abs;
      if (abstract_in_cache) {
        real_abs = cached_abstract;
      } else {
        real_abs = real_input->abstract();
      }
      if (real_abs->isa<abstract::AbstractTensor>()) {
        real_abs->set_value(out_tensor);
      } else if (real_abs->isa<abstract::AbstractTuple>()) {
        auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
        MS_EXCEPTION_IF_NULL(abstract_tuple);
        MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abstract_tuple->elements().size()), "Index is out of range.");
        auto tuple_elements = abstract_tuple->elements()[real_input_index];
        tuple_elements->set_value(out_tensor);
      }
    }
    if (abstract_in_cache) {
      if (cached_abstract->isa<abstract::AbstractTuple>()) {
        auto abs_tuple = cached_abstract->Clone()->cast<abstract::AbstractTuplePtr>();
        MS_EXCEPTION_IF_NULL(abs_tuple);
        MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abs_tuple->elements().size()), "Index is out of range.");
        auto abs_index = abs_tuple->elements()[real_input_index];
        (void)args_spec_list.emplace_back(abs_index);
      } else {
        (void)args_spec_list.emplace_back(cached_abstract->Clone());
      }
    } else {
      common::AnfAlgo::AddArgList(&args_spec_list, real_input, real_input_index);
    }
  }

  // Pynative mode is rely on the origin abstract of cnode, so cannot modify the abstract inplace, clone from old
  // abstract instead.
  auto old_abs = cnode->abstract();
  MS_EXCEPTION_IF_NULL(old_abs);
  auto new_abs = old_abs->Clone();
  opt::CppInferShape(primitive, args_spec_list, new_abs);
  MS_LOG(DEBUG) << "The abstract of " << cnode->fullname_with_scope() << " changes from " << old_abs << " to "
                << new_abs;
  cnode->set_abstract(new_abs);
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
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(WARNING) << "The node " << cnode->fullname_with_scope() << " is not dynamic shape.";
    return;
  }

  kernel::KernelArgs kernel_args;
  if (AnfAlgo::IsDynamicShapeSkipExecute(cnode)) {
    std::vector<TypeId> dtypes{common::AnfAlgo::GetOutputInferDataType(cnode, 0)};
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetInputDeviceShape(cnode, 0)}, cnode.get());
  } else {
    InferShape(cnode, &kernel_args.depend_tensor_map, args);
  }

  if (auto kernel_mod_type = kernel_mod->GetKernelModType();
      kernel_mod_type == kernel::KernelModType::NativeGpuKernelMod ||
      kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod) {
    auto update = kernel::AbstractArgsFromCNode(cnode);
    update.depend_tensor_map = std::move(kernel_args.depend_tensor_map);
    for (const auto &[i, tensor] : update.depend_tensor_map) {
      if (i >= update.inputs.size()) {
        MS_LOG(EXCEPTION) << "Type to store the data to KernelTensor, expect less than" << update.inputs.size()
                          << " but got " << i;
      }
      MS_EXCEPTION_IF_NULL(update.inputs[i]);
      MS_EXCEPTION_IF_NULL(tensor);
      auto address = std::make_shared<kernel::Address>(tensor->data_c(), tensor->Size());
      if (kernel_mod_type == kernel::KernelModType::NativeCpuKernelMod) {
        // Store the data address in device one for cpu.
        update.inputs[i]->SetData(address);
        continue;
      }
      update.inputs[i]->SetHostData(address);
    }

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
