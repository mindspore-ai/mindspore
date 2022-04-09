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
#include <vector>
#include <map>
#include <utility>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "kernel/kernel.h"
#include "utils/ms_context.h"
#include "abstract/primitive_infer_map.h"

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

void InferShape(const CNodePtr &cnode, std::map<uint32_t, tensor::TensorPtr> *depend_tensor_map) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(depend_tensor_map);
  MS_LOG(INFO) << "InferShape start, node:" << cnode->fullname_with_scope();
  std::set<int64_t> depend_list = abstract::GetDependsFormMap(cnode);
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
    MS_EXCEPTION_IF_NULL(real_input);
    if (skip_nop_node) {
      InferShapeForNopNode(real_input);
    }
    if (depend_list.find(i) != depend_list.end()) {
      auto output_addr = AnfAlgo::GetMutableOutputAddr(real_input, real_input_index, skip_nop_node);
      auto shapes = trans::GetRuntimePaddingShape(real_input, real_input_index);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(real_input, real_input_index);
      auto out_tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
      MS_EXCEPTION_IF_NULL(out_tensor);
      // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
      // address size will be wrong in the dynamic shape scenario.
      out_tensor->set_device_address(output_addr, false);
      auto ret2 = depend_tensor_map->try_emplace(i, out_tensor);
      if (!ret2.second) {
        MS_LOG(EXCEPTION) << "Insert map failed.";
      }
      out_tensor->data_sync();

      // cppcheck-suppress unreadVariable
      auto lock = AnfUtils::GetAbstractLock(real_input.get());
      auto real_abs = real_input->abstract();
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
    common::AnfAlgo::AddArgList(&args_spec_list, real_input, real_input_index);
  }
  auto eval_result = opt::CppInferShape(primitive, args_spec_list);
  cnode->set_abstract(eval_result);
}
}  // namespace
bool IsRealCNode(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n)) {
    CNodePtr cnode = utils::cast<CNodePtr>(n);
    return AnfUtils::IsRealKernel(cnode);
  }
  return false;
}

AnfNodePtr GenInferNode(const AnfNodePtr &node, bool fake_flag) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfUtils::CustomActorCallback actor_func;
  if (fake_flag) {
    actor_func = [](void *) -> void { return; };
  } else {
    actor_func = [cnode](void *) { InferOp(cnode); };
  }

  auto infer_node = AnfUtils::NewInferActorNode(actor_func, cnode, fake_flag);
  infer_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return infer_node;
}

AnfNodePtr GenInitNode(const AnfNodePtr &node, bool fake_flag) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfUtils::CustomActorCallback actor_func;
  if (fake_flag) {
    actor_func = [](void *) -> void { return; };
  } else {
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    actor_func = [kernel_mod, cnode](void *) { kernel_mod->InitOp(cnode->user_data<kernel::InitOpArgs>()); };
  }

  auto init_node = AnfUtils::NewInitActorNode(actor_func, cnode, fake_flag);
  init_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return init_node;
}

AnfNodePtr GenUpdateNode(const AnfNodePtr &node) {
  // Some not dynamic shape node should sync after launch for latter node.
  // Use a flag `just_sync_flag` to distinguish them with dynamic ones.
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto update_node = AnfUtils::NewUpdateActorNode([kernel_mod](void *) { kernel_mod->UpdateOp(); }, cnode);
  update_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  return update_node;
}

bool IsNeedUpdateOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->IsNeedUpdateOp();
}

void InferOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(WARNING) << "The node " << cnode->fullname_with_scope() << " is not dynamic shape.";
    return;
  }
  if (AnfAlgo::IsDynamicShapeSkipExecute(cnode)) {
    std::vector<TypeId> dtypes{common::AnfAlgo::GetOutputInferDataType(cnode, 0)};
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetInputDeviceShape(cnode, 0)}, cnode.get());
  } else {
    auto init_op_args = std::make_shared<kernel::InitOpArgs>();
    InferShape(cnode, &init_op_args->depend_tensor_map);
    cnode->set_user_data(init_op_args);
  }
}

CustomActorNodeManager &CustomActorNodeManager::Instance() {
  static CustomActorNodeManager instance{};
  return instance;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
