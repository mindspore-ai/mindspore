/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "kernel/kernel.h"

#include <algorithm>
#include <stack>
#include "utils/ms_context.h"
#include "utils/anf_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

void KernelMod::SetAtomicCleanNodes(const std::vector<CNodePtr> &atomic_clean_node) {
  atomic_clean_nodes_.resize(atomic_clean_node.size());
  for (size_t i = 0; i < atomic_clean_node.size(); ++i) {
    atomic_clean_nodes_[i] = atomic_clean_node[i];
  }
}

void KernelMod::InferShape() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "InferShape start, node:" << cnode->fullname_with_scope();
  GetDepndLists(cnode);
  auto ret = InferShapeForDefiniteOutputNode(cnode);
  if (ret) {
    return;
  }
  depend_tensor_map_.clear();
  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid inputs.";
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  AbstractBasePtrList args_spec_list;
  auto primitive = GetValueNode<PrimitivePtr>(inputs[0]);
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_size; i++) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
    auto real_input = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input);
    auto cnode_input = cnode->input(i + 1);
    MS_EXCEPTION_IF_NULL(cnode_input);
    InferShapeForNopNode(real_input);
    if (depend_list_.find(i) != depend_list_.end()) {
      auto pre_node_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, i);
      bool skip_nop_node = !context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
      auto output_addr = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, i, skip_nop_node);
      std::vector<int64_t> shapes =
        trans::GetRuntimePaddingShape(pre_node_with_index.first, pre_node_with_index.second);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(pre_node_with_index.first, pre_node_with_index.second);
      auto out_tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
      MS_EXCEPTION_IF_NULL(out_tensor);
      // The second parameter must be false, otherwise the device address cannot be released and allocated, and the
      // address size will be wrong in the dynamic shape scenario.
      out_tensor->set_device_address(output_addr, false);
      auto ret2 = depend_tensor_map_.try_emplace(i, out_tensor);
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
        auto tuple_get_item_index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode_input->cast<CNodePtr>());
        auto abstract_tuple = real_abs->cast<abstract::AbstractTuplePtr>();
        MS_EXCEPTION_IF_NULL(abstract_tuple);
        auto tuple_elements = abstract_tuple->elements()[tuple_get_item_index];
        tuple_elements->set_value(out_tensor);
      }
    }
    common::AnfAlgo::AddArgList(&args_spec_list, cnode_input, real_input, i);
  }
  auto eval_result = opt::CppInferShape(primitive, args_spec_list);
  cnode->set_abstract(eval_result);
}

void KernelMod::UpdateOutputSizeList() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  for (size_t i = 0; i < output_size_list_.size(); ++i) {
    auto ori_output_size = output_size_list_[i];
    auto real_output_size = AnfAlgo::GetOutputTensorMemSize(cnode, i);
    if (ori_output_size != real_output_size) {
      output_size_list_[i] = real_output_size;
    }
  }
}

bool KernelMod::InferShapeForDefiniteOutputNode(const CNodePtr &cnode) {
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

void KernelMod::InferShapeForNopNode(const AnfNodePtr &input_node) {
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
  /*lint -e716*/
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

  /*lint +e716*/
  while (!nop_road.empty()) {
    auto nop_node = nop_road.top();
    MS_EXCEPTION_IF_NULL(nop_node);
    AnfAlgo::InferShape(nop_node->cast<CNodePtr>());
    nop_road.pop();
  }
}

void KernelMod::GetDepndLists(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (depend_list_.size() != 0) {
    return;
  }
  auto ret = abstract::GetDependsFormMap(cnode);
  if (ret.empty()) {
    MS_LOG(DEBUG) << "No dynamic_shape_depends found.";
    return;
  }
  MS_LOG(INFO) << "Have depends.";
  (void)std::transform(ret.begin(), ret.end(), std::inserter(depend_list_, depend_list_.begin()),
                       [](const int64_t &value) { return static_cast<int>(value); });
  MS_LOG(INFO) << "Init End.";
}

bool KernelMod::NeedSkipExecute(const CNodePtr &cnode) {
  // Skip run ReduceSum when axis is a Empty Tensor
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  if (op_name != kReduceSumOpName) {
    return false;
  }

  const size_t axes_index = 1;
  if (cnode->inputs().size() <= axes_index + 1) {
    return false;
  }
  auto input_axes = cnode->input(axes_index + 1);
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(input_axes.get());
  auto axes_abs = input_axes->abstract()->Clone();
  MS_EXCEPTION_IF_NULL(axes_abs);
  auto axes_shape = AnfAlgo::GetInputDeviceShape(cnode, axes_index);
  if (axes_abs->isa<abstract::AbstractTensor>()) {
    if (std::any_of(axes_shape.begin(), axes_shape.end(), [](ssize_t shape) { return shape == 0; })) {
      return true;
    }
  }
  return false;
}
}  // namespace kernel
}  // namespace mindspore
