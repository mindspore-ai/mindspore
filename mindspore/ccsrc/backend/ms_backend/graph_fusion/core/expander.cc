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

#include "backend/ms_backend/graph_fusion/core/expander.h"

#include <string>
#include <algorithm>

#include "ir/scope.h"
#include "utils/anf_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/core/value_depend_op_utils.h"
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "backend/ms_backend/graph_fusion/expander/mindir_adapter/mindir_emitter.h"

namespace mindspore::graphkernel {
ExpanderPtr WrapExpander(const ExpanderPtr &base, const ExpanderCreatorFuncList &deco_creators) {
  ExpanderPtr result = base;
  for (auto it = deco_creators.rbegin(); it != deco_creators.rend(); ++it) {
    result = (*it)(result);
  }
  return result;
}

AnfNodePtr ExpanderDecorator::Run(const AnfNodePtr &node) { return node ? decorated_->Run(node) : nullptr; }

CNodePtr ExpanderDecorator::QuickCloneCNode(const AnfNodePtr &node, bool clone_prim) const {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_node = func_graph->NewCNode(cnode->inputs());
  new_node->CloneCNodeInfo(cnode);
  new_node->set_fullname_with_scope(node->fullname_with_scope());
  if (clone_prim) {
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    new_node->set_input(0, NewValueNode(prim->Clone()));
  }
  return new_node;
}

AnfNodePtr DependValueDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  MS_EXCEPTION_IF_NULL(cnode);
  (void)ValueDependOpUtils::AddConstInputToAttr(cnode, input_idx_);
  return decorated_->Run(cnode);
}

// check whether the res_idx-th output of subgraph has same shape, dtype, and format
// as the ori_idx-th output of the original node
bool IsOutputInfoIdentical(const expander::NodePtrList &outputs, const CNodePtr &node, size_t res_idx, size_t ori_idx,
                           const CallbackPtr &cb) {
  const auto &res_shape = outputs[res_idx]->GetShape();
  const auto &res_dtype = outputs[res_idx]->GetDtype()->type_id();
  const auto &res_format = outputs[res_idx]->GetFormat();
  const auto &ori_shape = cb->GetOutputShape(node, ori_idx);
  const auto &ori_dtype = cb->GetOutputType(node, ori_idx);
  const auto &ori_format = cb->GetOutputFormat(node, ori_idx);
  bool is_format_identical = (res_format == ori_format);
  if ((res_format == kOpFormat_DEFAULT && ori_format == kOpFormat_NCHW) ||
      (res_format == kOpFormat_NCHW && ori_format == kOpFormat_DEFAULT)) {
    is_format_identical = true;
  }
  if (res_shape == ori_shape && res_dtype == ori_dtype && is_format_identical) {
    auto out_node = outputs[res_idx]->as<AnfNodePtr>();
    MS_EXCEPTION_IF_NULL(out_node);
    auto ori_abs = node->abstract();
    if (ori_abs->isa<abstract::AbstractTuple>()) {
      ori_abs = ori_abs->cast<abstract::AbstractTuplePtr>()->elements()[ori_idx];
    }
    MS_EXCEPTION_IF_NULL(ori_abs);
    out_node->set_abstract(ori_abs);
    return true;
  }
  return false;
}

AnfNodePtr DefaultExpander::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_fg = ExpandToGraph(cnode);
  auto res = new_fg ? "success" : "failed";
  MS_LOG(DEBUG) << "Expanding node " << res << " : " << node->fullname_with_scope();
  if (new_fg == nullptr) {
    return nullptr;
  }
  return CreateCallCNode(new_fg, cnode);
}

FuncGraphPtr DefaultExpander::ExpandToGraph(const CNodePtr &node) {
  auto name = AnfUtils::GetCNodeName(node);
  auto &ib_registry = expander::IrBuilderRegistry::Instance();
  auto ib = ib_registry.GetOp(name);
  if (ib == nullptr) {
    MS_LOG(INFO) << "irbuilder not found: " << node->fullname_with_scope();
    return nullptr;
  }
  MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope() << " by MetaExpander";
  auto fg = std::make_shared<FuncGraph>();
  auto scope = std::make_shared<Scope>(node->scope()->name() + "/expand_" + name);
  auto e = std::make_shared<expander::MindirEmitter>(fg, cb_->IsUseDeviceInfo(), scope);
  auto inputs = e->Inputs(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  ib->Init(e, &inputs, &prim->attrs(), cb_->GetProcessor(node));
  auto outputs = ib->Expand();
  if (outputs.empty()) {
    return nullptr;
  }
  auto cb = Callback::Instance();
  if (!ib_registry.IsOutputNumInconsistent(name)) {
    for (size_t i = 0; i < outputs.size(); i++) {
      if (!IsOutputInfoIdentical(outputs, node, i, i, cb)) {
        MS_LOG(INFO) << "Expanding node: " << node->fullname_with_scope() << " failed, because subgraph output " << i
                     << " has a different shape/dtype/format from the corresponding output of the original cnode.";
        return nullptr;
      }
    }
  } else {
    auto real_output_indices = ib_registry.GetOutputNumInconsistentOps().at(name);
    for (size_t i = 0; i < real_output_indices.size(); i++) {
      if (!IsOutputInfoIdentical(outputs, node, real_output_indices[i], i, cb)) {
        MS_LOG(INFO) << "Expanding node: " << node->fullname_with_scope() << " failed, because subgraph output "
                     << real_output_indices[i] << " has a different shape/dtype/format from the corresponding output "
                     << i << " of the original cnode.";
        return nullptr;
      }
    }
  }
  if (outputs.size() > 1) {
    fg->set_output(e->MakeTuple(outputs)->as<AnfNodePtr>());
  } else {
    fg->set_output(outputs[0]->as<AnfNodePtr>());
  }
  return fg;
}

AnfNodePtr DefaultExpander::CreateCallCNode(const FuncGraphPtr &sub_fg, const CNodePtr &cnode) {
  const auto fg = NewValueNode(sub_fg);
  AnfNodeWeakPtrList inputs = {fg};
  // Be consistent with e->inputs(node)
  std::copy_if(cnode->weak_inputs().cbegin() + 1, cnode->weak_inputs().cend(), std::back_inserter(inputs),
               [](const AnfNodeWeakPtr &node) { return !(node.lock()->isa<ValueNode>()); });
  return cnode->func_graph()->NewCNodeWeak(inputs);
}

FuncGraphPtr LitegraphExpander::ExpandToGraph(const CNodePtr &node) {
  auto name = AnfUtils::GetCNodeName(node);
  if (expander::IrBuilderRegistry::Instance().HasOp(name)) {
    return DefaultExpander::ExpandToGraph(node);
  }
  MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope() << " by LitegraphExpander";
  auto op_desc = expanders::OpDescFactory::Instance().GetOp(name);
  if (op_desc == nullptr) {
    MS_LOG(INFO) << "expander not found " << node->fullname_with_scope();
    return nullptr;
  }
  expanders::BaseInfoList inputs(node->size() - 1);
  expanders::BaseInfoList outputs(AnfUtils::GetOutputTensorNum(node));
  MS_EXCEPTION_IF_NULL(cb_);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs[i].shape = cb_->GetInputShape(node, i);
    inputs[i].type = cb_->GetInputType(node, i);
    inputs[i].format = cb_->GetInputFormat(node, i);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i].shape = cb_->GetOutputShape(node, i);
    outputs[i].type = cb_->GetOutputType(node, i);
    outputs[i].format = cb_->GetOutputFormat(node, i);
  }
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &attrs = prim->attrs();
  auto litegraph = op_desc->Run(inputs, outputs, attrs, cb_->GetProcessor(node));
  if (litegraph == nullptr) {
    MS_LOG(INFO) << "undo expanding " << node->fullname_with_scope();
    return nullptr;
  }
  return GkUtils::LiteGraph2AnfGraph(litegraph, cb_);
}

AnfNodePtr LitegraphExpander::CreateCallCNode(const FuncGraphPtr &fg, const CNodePtr &cnode) {
  if (expander::IrBuilderRegistry::Instance().HasOp(AnfUtils::GetCNodeName(cnode))) {
    return DefaultExpander::CreateCallCNode(fg, cnode);
  }
  const auto fg_node = NewValueNode(fg);
  AnfNodeWeakPtrList inputs = {fg_node};
  (void)inputs.insert(inputs.end(), cnode->weak_inputs().cbegin() + 1, cnode->weak_inputs().cend());
  return cnode->func_graph()->NewCNodeWeak(inputs);
}
}  // namespace mindspore::graphkernel
