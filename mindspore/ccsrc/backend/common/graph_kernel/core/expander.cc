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

#include "backend/common/graph_kernel/core/expander.h"

#include <string>

#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/value_depend_op_utils.h"
#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "backend/common/graph_kernel/expander/base/ir_builder.h"
#include "backend/common/graph_kernel/expander/mindir_adapter/mindir_emitter.h"

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
    new_node->set_input(0, NewValueNode(GetCNodePrimitive(node)->Clone()));
  }
  return new_node;
}

AnfNodePtr DependValueDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  MS_EXCEPTION_IF_NULL(cnode);
  (void)ValueDependOpUtils::AddConstInputToAttr(cnode, input_idx_);
  return decorated_->Run(cnode);
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
  AnfNodePtrList inputs = {NewValueNode(new_fg)};
  (void)inputs.insert(inputs.end(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  return node->func_graph()->NewCNode(inputs);
}

FuncGraphPtr DefaultExpander::ExpandToGraph(const CNodePtr &node) {
  auto name = AnfUtils::GetCNodeName(node);
  auto ib = expander::IrBuilderRegistry::Instance().GetOp(name);
  if (ib == nullptr) {
    MS_LOG(INFO) << "irbuilder not found: " << node->fullname_with_scope();
    return nullptr;
  }
  MS_LOG(DEBUG) << "Expanding node: " << node->fullname_with_scope() << " by MetaExpander";
  auto fg = std::make_shared<FuncGraph>();
  auto scope = std::make_shared<Scope>(node->scope()->name() + "/expand_" + name);
  auto e = std::make_shared<expander::MindirEmitter>(fg, cb_->IsUseDeviceInfo(), scope);
  auto inputs = e->Inputs(node);
  ib->Init(e, &inputs, &GetCNodePrimitive(node)->attrs(), cb_->GetProcessor(node));
  auto outputs = ib->Expand();
  if (outputs.empty()) {
    return nullptr;
  }
  if (outputs.size() > 1) {
    fg->set_output(e->MakeTuple(outputs)->as<AnfNodePtr>());
  } else {
    fg->set_output(outputs[0]->as<AnfNodePtr>());
  }
  return fg;
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
  auto &attrs = GetCNodePrimitive(node)->attrs();
  auto litegraph = op_desc->Run(inputs, outputs, attrs, cb_->GetProcessor(node));
  if (litegraph == nullptr) {
    MS_LOG(INFO) << "undo expanding " << node->fullname_with_scope();
    return nullptr;
  }
  return GkUtils::LiteGraph2AnfGraph(litegraph, cb_);
}
}  // namespace mindspore::graphkernel
