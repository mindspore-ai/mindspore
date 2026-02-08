/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/core/graph_kernel_splitter_rebuild.h"
#include <string>
#include <algorithm>
#include <utility>
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "utils/anf_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_splitter.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/core/graph_builder.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel {
void Rebuilder::Rebuild() {
  CreateSubGraphs();
  ConnectSubGraphs();
  ConnectToMainGraph();
  Inline();
}

CNodePtr Rebuilder::InlineSubFuncGraph(const CNodePtr &main_node) {
  auto &user = mng_->node_users();
  auto func_graph = GetCNodeFuncGraph(main_node);
  const auto &inputs = main_node->inputs();
  auto output = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output);
  const auto &parameters = func_graph->parameters();
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> param_input;
  for (size_t i = 0; i < parameters.size(); ++i) {
    param_input[parameters[i]] = inputs[i + 1];
  }
  auto sub_nodes = TopoSort(func_graph->get_return());
  for (auto &node : sub_nodes) {
    if (auto cnode = node->cast<CNodePtr>(); cnode != nullptr) {
      cnode->set_func_graph(main_graph_);
      for (size_t i = 1; i < cnode->size(); ++i) {
        auto iter = param_input.find(cnode->input(i));
        if (iter != param_input.end()) {
          cnode->set_input(i, iter->second);
          user[iter->second].insert(std::make_pair(cnode, i));
        }
      }
      if (AnfUtils::IsRealKernel(node)) {
        cnode->AddAttr(kNeedResetKernelInfo, MakeValue(true));
      }
    }
  }
  return output;
}

void Rebuilder::Inline() {
  for (auto &node : need_inline_cnodes_) {
    auto new_node = InlineSubFuncGraph(node);
    (void)mng_->Replace(node, new_node);
  }
}

void Rebuilder::ConnectToMainGraph() {
  auto fg = GetCNodeFuncGraph(main_cnode_);
  auto out = fg->output();
  AnfNodePtr new_cnode = nullptr;
  if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
    AnfNodePtrList maketuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    auto &inputs = out->cast<CNodePtr>()->inputs();
    (void)std::transform(
      inputs.begin() + 1, inputs.end(), std::back_inserter(maketuple_inputs), [this](const AnfNodePtr &node) {
        auto it = this->old2new_.find(node);
        MS_EXCEPTION_IF_CHECK_FAIL(it != this->old2new_.end(), node->ToString() + " is not in old2new!");
        return it->second;
      });
    new_cnode = main_graph_->NewCNode(maketuple_inputs);
    AbstractBasePtrList out_abs_list;
    (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(out_abs_list),
                         [](const AnfNodePtr &node) { return node->abstract(); });
    new_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
  } else {
    auto it = old2new_.find(out);
    MS_EXCEPTION_IF_CHECK_FAIL(it != old2new_.end(), out->ToString() + " is not in old2new!");
    new_cnode = it->second;
  }
  (void)mng_->Replace(main_cnode_, new_cnode);
}

void Rebuilder::ConnectSubGraphs() {
  auto &user = mng_->node_users();
  for (auto &call : call_nodes_) {
    auto inputs = call->inputs();
    for (size_t i = 1; i < inputs.size(); i++) {
      auto input = inputs[i];
      if (input->isa<Parameter>()) {
        auto param = input->cast<ParameterPtr>();
        auto it = param_to_main_graph_node_map_.find(param);
        MS_EXCEPTION_IF_CHECK_FAIL(it != param_to_main_graph_node_map_.end(),
                                   param->ToString() + " is not in param_to_main_graph_node_map!");
        call->set_input(i, it->second);
        user[it->second].insert(std::make_pair(call, i));
      } else {
        auto it = old2new_.find(input);
        MS_EXCEPTION_IF_CHECK_FAIL(it != old2new_.end(), input->ToString() + " is not in old2new!");
        call->set_input(i, it->second);
        user[it->second].insert(std::make_pair(call, i));
      }
    }
  }
}

void Rebuilder::SetSplitNodeName(const AnfNodePtr &callnode, size_t i) const {
  auto fg = GetCNodeFuncGraph(main_cnode_);
  std::string ori_node_name;
  if (fg->has_attr(kAttrNodeName)) {
    ori_node_name = GetValue<std::string>(fg->get_attr(kAttrNodeName));
  } else {
    ori_node_name = GetValue<std::string>(fg->get_attr("graph_kernel"));
  }
  if (!split_schemer_->NeedInline(i)) {
    std::string node_name = ori_node_name + "_" + std::to_string(i);
    AnfUtils::SetNodeAttr(kAttrNodeName, MakeValue(node_name), callnode);
    auto sub_fg = GetCNodeFuncGraph(callnode);
    auto attr = GkUtils::ExtractGraphKernelName(TopoSort(sub_fg->get_return()), "", "split");
    sub_fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(attr));
    mng_->AddFuncGraph(sub_fg);
  }
}

void Rebuilder::CreateSubGraphs() {
  auto plan = split_schemer_->split_plan();
  for (size_t i = 0; i < plan.size(); i++) {
    auto nodes = plan[i];
    if (nodes.size() == 2 && IsPrimitiveCNode(nodes[0], prim::kPrimMakeTuple) &&
        IsPrimitiveCNode(nodes[1], prim::kPrimReturn)) {
      continue;
    }
    auto config = graphkernel::ClusterConfig();
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = BuildGraphFromNodesInner(nodes, config);

    AnfNodePtrList call_inputs = {NewValueNode(fg)};
    (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(call_inputs),
                         [](const AnfNodePtr &node) { return node; });
    auto callnode = main_graph_->NewCNode(call_inputs);
    callnode->set_abstract(fg->output()->abstract());
    callnode->AddAttr(kNeedKernelInfo, MakeValue(true));
    SetSplitNodeName(callnode, i);
    if (split_schemer_->NeedInline(i)) {
      need_inline_cnodes_.emplace_back(callnode);
    }
    call_nodes_.emplace_back(callnode);
    if (outputs.size() == 1) {
      old2new_[outputs[0]] = callnode;
    } else {
      for (size_t j = 0; j < outputs.size(); j++) {
        auto idx_val = SizeToLong(j);
        auto idx = NewValueNode(idx_val);
        idx->set_abstract(std::make_shared<abstract::AbstractScalar>(idx_val));
        AnfNodePtrList getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem), callnode, idx};
        auto getitem_node = main_graph_->NewCNode(getitem_inputs);
        auto abs_tuple = dyn_cast<abstract::AbstractTuple>(callnode->abstract());
        MS_EXCEPTION_IF_CHECK_FAIL(j < abs_tuple->elements().size(), "overflow!");
        getitem_node->set_abstract(abs_tuple->elements()[j]);
        old2new_[outputs[j]] = getitem_node;
      }
    }
  }
}
}  // namespace mindspore::graphkernel
