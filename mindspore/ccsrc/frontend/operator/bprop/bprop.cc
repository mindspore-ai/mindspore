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
#include "frontend/operator/bprop/bprop.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include "expander/infer.h"
#include "utils/anf_utils.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace expander {
namespace bprop {
class BpropExpander {
 public:
  BpropExpander(CNodePtrList *outputs, DoutUserType *dout_user, UserType *users)
      : outputs_(outputs), dout_user_(dout_user), users_(users) {}
  ~BpropExpander() = default;

  NodePtrList ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder) {
    NodePtrList nodes;
    nodes.reserve(cnode->size());
    (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(nodes),
                         [ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, ir_builder); });
    return nodes;
  }

  bool Run(const CNodePtr &cnode) {
    auto infer = std::make_shared<CppInfer>();
    auto name = AnfUtils::GetCNodeName(cnode);
    auto ir_builder = std::make_unique<BpropIRBuilder>(name, cnode->func_graph(), infer);
    auto inputs = ExtractInputs(cnode, ir_builder.get());
    auto &attrs = GetCNodePrimitive(cnode)->attrs();
    auto ret = ir_builder->Run(inputs, attrs, outputs_);
    if (!ret) {
      return false;
    }
    PostProcess(inputs);
    static bool dump_result = (common::GetEnv("MS_DEV_DUMP_BPROP") == "on");
    if (dump_result) {
      DumpResult(name, inputs);
    }
    return true;
  }

  void PostProcess(const NodePtrList &inputs) const {
    std::set<AnfNodePtr> visited;
    // do not visit the inputs again.
    std::for_each(inputs.cbegin(), inputs.cend(), [&visited](const NodePtr &node) { visited.insert(node->get()); });

    std::queue<CNodePtr> que;
    std::for_each(outputs_->cbegin(), outputs_->cend(), [&que](const CNodePtr &cnode) { que.push(cnode); });

    AnfNodePtr dout = inputs.back()->get();
    while (!que.empty()) {
      auto node = que.front();
      que.pop();
      for (size_t i = 1; i < node->size(); ++i) {
        const auto &inp = node->input(i);
        // record parameter's and dout's user
        if (dout_user_ != nullptr) {
          if (inp == dout) {
            (void)dout_user_->emplace_back(node, i);
          }
        } else {  // users_ != nullptr
          if (inp == dout || inp->isa<Parameter>()) {
            (*users_)[inp].emplace_back(node, i);
          }
        }
        if (IsPrimitiveCNode(inp, prim::kPrimTupleGetItem)) {
          auto getitem = inp->cast<CNodePtr>();
          auto real_input = getitem->input(kIndex1);
          // record the dout's successor getitem's users
          if (users_ != nullptr && real_input == dout) {
            (*users_)[inp].emplace_back(node, i);
          } else if (real_input->isa<ValueNode>()) {
            // eliminate redundant getitem
            auto real_input_value = real_input->cast<ValueNodePtr>()->value();
            if (real_input_value->isa<ValueSequence>()) {
              auto item_idx = GetValue<int64_t>(getitem->input(kIndex2)->cast<ValueNodePtr>()->value());
              auto newnode = NewValueNode((*(real_input_value->cast<ValueSequencePtr>()))[item_idx]);
              newnode->set_abstract(newnode->value()->ToAbstract());
              node->set_input(i, newnode);
              continue;  // do not visit the getitem again from this node
            }
          }
        }
        if (inp->isa<CNode>() && visited.count(inp) == 0) {
          (void)visited.insert(inp);
          que.push(inp->cast<CNodePtr>());
        }
      }
    }
  }

  void DumpResult(const std::string &name, const NodePtrList &inputs) const {
    auto fg = std::make_shared<FuncGraph>();
    std::map<AnfNodePtr, AnfNodePtr> node_map;
    CNodePtrList newcnodes;
    for (auto &inp : inputs) {
      auto p = fg->add_parameter();
      p->set_abstract(inp->get()->abstract());
      node_map[inp->get()] = p;
    }
    std::queue<CNodePtr> que;
    std::for_each(outputs_->cbegin(), outputs_->cend(), [&que](const CNodePtr &cnode) { que.push(cnode); });

    while (!que.empty()) {
      auto node = que.front();
      que.pop();
      if (node_map.count(node)) {
        continue;
      }
      auto new_node = fg->NewCNode(node->inputs());
      new_node->CloneCNodeInfo(node);
      new_node->set_fullname_with_scope(node->fullname_with_scope());
      node_map[node] = new_node;
      newcnodes.push_back(new_node);
      for (size_t i = 1; i < node->size(); ++i) {
        const auto &inp = node->input(i);
        if (inp->isa<CNode>() && node_map.count(inp) == 0) {
          que.push(inp->cast<CNodePtr>());
        }
      }
    }

    for (auto &cnode : newcnodes) {
      for (size_t i = 1; i < cnode->size(); i++) {
        if (node_map.count(cnode->input(i)) != 0) {
          cnode->set_input(i, node_map[cnode->input(i)]);
        }
      }
    }

    if (outputs_->size() == 1) {
      fg->set_output(node_map[(*outputs_)[0]]);
    } else {
      AnfNodePtrList new_outputs{NewValueNode(prim::kPrimMakeTuple)};
      AbstractBasePtrList abs;
      (void)std::transform(outputs_->cbegin(), outputs_->cend(), std::back_inserter(new_outputs),
                           [&node_map, &abs](const CNodePtr &node) {
                             abs.push_back(node->abstract());
                             return node_map[node];
                           });
      auto mt = fg->NewCNode(new_outputs);
      mt->set_abstract(std::make_shared<abstract::AbstractTuple>(abs));
      fg->set_output(mt);
    }
    DumpIR("bprop/bprop_expander_" + name + ".ir", fg, true);

    if (dout_user_ != nullptr) {
      for (auto &iter : *dout_user_) {
        MS_LOG(INFO) << "Dout User: " << iter.first->fullname_with_scope() << "  index: " << iter.second;
      }
    } else {  // users_ != nullptr
      for (auto &uiter : *users_) {
        for (auto &iter : uiter.second) {
          MS_LOG(INFO) << "Node " << uiter.first->ToString() << " user: " << iter.first->fullname_with_scope()
                       << "  index: " << iter.second;
        }
      }
    }
  }

 private:
  CNodePtrList *outputs_;
  DoutUserType *dout_user_;
  UserType *users_;
};
}  // namespace bprop
}  // namespace expander

// deprecated
void BuildBprop(const CNodePtr &cnode, CNodePtrList *outputs, DoutUserType *dout_user) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(dout_user);
  expander::bprop::BpropExpander e(outputs, dout_user, nullptr);
  (void)e.Run(cnode);
}

bool BuildBprop(const CNodePtr &cnode, CNodePtrList *outputs, UserType *users) {
  MS_LOG(DEBUG) << "Begin building bprop for " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(users);
  bool ret = true;
  try {
    expander::bprop::BpropExpander e(outputs, nullptr, users);
    ret = e.Run(cnode);
  } catch (const std::exception &e) {
    auto node_name = AnfUtils::GetCNodeName(cnode);
    MS_LOG(DEBUG) << "Bprop \"" << node_name << "\" encounter a problem: [" << e.what() << "]";
    MS_LOG(INFO) << "Python bprop will be used for \"" << node_name << "\"";
    outputs->clear();
    ret = false;
  }
  MS_LOG(DEBUG) << "Finish building bprop for " << cnode->fullname_with_scope();
  return ret;
}
}  // namespace mindspore
