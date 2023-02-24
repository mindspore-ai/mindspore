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
#include "pipeline/pynative/grad/bprop_expander/bprop.h"

#include <algorithm>
#include <queue>
#include <set>
#include "expander/infer.h"
#include "utils/anf_utils.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace expander {
namespace bprop {
bool BpropExpander::Run(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Begin building bprop for " << cnode->fullname_with_scope();
  bool ret = true;
  if (outputs_ != nullptr) {
    outputs_->clear();
  }
  try {
    ret = RunBprop(cnode);
  } catch (const std::exception &e) {
    auto node_name = AnfUtils::GetCNodeName(cnode);
    MS_LOG(DEBUG) << "Bprop \"" << node_name << "\" encounter a problem: [" << e.what() << "]";
    MS_LOG(INFO) << "Python bprop will be used for \"" << node_name << "\"";
    if (outputs_ != nullptr) {
      outputs_->clear();
    }
    ret = false;
  }
  MS_LOG(DEBUG) << "Finish building bprop for " << cnode->fullname_with_scope();
  return ret;
}

const std::vector<size_t> &BpropExpander::GetUnusedInputs(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  auto name = AnfUtils::GetCNodeName(cnode);
  auto handle = GetBpropHandle(name);
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << name << "] is not registered in bprop expander.";
    static std::vector<size_t> empty{};
    return empty;
  }
  return handle->unused_inputs;
}

void BpropExpander::ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder) {
  input_nodes_.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(input_nodes_),
                       [ir_builder](const AnfNodePtr &no) { return std::make_shared<Node>(no, ir_builder); });
}

std::unique_ptr<BpropIRBuilder> BpropExpander::CreateIRBuilder(const std::string &name, const CNodePtr &cnode) {
  auto infer = std::make_shared<CppInfer>();
  return std::make_unique<BpropIRBuilder>(name, cnode->func_graph(), infer);
}

bool BpropExpander::RunBprop(const CNodePtr &cnode) {
  auto name = AnfUtils::GetCNodeName(cnode);
  auto ir_builder = CreateIRBuilder(name, cnode);
  ExtractInputs(cnode, ir_builder.get());
  auto &attrs = GetCNodePrimitive(cnode)->attrs();
  auto handle = GetBpropHandle(name);
  if (handle == nullptr) {
    MS_LOG(DEBUG) << "Bprop IRBuilder [" << name << "] is not registered in bprop expander.";
    return false;
  }
  output_nodes_ = ir_builder->Run(input_nodes_, attrs, *handle);
  if (output_nodes_.empty()) {
    MS_LOG(DEBUG) << "The output nodes of bprop function [" << name << "] is empty.";
    return false;
  }
  PostProcess();
  DumpResult(name);
  input_nodes_.clear();
  return true;
}

void BpropExpander::PostProcess() const {
  outputs_->reserve(output_nodes_.size());
  (void)std::transform(output_nodes_.cbegin(), output_nodes_.cend(), std::back_inserter(*outputs_),
                       [](const NodePtr &node) {
                         auto cnode = node->get<CNodePtr>();
                         return cnode;
                       });
  std::set<AnfNodePtr> visited;
  // do not visit the inputs again.
  std::for_each(input_nodes_.cbegin(), input_nodes_.cend(),
                [&visited](const NodePtr &node) { visited.insert(node->get()); });

  std::queue<CNodePtr> que;
  std::for_each(outputs_->cbegin(), outputs_->cend(), [&que](const CNodePtr &cnode) { que.push(cnode); });

  AnfNodePtr dout = input_nodes_.back()->get();
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    for (size_t i = 1; i < node->size(); ++i) {
      const auto &inp = node->input(i);
      // record parameter's and dout's user
      if (users_ != nullptr) {
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

void BpropExpander::DumpResult(const std::string &name) const {
  static bool dump_result = (common::GetEnv("MS_DEV_DUMP_BPROP") == "on");
  if (!dump_result) {
    return;
  }
  auto fg = std::make_shared<FuncGraph>();
  std::map<AnfNodePtr, AnfNodePtr> node_map;
  CNodePtrList newcnodes;
  for (auto &inp : input_nodes_) {
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

  if (users_ != nullptr) {
    for (auto &uiter : *users_) {
      for (auto &iter : uiter.second) {
        auto user = iter.first.lock();
        if (user == nullptr) {
          continue;
        }
        MS_LOG(INFO) << "Node " << uiter.first->ToString() << " user: " << user->fullname_with_scope()
                     << "  index: " << iter.second;
      }
    }
  }
}

void BpropExpanderInGraphMode::ExtractInputs(const CNodePtr &cnode, const BpropIRBuilder *ir_builder) {
  input_nodes_.reserve(cnode->size());

  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(input_nodes_),
                       [ir_builder, this](const AnfNodePtr &no) {
                         auto p = this->fg_->add_parameter();
                         p->set_abstract(no->abstract());
                         return std::make_shared<Node>(p, ir_builder);
                       });
}

class LazyInfer : public CppInfer {
 public:
  void Infer(const NodePtr &node) override { return; }

  AbstractBasePtr GetAbstract(const NodePtr &node) override {
    auto anfnode = node->get();
    if (anfnode->abstract() == nullptr) {
      InferNow(anfnode);
    }
    return anfnode->abstract();
  }

 protected:
  void InferNow(const AnfNodePtr &node) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->size(); i++) {
        if (cnode->input(i)->abstract() == nullptr) {
          InferNow(cnode->input(i));
        }
      }
    }
    CppInfer::InferAnfnode(node);
  }
};

std::unique_ptr<BpropIRBuilder> BpropExpanderInGraphMode::CreateIRBuilder(const std::string &name,
                                                                          const CNodePtr &cnode) {
  fg_ = std::make_shared<FuncGraph>();
  ExpanderInferPtr infer;
  // default use LazyInfer in graph mode.
  static bool use_imm_infer = (common::GetEnv("MS_DEV_BPROP_IMM_INFER") == "on");
  if (use_imm_infer) {
    infer = std::make_shared<CppInfer>();
  } else {
    infer = std::make_shared<LazyInfer>();
  }
  return std::make_unique<BpropIRBuilder>(name, fg_, infer);
}

void BpropExpanderInGraphMode::PostProcess() const {
  auto mt = output_nodes_[0]->emitter()->MakeTuple(output_nodes_)->get();
  fg_->set_output(mt);

  // clear all abstract, to let the specializer re-infer the subgraph of controlflow graphs.
  auto todos = TopoSort(fg_->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (auto &no : todos) {
    no->set_abstract(nullptr);
    if (IsValueNode<FuncGraph>(no)) {
      auto fg = GetValueNode<FuncGraphPtr>(no);
      for (auto &p : fg->parameters()) {
        p->set_abstract(nullptr);
      }
    }
  }
}

void BpropExpanderInGraphMode::DumpResult(const std::string &name) const {
  static bool dump_result = (common::GetEnv("MS_DEV_DUMP_BPROP") == "on");
  if (!dump_result) {
    return;
  }
  DumpIR("bprop/bprop_expander_" + name + ".ir", fg_, true);
}

#ifdef _MSC_VER
void RegGradArrayOps();
void RegGradClipOps();
void RegGradCommOps();
void RegGradDebugOps();
void RegGradImageOps();
void RegGradImplementationsOps();
void RegGradInnerOps();
void RegGradLinalgOps();
void RegGradMathOps();
void RegGradNnOps();
void RegGradOtherOps();
void RegGradQuantOps();
void RegGradScipyOps();
void RegGradSparseOps();

WinBpropRegister::WinBpropRegister() {
  RegGradArrayOps();
  RegGradClipOps();
  RegGradCommOps();
  RegGradDebugOps();
  RegGradImageOps();
  RegGradImplementationsOps();
  RegGradInnerOps();
  RegGradLinalgOps();
  RegGradMathOps();
  RegGradNnOps();
  RegGradOtherOps();
  RegGradQuantOps();
  RegGradScipyOps();
  RegGradSparseOps();
}
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
