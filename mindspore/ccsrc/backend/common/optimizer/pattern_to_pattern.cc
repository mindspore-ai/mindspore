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

#include "include/backend/optimizer/pattern_to_pattern.h"
#include <algorithm>
#include <set>
#include <queue>
#include "ir/manager.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
bool AlwaysReturnTrue(const BaseRef &n) { return true; }

bool PatternMap::Contains(const std::string &name) const { return name_set_.count(name) > 0; }

bool PatternMap::CheckSeq(const std::string &name) const {
  return name_set_.count(name) > 0 && seq_map_.count(name) > 0;
}

void PatternMap::Erase(const mindspore::HashSet<std::string> &del_set) {
  for (auto &s : del_set) {
    name_set_.erase(s);
    node_map_.erase(s);
  }
}

AnfNodePtr PatternMap::Get(const std::string &name) const {
  if (!Contains(name)) {
    MS_LOG(EXCEPTION) << "Key: " << name << " is not in PatternMap";
  }

  auto iter = node_map_.find(name);
  if (iter == node_map_.end()) {
    MS_LOG(EXCEPTION) << "Var Key: " << name << " is not in PatternMap";
  }
  return iter->second;
}

const std::vector<AnfNodePtr> &PatternMap::GetSeq(const std::string &name) const {
  if (!Contains(name)) {
    MS_LOG(EXCEPTION) << "Key: " << name << " is not in PatternMap";
  }

  auto iter = seq_map_.find(name);
  if (iter == seq_map_.end()) {
    MS_LOG(EXCEPTION) << "SeqVar Key: " << name << " is not in PatternMap";
  }
  return iter->second;
}

bool PatternMap::Emplace(const std::string &name, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  name_set_.insert(name);
  if (seq_map_.find(name) != seq_map_.end()) {
    MS_LOG(EXCEPTION) << "Var Key: " << name << " should not be in SeqVarMap.";
  }

  opt_scope_.insert(node);

  auto iter = node_map_.find(name);
  if (iter == node_map_.end()) {
    node_map_.emplace(name, node);
  } else if (!opt::AnfEqual(node, iter->second)) {
    MS_EXCEPTION_IF_NULL(iter->second);
    MS_LOG(INFO) << "The value of key: " << name
                 << " is not equal to origin value, value: " + node->fullname_with_scope()
                 << " origin value: " << iter->second->fullname_with_scope();
    return false;
  }
  return true;
}

bool PatternMap::Emplace(const std::string &name, const std::vector<AnfNodePtr> &v) {
  name_set_.insert(name);
  if (node_map_.find(name) != node_map_.end()) {
    MS_LOG(EXCEPTION) << "SeqVar Key: " << name << " should not be in VarMap.";
  }

  for (const auto &node : v) {
    opt_scope_.insert(node);
  }

  auto iter = seq_map_.find(name);
  if (iter == seq_map_.end()) {
    seq_map_.emplace(name, v);
  } else {
    auto &origin_v = iter->second;
    if (v.size() != origin_v.size()) {
      MS_LOG(INFO) << "The value of key: " << name
                   << " is not equal to origin value, v size: " + std::to_string(v.size()) +
                        ", origin_v size: " + std::to_string(origin_v.size());
      return false;
    }

    for (size_t i = 0; i < v.size(); i++) {
      MS_EXCEPTION_IF_NULL(v[i]);
      MS_EXCEPTION_IF_NULL(origin_v[i]);
      if (!opt::AnfEqual(v[i], origin_v[i])) {
        MS_LOG(INFO) << "The value of key: " << name
                     << " is not equal to origin value, value: " + v[i]->fullname_with_scope()
                     << " origin value: " << origin_v[i]->fullname_with_scope();
        return false;
      }
    }
  }
  return true;
}

void PatternMap::Clear() {
  name_set_.clear();
  node_map_.clear();
  seq_map_.clear();
}

bool PatternMap::Check(const std::string &name, const AnfNodePtr &node) const { return opt::AnfEqual(node, Get(name)); }

SrcPattern &SrcPattern::AddVar(const std::string &name, const ConditionFunc &f) {
  if (ref_map_.find(name) != ref_map_.end()) {
    MS_LOG(EXCEPTION) << "Var: " << name << " is already in SrcPattern.";
  }

  auto var = std::make_shared<CondVar>(f);
  ref_map_.emplace(name, var);
  return *this;
}

SrcPattern &SrcPattern::AddSeqVar(const std::string &name, const ConditionFunc &f) {
  if (ref_map_.find(name) != ref_map_.end()) {
    MS_LOG(EXCEPTION) << "SeqVar: " << name << " is already in SrcPattern.";
  }

  auto seq_var = std::make_shared<SeqVar>(f);
  ref_map_.emplace(name, seq_var);
  return *this;
}

const BaseRef &SrcPattern::GetRef(const std::string &name) const {
  auto iter = ref_map_.find(name);
  if (iter == ref_map_.end()) {
    MS_LOG(EXCEPTION) << "Key: " << name << " not in PatternMap";
  }
  return iter->second;
}

SrcPattern &SrcPattern::AddCNode(const std::string &name, const std::initializer_list<PatternNode> &v) {
  if (ref_map_.find(name) != ref_map_.end()) {
    MS_LOG(EXCEPTION) << "CNode: " << name << " is already in SrcPattern.";
  }

  std::vector<BaseRef> ele;
  for (auto &node : v) {
    if (node.type_ == "name") {
      ele.emplace_back(GetRef(node.name_));
    } else if (node.type_ == "prim") {
      ele.emplace_back(node.p_);
    } else {
      MS_LOG(EXCEPTION) << "Error MatchNode Type: " << node.type_ << ", CNode: " << name;
    }
  }

  MS_EXCEPTION_IF_CHECK_FAIL(
    ele.size() == v.size(),
    "The length of BaseRef Vector and CNode Input is not equal, BaseRef Vector length: " + std::to_string(ele.size()) +
      " CNode Input length: " + std::to_string(v.size()) + ", CNode: " + name);

  inputs_map_.emplace(name, v);
  auto vec = VectorRef(ele);
  ref_map_.emplace(name, vec);
  has_root_ = true;
  root_ = name;
  return *this;
}

BaseRef SrcPattern::GetRoot() const {
  if (!has_root_) {
    MS_LOG(EXCEPTION) << "This SrcPattern has no root node.";
  }
  return GetRef(root_);
}

const Seq &GetSeq(const std::string &pattern_name, const std::string &node_name, const VarPtr &var,
                  const EquivPtr &equiv) {
  MS_EXCEPTION_IF_NULL(equiv);
  auto equiv_iter = equiv->find(var);
  if (equiv_iter == equiv->end()) {
    MS_LOG(EXCEPTION) << "The SeqVar Key: " << pattern_name << " is not in EquivMap, node name: " << node_name;
  }

  BaseRef &seq_ref = equiv_iter->second;
  if (utils::isa<Seq>(seq_ref)) {
    const Seq &seq = utils::cast<Seq>(seq_ref);
    return seq;
  }
  MS_LOG(EXCEPTION) << "The value of SeqVar Key: " << pattern_name << " is not a seq, node name: " << node_name;
}

bool SrcPattern::CheckEmptySeqVar(const std::string &name, const EquivPtr &equiv,
                                  const std::vector<PatternNode> &inputs, size_t *now_pattern) {
  if (inputs.size() - (*now_pattern) == 1 && inputs.at(*now_pattern).type_ == "name") {
    auto &pattern_node = inputs.at(*now_pattern);
    auto &ref = GetRef(pattern_node.name_);
    if (utils::isa<VarPtr>(ref) && utils::cast<VarPtr>(ref)->isa<SeqVar>()) {
      const Seq &seq = GetSeq(pattern_node.name_, name, utils::cast<VarPtr>(ref), equiv);
      MS_EXCEPTION_IF_CHECK_FAIL(seq.size() == IntToSize(0), "Match Failed, need zero seq, but get seq length: " +
                                                               std::to_string(seq.size()) + ", node name: " + name);
      std::vector<AnfNodePtr> v;
      MS_EXCEPTION_IF_NULL(m_);
      if (!m_->Emplace(pattern_node.name_, v)) {
        return false;
      }
      (*now_pattern)++;
    }
  }
  return true;
}

bool SrcPattern::match(const std::string &name, const AnfNodePtr &node, const EquivPtr &equiv) {
  MS_EXCEPTION_IF_NULL(m_);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto input_iter = inputs_map_.find(name);
  if (input_iter == inputs_map_.end()) {
    MS_LOG(EXCEPTION) << "Key: " << name << " is not a CNode.";
  }

  if (m_->Contains(name)) {
    return m_->Check(name, node);
  }

  auto &inputs = input_iter->second;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_inputs = cnode->inputs();
  size_t now_pattern = 0;
  size_t now_match = 0;
  for (; now_pattern < inputs.size() && now_match < cnode_inputs.size(); now_pattern++, now_match++) {
    auto &pattern_node = inputs[now_pattern];
    auto &match_node = cnode_inputs[now_match];
    if (pattern_node.type_ == "prim") {
      // prim
      MS_EXCEPTION_IF_NULL(pattern_node.p_);
      MS_EXCEPTION_IF_NULL(match_node);
      if (!opt::AnfEqual(pattern_node.p_, match_node)) {
        MS_LOG(EXCEPTION) << "The value of Primitive is not equal to matched value, pattern value: " +
                               pattern_node.p_->ToString()
                          << " matched value: " + match_node->ToString() + ", node name: " + name;
      }
      continue;
    }
    // name
    MS_EXCEPTION_IF_CHECK_FAIL(pattern_node.type_ == "name",
                               "Error MatchNode Type: " + pattern_node.type_ + ", node name: " + name);
    auto &ref = GetRef(pattern_node.name_);
    if (utils::isa<VarPtr>(ref)) {
      if (utils::cast<VarPtr>(ref)->isa<SeqVar>()) {
        // seq var
        const Seq &seq = GetSeq(pattern_node.name_, name, utils::cast<VarPtr>(ref), equiv);
        std::vector<AnfNodePtr> v;
        for (size_t i = 0; i < seq.size(); i++) {
          v.emplace_back(cnode_inputs.at(now_match + i));
        }
        if (!m_->Emplace(pattern_node.name_, v)) {
          return false;
        }
        now_match += seq.size() - 1;
        continue;
      }
    } else {
      // cnode
      if (!match(pattern_node.name_, match_node, equiv)) {
        return false;
      }
    }
    if (!m_->Emplace(pattern_node.name_, match_node)) {
      return false;
    }
  }
  // has a SeqVar at the end
  if (now_match == cnode_inputs.size()) {
    if (!CheckEmptySeqVar(name, equiv, inputs, &now_pattern)) {
      return false;
    }
  }

  MS_EXCEPTION_IF_CHECK_FAIL(
    now_pattern == inputs.size() && now_match == cnode_inputs.size(),
    "Match Failed, now_pattern: " + std::to_string(now_pattern) + ", inputs.size(): " + std::to_string(inputs.size()) +
      ", now_match: " + std::to_string(now_match) + ", cnode_inputs.size(): " + std::to_string(cnode_inputs.size()) +
      ", node name: " + name);

  return m_->Emplace(name, node);
}

bool SrcPattern::build_pattern_map(const AnfNodePtr &node, const EquivPtr &equiv) {
  MS_EXCEPTION_IF_NULL(m_);
  if (!has_root_) {
    MS_LOG(EXCEPTION) << "This SourcePattern has no root node.";
  }
  m_->Clear();
  return match(root_, node, equiv);
}

DstPattern &DstPattern::AddCNode(const string &name, const std::initializer_list<PatternNode> &inputs,
                                 const BuildCNodeFunc &buildfunc) {
  MS_EXCEPTION_IF_NULL(m_);
  if (fail_) {
    return *this;
  }

  if (m_->Contains(name)) {
    MS_LOG(EXCEPTION) << "CNode: " + name + " is already in DstPattern";
  }

  std::vector<AnfNodePtr> anf_inputs;
  for (auto &r : inputs) {
    if (r.type_ == "prim") {
      anf_inputs.emplace_back(r.p_);
    } else if (r.type_ == "name") {
      if (m_->CheckSeq(r.name_)) {
        auto &v = m_->GetSeq(r.name_);
        std::copy(v.begin(), v.end(), std::back_inserter(anf_inputs));
      } else {
        anf_inputs.emplace_back(m_->Get(r.name_));
      }
    } else if (r.type_ == "unpack") {
      for (auto &it : r.v_) {
        if (it.node_ == nullptr) {
          anf_inputs.emplace_back(m_->Get(it.key_));
        } else {
          anf_inputs.emplace_back(it.node_);
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Error ReplaceNode Type: " << r.type_ << ", CNode: " << name;
    }
  }

  MS_EXCEPTION_IF_NULL(pass_);
  auto default_node = pass_->NewCNode(anf_inputs, fg_);
  auto new_node = buildfunc(*m_, default_node);
  if (new_node == nullptr) {
    fail_ = true;
  } else {
    auto cnode = new_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (anf_inputs.size() != cnode->inputs().size()) {
      MS_LOG(EXCEPTION)
        << "The actual input size does not correspond to the input size of the pattern, actual input size: "
        << anf_inputs.size() << ", pattern input size: " << new_node->cast<CNodePtr>()->inputs().size()
        << ", CNode: " << name;
    }
    for (size_t i = 0; i < anf_inputs.size(); i++) {
      MS_EXCEPTION_IF_NULL(anf_inputs[i]);
      MS_EXCEPTION_IF_NULL(cnode->input(i));
      if (!opt::AnfEqual(anf_inputs[i], cnode->input(i))) {
        MS_LOG(EXCEPTION) << "The actual input does not correspond to the input of the pattern, the input index: " << i
                          << ", actual input: " << anf_inputs[i]->DebugString()
                          << ", pattern input: " << new_node->cast<CNodePtr>()->input(i)->DebugString()
                          << ", CNode: " << name;
      }
    }
  }

  if (!m_->Emplace(name, new_node)) {
    MS_LOG(EXCEPTION) << "CNode: " + name + " is already in DstPattern";
  }
  root_ = new_node;
  return *this;
}

DstPattern &DstPattern::AddValueNode(const string &name, const BuildValueFunc &buildfunc) {
  MS_EXCEPTION_IF_NULL(m_);
  if (fail_) {
    return *this;
  }

  if (m_->Contains(name)) {
    MS_LOG(EXCEPTION) << "ValueNode: " + name + " is already in DstPattern";
  }

  auto node = buildfunc(*m_);
  if (node == nullptr) {
    fail_ = true;
  }
  dst_set_.insert(name);
  m_->Emplace(name, node);
  root_ = node;
  return *this;
}

void DstPattern::clear() {
  MS_EXCEPTION_IF_NULL(m_);
  fail_ = false;
  root_ = nullptr;
  m_->Erase(dst_set_);
  dst_set_.clear();
  fg_ = nullptr;
  pass_ = nullptr;
}

void DstPattern::set_info(PatternToPatternPass *now_pass, const FuncGraphPtr &func_graph) {
  pass_ = now_pass;
  fg_ = func_graph;
}

AnfNodePtr DstPattern::Root() {
  if (fail_) {
    return nullptr;
  } else {
    return root_;
  }
}

UnpackNode &UnpackNode::operator=(const std::string &name) {
  key_ = name;
  node_ = nullptr;
  return *this;
}

AnfNodePtr PatternToPatternPass::GetSrcPatternRoot() {
  if (src_pattern_root_ == nullptr) {
    DefineSrcPattern(&src_pattern_);
    VarPtr fg = std::make_shared<Var>("RootG");
    src_pattern_root_ = SexpToNode(src_pattern_.GetRoot(), fg, primitive_vars_.get(), multigraph_);
  }
  return src_pattern_root_;
}

std::string PatternToPatternPass::GetPatternRootPrimitiveName() {
  auto src_pattern_root = GetSrcPatternRoot();
  auto prim = GetCNodePrimitive(src_pattern_root);
  if (prim != nullptr) {
    return prim->name();
  }
  return "";
}

AnfNodePtr PatternToPatternPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (src_pattern_root_ == nullptr) {
    (void)GetSrcPatternRoot();
  }

  auto primitive = GetCNodePrimitive(src_pattern_root_);
  if (IsPrimitiveCNode(node, primitive)) {
    MS_EXCEPTION_IF_NULL(primitive_vars_);
    MS_EXCEPTION_IF_NULL(equiv_);
    equiv_->clear();
    EquivPtr equiv = pattern_engine_.Match(src_pattern_root_, node, *primitive_vars_, equiv_);
    if (equiv != nullptr && !equiv->empty()) {
      if (!src_pattern_.build_pattern_map(node, equiv)) {
        return nullptr;
      }
      if (!CheckMatchedDAG(*m_, func_graph, node)) {
        return nullptr;
      }
      dst_pattern_.clear();
      dst_pattern_.set_info(this, func_graph);
      DefineDstPattern(&dst_pattern_);
      return dst_pattern_.Root();
    }
  }
  return nullptr;
}

namespace {
const auto kStageZero = 0;
const auto kStageOne = 1;
const auto kStageTwo = 2;

void DeleteCNode(const AnfNodePtr &node, const FuncGraphPtr &sub_graph, const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  if (node->isa<CNode>()) {
    auto name_to_cnode_iter = func_graph_index->name_to_cnode_.find(GetCNodeKey(node));
    if (name_to_cnode_iter == func_graph_index->name_to_cnode_.end()) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, name_to_cnode_ can't find cnode_name: "
                        << common::AnfAlgo::GetCNodeName(node);
    }
    auto &cnode_set = name_to_cnode_iter->second;
    auto cnode_set_iter = cnode_set.find(node);
    if (cnode_set_iter == cnode_set.end()) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, name_to_cnode_ can't find node: " << node->fullname_with_scope();
    }
    cnode_set.erase(cnode_set_iter);
    ModifyOutputAndCallerToMap(node->cast<CNodePtr>(), sub_graph, &func_graph_index->subgraph_out_caller_map_, false);
  }
}

void AppendChild(const AnfNodePtr &node, const FuncGraphPtr &fg,
                 std::queue<std::pair<AnfNodePtr, FuncGraphPtr>> *anf_q) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(anf_q);
  if (IsValueNode<FuncGraph>(node)) {
    auto const_func_graph = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(const_func_graph);
    if (!const_func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
      anf_q->emplace(const_func_graph->output(), const_func_graph);
    }
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (const auto &input_node : cnode->inputs()) {
      anf_q->emplace(input_node, fg);
    }
  }
}

bool DelSrcPattern(const std::pair<AnfNodePtr, FuncGraphPtr> &top, const AnfNodePtr &root,
                   const mindspore::HashSet<AnfNodePtr> &opt_scope,
                   std::set<std::pair<AnfNodePtr, FuncGraphPtr>> *need_delete,
                   const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(need_delete);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  auto node = top.first;
  auto fg = top.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(fg);
  if (node != root) {
    auto degree_iter = func_graph_index->node_degree_.find(node);
    if (degree_iter == func_graph_index->node_degree_.end()) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, node: " << node->fullname_with_scope() << " not in degree map";
    }
    if (degree_iter->second <= 0) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, node: " << node->fullname_with_scope()
                        << " degree error, degree: " << degree_iter->second;
    }
    degree_iter->second--;
    if (degree_iter->second > 0) {
      return false;
    }
  }
  if (opt_scope.find(node) == opt_scope.end()) {
    (*need_delete).insert({node, fg});
    return false;
  }

  DeleteCNode(node, fg, func_graph_index);
  return true;
}

bool AddDstPattern(const std::pair<AnfNodePtr, FuncGraphPtr> &top, const AnfNodePtr &root,
                   const mindspore::HashSet<AnfNodePtr> &opt_scope,
                   std::set<std::pair<AnfNodePtr, FuncGraphPtr>> *need_delete,
                   const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(need_delete);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  auto node = top.first;
  auto fg = top.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(fg);
  if (node->isa<CNode>()) {
    ModifyOutputAndCallerToMap(node->cast<CNodePtr>(), fg, &func_graph_index->subgraph_out_caller_map_);
    func_graph_index->name_to_cnode_[GetCNodeKey(node)].insert(node);
    func_graph_index->node_to_fg_[node] = fg;
  }

  if (node != root) {
    auto degree_iter = func_graph_index->node_degree_.find(node);
    if (degree_iter == func_graph_index->node_degree_.end()) {
      func_graph_index->node_degree_[node] = 0;
      degree_iter = func_graph_index->node_degree_.find(node);
    }
    degree_iter->second++;
    if (degree_iter->second != 1) {
      return false;
    }
  }
  if (opt_scope.find(node) == opt_scope.end()) {
    (*need_delete).erase({node, fg});
    return false;
  }
  return true;
}

bool DelCascadeNode(const std::pair<AnfNodePtr, FuncGraphPtr> &top,
                    std::set<std::pair<AnfNodePtr, FuncGraphPtr>> *need_delete,
                    const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(need_delete);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  auto node = top.first;
  auto fg = top.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(fg);
  if ((*need_delete).find({node, fg}) == (*need_delete).end()) {
    auto degree_iter = func_graph_index->node_degree_.find(node);
    if (degree_iter == func_graph_index->node_degree_.end()) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, node: " << node->fullname_with_scope() << " not in degree map";
    }
    if (degree_iter->second <= 0) {
      MS_LOG(EXCEPTION) << "ProcessFastPass Error, node: " << node->fullname_with_scope()
                        << " degree error, degree: " << degree_iter->second;
    }
    degree_iter->second--;
    if (degree_iter->second > 0) {
      return false;
    }
  }

  DeleteCNode(node, fg, func_graph_index);
  return true;
}

void BFS(const AnfNodePtr &root, const FuncGraphPtr &sub_graph, const mindspore::HashSet<AnfNodePtr> &opt_scope,
         std::set<std::pair<AnfNodePtr, FuncGraphPtr>> *need_delete, const FuncGraphIndexPtr &func_graph_index,
         size_t stage) {
  std::queue<std::pair<AnfNodePtr, FuncGraphPtr>> anf_q;

  if (stage == kStageZero || stage == kStageOne) {
    anf_q.emplace(root, sub_graph);
  } else if (stage == kStageTwo) {
    for (const auto &p : (*need_delete)) {
      anf_q.push(p);
    }
  } else {
    MS_LOG(EXCEPTION) << "Illegal BFS stage, expected stage is 0/1/2, but get stage: " << stage;
  }

  while (!anf_q.empty()) {
    auto top = anf_q.front();
    anf_q.pop();

    bool ret = false;
    if (stage == kStageZero) {
      ret = DelSrcPattern(top, root, opt_scope, need_delete, func_graph_index);
    } else if (stage == kStageOne) {
      ret = AddDstPattern(top, root, opt_scope, need_delete, func_graph_index);
    } else if (stage == kStageTwo) {
      ret = DelCascadeNode(top, need_delete, func_graph_index);
    } else {
      MS_LOG(EXCEPTION) << "Illegal BFS stage, expected stage is 0/1/2, but get stage: " << stage;
    }
    if (!ret) {
      continue;
    }

    AppendChild(top.first, top.second, &anf_q);
  }
}
}  // namespace

void PatternToPatternPass::AfterProcess(const AnfNodePtr &old_node, const AnfNodePtr &new_node,
                                        const FuncGraphPtr &sub_graph, const FuncGraphIndexPtr &func_graph_index) {
  MS_EXCEPTION_IF_NULL(m_);
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  MS_EXCEPTION_IF_NULL(sub_graph);
  MS_EXCEPTION_IF_NULL(func_graph_index);
  std::set<std::pair<AnfNodePtr, FuncGraphPtr>> need_delete;
  auto &opt_scope = m_->GetOptScope();

  auto old_node_iter = func_graph_index->node_degree_.find(old_node);
  if (old_node_iter == func_graph_index->node_degree_.end()) {
    MS_LOG(EXCEPTION) << "ProcessFastPass Error, old_node: " << old_node->fullname_with_scope() << " not in degree map";
  }
  auto origin_degree = old_node_iter->second;

  func_graph_index->node_degree_[new_node] = origin_degree;
  func_graph_index->node_degree_[old_node] = 0;

  BFS(old_node, sub_graph, opt_scope, &need_delete, func_graph_index, kStageZero);
  BFS(new_node, sub_graph, opt_scope, &need_delete, func_graph_index, kStageOne);
  BFS(new_node, sub_graph, opt_scope, &need_delete, func_graph_index, kStageTwo);
}

std::vector<UnpackNode> PatternToPatternPass::Unpacking(const std::string &s) {
  MS_EXCEPTION_IF_NULL(m_);
  auto v = m_->GetSeq(s);
  std::vector<UnpackNode> ret;
  std::transform(v.begin(), v.end(), std::back_inserter(ret), [](const AnfNodePtr &node) { return UnpackNode(node); });
  return ret;
}

bool PatternToPatternPass::IsFastPass() { return is_fast_pass_; }
}  // namespace opt
}  // namespace mindspore
