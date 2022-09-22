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

#include "backend/common/optimizer/pattern_to_pattern.h"
#include <algorithm>
#include "ir/manager.h"

namespace mindspore {
namespace opt {
bool BACKEND_EXPORT AlwaysReturnTrue(const BaseRef &n) { return true; }

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
  name_set_.insert(name);
  if (seq_map_.find(name) != seq_map_.end()) {
    MS_LOG(EXCEPTION) << "Var Key: " << name << " should not be in SeqVarMap.";
  }

  auto iter = node_map_.find(name);
  if (iter == node_map_.end()) {
    node_map_.emplace(name, node);
  } else if (!opt::AnfEqual(node, iter->second)) {
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

bool SrcPattern::match(const std::string &name, const AnfNodePtr &node, const EquivPtr &equiv) {
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
  if (inputs.size() - now_pattern == 1 && now_match == cnode_inputs.size() && inputs[now_pattern].type_ == "name") {
    auto &pattern_node = inputs[now_pattern];
    auto &ref = GetRef(pattern_node.name_);
    if (utils::isa<VarPtr>(ref) && utils::cast<VarPtr>(ref)->isa<SeqVar>()) {
      const Seq &seq = GetSeq(pattern_node.name_, name, utils::cast<VarPtr>(ref), equiv);
      MS_EXCEPTION_IF_CHECK_FAIL(seq.size() == IntToSize(0), "Match Failed, need zero seq, but get seq length: " +
                                                               std::to_string(seq.size()) + ", node name: " + name);
      std::vector<AnfNodePtr> v;
      m_->Emplace(pattern_node.name_, v);
      now_pattern++;
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
      if (!opt::AnfEqual(anf_inputs[i], cnode->input(i))) {
        MS_LOG(EXCEPTION) << "The actual input does not correspond to the input of the pattern, the input index: " << i
                          << ", actual input: " << anf_inputs[i]->fullname_with_scope()
                          << ", pattern input: " << new_node->cast<CNodePtr>()->input(i)->fullname_with_scope()
                          << ", CNode: " << name;
      }
    }
  }

  m_->Emplace(name, new_node);
  root_ = new_node;
  return *this;
}

DstPattern &DstPattern::AddValueNode(const string &name, const BuildValueFunc &buildfunc) {
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

AnfNodePtr PatternToPatternPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (src_pattern_root_ == nullptr) {
    DefineSrcPattern(&src_pattern_);
    VarPtr fg = std::make_shared<Var>("RootG");
    src_pattern_root_ = SexpToNode(src_pattern_.GetRoot(), fg, primitive_vars_.get(), multigraph_);
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

std::vector<UnpackNode> PatternToPatternPass::Unpacking(const std::string &s) {
  auto v = m_->GetSeq(s);
  std::vector<UnpackNode> ret;
  std::transform(v.begin(), v.end(), std::back_inserter(ret), [](const AnfNodePtr &node) { return UnpackNode(node); });
  return ret;
}
}  // namespace opt
}  // namespace mindspore
