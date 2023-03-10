/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "include/backend/optimizer/pattern_engine.h"
#include "frontend/optimizer/opt.h"
#include "ir/anf.h"
#include "utils/convert_utils_base.h"
#include "utils/overload.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
static int GetNextTag() {
  static int kID = 0;
  return kID++;
}

void Var::EnsureTag() {
  if (tag_.length() == 0) {
    std::ostringstream buffer;
    buffer << "_" << GetNextTag();
    tag_ = buffer.str();
  }
}

bool operator==(const VarPtr &lhs, const VarPtr &rhs) {
  if (lhs->isa<CondVar>() && rhs->isa<CondVar>()) {
    CondVarPtr v1 = dyn_cast<CondVar>(lhs);
    CondVarPtr v2 = dyn_cast<CondVar>(rhs);
    return *v1 == *v2;
  }

  if (lhs->isa<SeqVar>() && rhs->isa<SeqVar>()) {
    SVarPtr v1 = dyn_cast<SeqVar>(lhs);
    SVarPtr v2 = dyn_cast<SeqVar>(rhs);
    return *v1 == *v2;
  }
  return (*lhs == *rhs);
}

std::string SeqVar::ToString() const {
  std::ostringstream buffer;
  buffer << "SeqVar(" << tag() << ", " << subvar_->ToString() << ")";
  return buffer.str();
}

std::ostream &operator<<(std::ostream &os, const VarPtr &var) {
  if (var == nullptr) {
    os << "";
  } else {
    os << var->ToString();
  }
  return os;
}

template <>
std::ostream &operator<<<VarPtr, BaseRef>(std::ostream &os, const Equiv &equiv) {
  os << "[Equiv]"
     << "\n";
  for (auto &equiv_item : equiv) {
    auto k = equiv_item.first;
    os << k << ":";
    BaseRef x = equiv_item.second;
    if (utils::isa<AnfNodePtr>(x)) {
      auto node = utils::cast<AnfNodePtr>(x);
      os << "TypeString[" << node->type_name() << "]";
      if (IsValueNode<FuncGraph>(node)) {
        os << "IsValueNodeGraph ";
      }
      os << "type " << node->type_name();
      if (node->isa<ValueNode>()) {
        os << " value " << GetValueNode(node);
      }
      os << " addr: " << node;
    } else if (utils::isa<Named>(x)) {
      os << "Named " << x.ToString().c_str();
    } else if (utils::isa<VarPtr>(x)) {
      os << "TypeString[Var]";
      os << (utils::cast<VarPtr>(x));
    } else if (utils::isa<FuncGraphPtr>(x)) {
      os << "TypeString[Graph]";
    }
    os << "\n";
  }
  return os;
}

static BaseRef GetVar(const BaseRef &x) {
  MS_LOG(DEBUG) << "getVar start :%s" + x.ToString();
  if (utils::isa<AnfNodePtr>(x)) {
    auto node = utils::cast<AnfNodePtr>(x);
    MS_LOG(DEBUG) << "TypeString [" + node->type_name() + "]";
    if (node->isa<VarNode>()) {
      MS_LOG(DEBUG) << "IsVarNode " + node->cast<VarNodePtr>()->var_->ToString();
      return node->cast<VarNodePtr>()->var_;
    }
    if (node->isa<ValueNode>()) {
      MS_LOG(DEBUG) << "value " + GetValueNode(node)->ToString() + " addr: " + node->ToString();
    } else {
      MS_LOG(DEBUG) << "type " + node->type_name();
    }
  } else if (utils::isa<Named>(x)) {
    MS_LOG(DEBUG) << "Named " + x.ToString();
  } else if (utils::isa<VectorRef>(x)) {
    MS_LOG(DEBUG) << "VectorRef";
  } else if (utils::isa<VarPtr>(x)) {
    MS_LOG(DEBUG) << "TypeString[Var] " + x.ToString();
  }
  MS_LOG(DEBUG) << "GetVar end: " + x.ToString();
  return x;
}

EquivPtr MatchOnVar(const BaseRef &pattern, const BaseRef &expr, EquivPtr equiv) {
  MS_LOG(DEBUG) << "MatchOnVar pattern " + pattern.ToString() + " expr: " + expr.ToString();

  MS_EXCEPTION_IF_NULL(equiv);
  if (utils::isa<VarPtr>(pattern)) {
    VarPtr var = utils::cast<VarPtr>(pattern);
    if (var->matches(expr)) {
      (*equiv)[var] = expr;
      MS_LOG(DEBUG) << "pattern is var match: " + pattern.ToString() + ", " + expr.ToString();
      return equiv;
    } else {
      MS_LOG(DEBUG) << "pattern not match: " + pattern.ToString() + ", " + expr.ToString();
    }
  }
  return nullptr;
}

bool PatternEngine::ToVector(const VectorRef &pattern_ref, const VectorRef &expr_ref, VectorRef *const values_pattern,
                             VectorRef *const values_expr) const {
  MS_EXCEPTION_IF_NULL(values_expr);
  if (utils::isa<SeqPtr>(pattern_ref)) {
    *values_pattern = pattern_ref;
    *values_expr = expr_ref;
    return true;
  }
  return false;
}

bool PatternEngine::ToVector(const BaseRef &pattern_ref, const BaseRef &expr_ref, VectorRef *const values_pattern,
                             VectorRef *const values_expr) const {
  MS_EXCEPTION_IF_NULL(values_expr);
  MS_LOG(DEBUG) << "visit pattern_ref";
  bool success = visitor_->Visit(pattern_ref, values_pattern, nullptr);
  if (!success) {
    return false;
  }
  MS_LOG(DEBUG) << "visit expr_ref";
  return visitor_->Visit(expr_ref, values_expr, nullptr);
}

static int GetSVarStartIndex(const VectorRef &values) {
  int index = -1;
  int count = 0;
  for (auto &value : values) {
    if (utils::isa<VarPtr>(value) && utils::cast<VarPtr>(value)->isa<SeqVar>()) {
      if (index != -1) {
        MS_LOG(DEBUG) << "Multiple SVars in sequence";
        return kInvalidVarIndex;
      }
      index = count;
    }
    count++;
  }
  return index;
}

void UpdateEquivMap(const VectorRef &values_pattern, const BaseRef &expr_ref, const PrimitiveVarMap &primitive_vars,
                    const EquivPtr &equiv) {
  if (equiv == nullptr || values_pattern.empty() || !utils::isa<AnfNodePtr>(values_pattern[0]) ||
      !utils::isa<AnfNodePtr>(expr_ref)) {
    return;
  }
  auto real_node = utils::cast<AnfNodePtr>(expr_ref);
  MS_EXCEPTION_IF_NULL(real_node);
  if (!real_node->isa<CNode>()) {
    return;
  }
  auto prim_node = utils::cast<AnfNodePtr>(values_pattern[0]);
  MS_EXCEPTION_IF_NULL(prim_node);
  if (!IsValueNode<Primitive>(prim_node)) {
    return;
  }
  ValuePtr value = GetValueNode(prim_node);
  MS_EXCEPTION_IF_NULL(value);
  auto prim = value->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto iter = primitive_vars.find(prim);
  if (iter == primitive_vars.end()) {
    return;
  }
  (*equiv)[iter->second] = real_node;
}

EquivPtr PatternEngine::AlignSVar(const VectorRef &values_pattern, const VectorRef &values_expr,
                                  const PrimitiveVarMap &primitive_vars, EquivPtr equiv) const {
  int svar_index = GetSVarStartIndex(values_pattern);
  if (svar_index == kInvalidVarIndex) {
    return nullptr;
  }

  size_t values_pattern_len = values_pattern.size();
  size_t values_expr_len = values_expr.size();

  if (svar_index == -1) {
    if (values_pattern_len != values_expr_len) {
      MS_LOG(DEBUG) << "Structures of differing size: pattern len " << values_pattern_len << ", expr len "
                    << values_expr_len;
      return nullptr;
    }
  }
  if ((values_expr_len == 0) && (values_pattern_len == 0)) {
    return equiv;
  }
  if (values_expr_len < values_pattern_len - 1) {
    MS_LOG(DEBUG) << "invalid size: pattern len " << values_pattern_len << ", expr len " << values_expr_len;
    return nullptr;
  }
  size_t diff = values_expr_len - values_pattern_len + 1;
  for (size_t i = 0; i < values_pattern_len; i++) {
    size_t expr_i = i;
    if (svar_index != -1 && i == IntToSize(svar_index)) {
      auto seq =
        std::vector<BaseRef>(values_expr.begin() + svar_index, values_expr.begin() + svar_index + SizeToInt(diff));
      equiv = Match(values_pattern[svar_index], seq, primitive_vars, equiv);
    } else {
      if (svar_index != -1 && i > IntToSize(svar_index)) {
        expr_i = i + diff - 1;
      }
      equiv = Match(values_pattern[i], values_expr[expr_i], primitive_vars, equiv);
    }
    if (equiv == nullptr) {
      return nullptr;
    }
  }
  return equiv;
}

EquivPtr PatternEngine::Match(const BaseRef &pattern, const BaseRef &expr, const PrimitiveVarMap &primitive_vars,
                              EquivPtr equiv) const {
  MS_LOG(DEBUG) << "-----[in Match]";
  MS_LOG(DEBUG) << "GetVar w";
  BaseRef pattern_ref = GetVar(pattern);
  MS_LOG(DEBUG) << "GetVar v";
  BaseRef expr_ref = expr;

  if (equiv == nullptr) {
    MS_LOG(EXCEPTION) << "Equiv pointer is null";
  }

  MS_LOG(DEBUG) << "Pattern ref " + pattern_ref.ToString() + ", expr ref" + expr_ref.ToString();
  // 1. if pattern_ref is var and already in equiv, replace it.
  if (utils::isa<VarPtr>(pattern_ref)) {
    VarPtr var = utils::cast<VarPtr>(pattern_ref);
    auto iter = equiv->find(var);
    if (iter != equiv->end()) {
      pattern_ref = iter->second;
    }
  }

  // 2. check equal
  if (opt::AnfEqual(pattern_ref, expr_ref)) {
    return equiv;
  }

  // 3. match var
  EquivPtr ret_equiv = MatchOnVar(pattern_ref, expr_ref, equiv);
  if (ret_equiv) {
    return ret_equiv;
  }

  // 4. here the type can be std:vector, std:list,
  // or cnode.
  if (!PatternEngine::CNodeTypeEqual(pattern_ref, expr_ref)) {
    MS_LOG(DEBUG) << "Type mismatch";
    return nullptr;
  }

  // 5. transfer the Containers by visitor to std::vector
  VectorRef values_pattern;
  VectorRef values_expr;
  if (!ToVector(pattern_ref, expr_ref, &values_pattern, &values_expr)) {
    return nullptr;
  }

  // 6. if any svar in both side, find the SeqVar index,
  // try to pack the Var s in std::vector to a Seq and match elements one by one.
  // check svar
  equiv = AlignSVar(values_pattern, values_expr, primitive_vars, equiv);
  UpdateEquivMap(values_pattern, expr_ref, primitive_vars, equiv);
  return equiv;
}

bool PatternEngine::CNodeTypeEqual(const BaseRef &a, const BaseRef &b) {
  // To matchCNode and Kernel's type
  if (utils::isa<CNode>(a) && utils::isa<CNode>(b)) {
    return true;
  }
  return a.type() == b.type();
}
}  // namespace mindspore
