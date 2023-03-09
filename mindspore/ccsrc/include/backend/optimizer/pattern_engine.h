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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PATTERN_ENGINE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PATTERN_ENGINE_H_

#include <string>
#include <sstream>
#include <memory>
#include <vector>
#include <initializer_list>
#include <iostream>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <list>
#include <utility>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "base/base.h"
#include "utils/log_adapter.h"
#include "base/base_ref.h"
#include "include/backend/visible.h"
#include "include/backend/optimizer/visitor.h"

namespace mindspore {
class CondVar;
class SeqVar;
using CondVarPtr = std::shared_ptr<CondVar>;
using SVarPtr = std::shared_ptr<SeqVar>;
const int kInvalidVarIndex = -2;

using ConditionFunc = std::function<bool(const BaseRef &)>;

// Base wildcard variable which could match any anf node.
class BACKEND_EXPORT Var : public Base {
 public:
  explicit Var(std::string tag = "") : tag_(std::move(tag)), primitive_(nullptr) { EnsureTag(); }
  explicit Var(const PrimitivePtr &primitive, std::string tag = "") : tag_(std::move(tag)), primitive_(primitive) {
    EnsureTag();
  }
  Var(const Var &other) : Base(other), tag_(other.tag_), primitive_(other.primitive_) {}
  virtual Var &operator=(const Var &other) {
    if (&other == this) {
      return *this;
    }
    this->tag_ = other.tag_;
    this->primitive_ = other.primitive_;
    return *this;
  }
  ~Var() override = default;
  MS_DECLARE_PARENT(Var, Base);

  virtual bool matches(const BaseRef &) { return true; }

  virtual bool operator==(const Var &other) const { return tag_ == other.tag_; }
  bool operator!=(const Var &other) const { return !(&other == this); }

  std::string tag() const { return tag_; }
  PrimitivePtr primitive() const { return primitive_; }
  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << "Var(" << tag_ << ")";
    return buffer.str();
  }
  std::size_t hash() const override { return std::hash<std::string>()(tag_); }

 protected:
  void EnsureTag();

  std::string tag_;
  PrimitivePtr primitive_;
};

// VarNode means variable node, a subclass of AnfNode
class VarNode : public AnfNode {
 public:
  VarNode(const VarPtr &value, const FuncGraphPtr &func_graph) : AnfNode(func_graph), var_(value) {}
  ~VarNode() override = default;
  MS_DECLARE_PARENT(VarNode, AnfNode);

  const VarPtr var_;
};
using VarNodePtr = std::shared_ptr<VarNode>;

// Condition Var, match an anf node when condition function return true.
class CondVar : public Var {
 public:
  explicit CondVar(const ConditionFunc &cond) : cond_fn_(cond) {}
  explicit CondVar(const ConditionFunc &cond, std::string tag) : Var(tag), cond_fn_(cond) {}
  ~CondVar() override = default;
  MS_DECLARE_PARENT(CondVar, Var);
  bool matches(const BaseRef &value) override {
    MS_LOG(DEBUG) << "CondVarPtr match: " + value.ToString();
    if (utils::isa<Var>(value)) {
      return false;
    }
    return cond_fn_(value);
  }

 private:
  ConditionFunc cond_fn_;
};

using Seq = VectorRef;
using SeqPtr = std::shared_ptr<Seq>;

// Sequence Var which could match multiple consecutive input nodes of a CNode.
class BACKEND_EXPORT SeqVar : public Var {
 public:
  SeqVar() { subvar_ = std::make_shared<Var>(); }
  ~SeqVar() override = default;
  MS_DECLARE_PARENT(SeqVar, Var);
  explicit SeqVar(const VarPtr subvar) : subvar_(nullptr) { subvar_ = subvar; }
  explicit SeqVar(const ConditionFunc &cond) { subvar_ = std::make_shared<CondVar>(cond); }
  bool matches(const BaseRef &value) override {
    // match Seq.
    if (utils::isa<Seq>(value)) {
      const Seq &seq = utils::cast<Seq>(value);
      return std::all_of(seq.begin(), seq.end(), [this](const BaseRef &v) {
        auto eq = subvar_->matches(v);
        return eq;
      });
    }
    return false;
  }
  bool operator==(const SeqVar &other) const { return *subvar_ == *other.subvar_; }
  std::string ToString() const override;

 private:
  VarPtr subvar_;
};

bool operator==(const VarPtr &lhs, const VarPtr &rhs);

inline bool operator!=(const VarPtr &lhs, const VarPtr &rhs) { return !(lhs == rhs); }

std::ostream &operator<<(std::ostream &os, const VarPtr &var);

using Equiv = std::map<VarPtr, BaseRef>;
using EquivPtr = std::shared_ptr<Equiv>;
using PrimitiveVarMap = mindspore::HashMap<PrimitivePtr, VarPtr>;
using PrimitiveVarMapPtr = std::shared_ptr<PrimitiveVarMap>;

inline bool DefaultTypeEq(const BaseRef &x, const BaseRef &y) { return x.type() == y.type(); }

class PatternEngine {
 public:
  explicit PatternEngine(const std::shared_ptr<Visitor> &visitor) : visitor_(visitor) {}
  ~PatternEngine() = default;

  EquivPtr Match(const BaseRef &pattern, const BaseRef &expr, const PrimitiveVarMap &primitive_vars,
                 EquivPtr equiv) const;
  // Replace pattern with equivalent

 private:
  EquivPtr AlignSVar(const VectorRef &values_pattern, const VectorRef &values_expr,
                     const PrimitiveVarMap &primitive_vars, EquivPtr equiv) const;
  bool ToVector(const BaseRef &pattern, const BaseRef &expr, VectorRef *const values_pattern,
                VectorRef *const values_expr) const;
  bool ToVector(const VectorRef &pattern_ref, const VectorRef &expr_ref, VectorRef *const values_pattern,
                VectorRef *const values_expr) const;
  static bool CNodeTypeEqual(const BaseRef &a, const BaseRef &b);
  std::shared_ptr<Visitor> visitor_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_PATTERN_ENGINE_H_
