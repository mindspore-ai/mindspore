/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SYMBOLIC_H_
#define MINDSPORE_CORE_UTILS_SYMBOLIC_H_

#include <unordered_map>
#include <memory>
#include <algorithm>
#include <utility>
#include <string>

#include "ir/anf.h"
#include "abstract/abstract_value.h"

namespace mindspore {
class SymbolicKeyInstance : public Value {
 public:
  SymbolicKeyInstance(const AnfNodePtr &node, const abstract::AbstractBasePtr &abstract)
      : node_(node), abstract_(abstract) {}
  ~SymbolicKeyInstance() override = default;
  MS_DECLARE_PARENT(SymbolicKeyInstance, Value);
  AnfNodePtr node() const { return node_; }
  abstract::AbstractBasePtr abstract() const { return abstract_; }
  bool operator==(const SymbolicKeyInstance &other) const {
    return (*node_ == *other.node_) && (*abstract_ == *other.abstract_);
  }

  std::size_t hash() const override { return std::hash<AnfNodePtr>{}(node_); }
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<SymbolicKeyInstance> &inst) {
    if (inst == nullptr) {
      os << "[Key]["
         << "Invalid symbolic key instance"
         << "]";
    } else {
      os << "[Key][" << inst->node_->type_name() << "]" << inst->node_->ToString();
    }
    return os;
  }
  std::string ToString() const override {
    return node_ == nullptr ? "Invalid node" : "[Key][" + node_->type_name() + "]" + node_->ToString();
  }
  bool operator==(const Value &other) const override {
    if (other.isa<SymbolicKeyInstance>()) {
      auto other_ = static_cast<const SymbolicKeyInstance &>(other);
      return *this == other_;
    } else {
      return false;
    }
  }
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<SymbolicKeyInstance>(),
                                                      std::make_shared<SymbolicKeyType>());
  }

 private:
  AnfNodePtr node_;
  abstract::AbstractBasePtr abstract_;
};

using SymbolicKeyInstancePtr = std::shared_ptr<SymbolicKeyInstance>;

struct SymbolicKeyInstanceHash {
  std::size_t operator()(const SymbolicKeyInstancePtr s) const {
    if (s == nullptr) {
      return 0;
    }
    return s->abstract()->hash();
  }
};

struct SymbolicKeyInstanceEqual {
  bool operator()(const SymbolicKeyInstancePtr lhs, const SymbolicKeyInstancePtr rhs) const {
    if (lhs == nullptr || rhs == nullptr) {
      return false;
    }
    MS_EXCEPTION_IF_NULL(lhs->node());
    MS_EXCEPTION_IF_NULL(rhs->node());
    MS_EXCEPTION_IF_NULL(lhs->abstract());
    MS_EXCEPTION_IF_NULL(rhs->abstract());
    return (*lhs->node() == *rhs->node()) && (*lhs->abstract() == *rhs->abstract());
  }
};

using EnvInstanceContentsMap =
  std::unordered_map<SymbolicKeyInstancePtr, Any, SymbolicKeyInstanceHash, SymbolicKeyInstanceEqual>;

// Environment mapping keys to values.
// Keys are SymbolicKeyInstances, which represent nodes in the graph along
// with inferred properties.
class EnvInstance : public Value {
 public:
  friend std::ostream &operator<<(std::ostream &out, const std::shared_ptr<EnvInstance> &env);

  explicit EnvInstance(const EnvInstanceContentsMap &contents = {}) : contents_(contents) {}
  ~EnvInstance() override = default;
  MS_DECLARE_PARENT(EnvInstance, Value);
  abstract::AbstractBasePtr ToAbstract() override {
    return std::make_shared<abstract::AbstractScalar>(shared_from_base<EnvInstance>(), std::make_shared<EnvType>());
  }
  bool operator==(const EnvInstance &other) const;
  bool operator==(const Value &other) const override;
  EnvInstance(const EnvInstance &v) : Value(v), contents_(v.contents_) {}
  EnvInstance(EnvInstance &&v) = default;
  EnvInstance &operator=(EnvInstance &&src) noexcept {
    if (&src != this) {
      contents_ = src.contents_;
    }
    return *this;
  };

  // Get the sensitivity list for the given key
  const Any &Get(const SymbolicKeyInstancePtr &key, const Any &def) const {
    auto iterator = contents_.find(key);
    if (iterator != contents_.end()) {
      return iterator->second;
    }
    return def;
  }

  // Set a value for the given key.
  EnvInstance Set(const SymbolicKeyInstancePtr &key, const Any &value) const {
    EnvInstance rval(contents_);
    rval.contents_[key] = value;
    return rval;
  }

  // Add two EnvInstances.
  EnvInstance Add(const EnvInstance &other) const {
    EnvInstance rval(contents_);
    for (auto iter_other : other.contents_) {
      auto item_self = contents_.find(iter_other.first);
      if (item_self != contents_.end()) {
        MS_LOG(DEBUG) << "Need to use add";
      } else {
        rval.contents_[iter_other.first] = iter_other.second;
      }
    }
    return rval;
  }

  size_t Len() const { return contents_.size(); }
  std::size_t hash() const override {
    // deterministic characteristic of member variables.
    return Len();
  }

 private:
  EnvInstanceContentsMap contents_;
};

using EnvInstancePtr = std::shared_ptr<EnvInstance>;

extern std::shared_ptr<EnvInstance> newenv;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_SYMBOLIC_H_
