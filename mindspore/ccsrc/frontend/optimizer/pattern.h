/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PATTERN_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PATTERN_H_
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include "base/base.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "pybind_api/ir/primitive_py.h"
#include "pybind_api/ir/tensor_py.h"

namespace mindspore {
namespace opt {
namespace python_pass {
using std::string;
using std::vector;

class MatchResult;
using MatchResultPtr = std::shared_ptr<MatchResult>;
class Pattern;
using PatternPtr = std::shared_ptr<Pattern>;
class Prim;
using PrimPtr = std::shared_ptr<Prim>;
class Call;
using CallPtr = std::shared_ptr<Call>;
class NewTensor;
using NewTensorPtr = std::shared_ptr<NewTensor>;
class NewParameter;
using NewParameterPtr = std::shared_ptr<NewParameter>;
class Imm;
using ImmPtr = std::shared_ptr<Imm>;
struct PatternHasher;
struct PatternEqual;
using PatternNodeMap = std::unordered_map<PatternPtr, AnfNodePtr, PatternHasher, PatternEqual>;

class Pattern : public Base {
 public:
  Pattern() : unique_name_(std::to_string(g_id_++)) {}
  ~Pattern() = default;
  virtual MatchResultPtr match(const AnfNodePtr &node) { return nullptr; }
  virtual bool operator==(const Pattern &other) const { return unique_name_ == other.unique_name_; }
  string unique_name() const { return unique_name_; }
  vector<PatternPtr> inputs() { return inputs_; }
  virtual void reset() {}
  static void reset_gid() { g_id_ = 0; }

 protected:
  static int64_t g_id_;
  // NOTE: To ensure uniqueness of the name, raise g_id_ by 1 every time a pattern got constructed
  string unique_name_;
  vector<PatternPtr> inputs_;
};

struct PatternEqual {
  bool operator()(PatternPtr const &p1, PatternPtr const &p2) const {
    MS_EXCEPTION_IF_NULL(p1);
    MS_EXCEPTION_IF_NULL(p2);
    return p1->unique_name() == p2->unique_name();
  }
};

struct PatternHasher {
  std::size_t operator()(PatternPtr const &p) const {
    MS_EXCEPTION_IF_NULL(p);
    return std::hash<string>()(p->unique_name());
  }
};

class Prim : public Pattern {
 public:
  Prim() { unique_name_ = std::to_string(g_id_++); }
  ~Prim() = default;
  Prim(vector<py::object> prim_objs, string name) : name_(name) {
    unique_name_ = std::to_string(g_id_++) + "Prim_" + name;
    for (auto &prim_obj : prim_objs) {
      if (py::isinstance<PrimitivePyAdapter>(prim_obj)) {
        auto prim_adapter = prim_obj.cast<PrimitivePyAdapterPtr>();
        primitives_.push_back(std::make_shared<PrimitivePy>(prim_obj, prim_adapter));
      } else if (py::isinstance<py::str>(prim_obj)) {
        std::string prim_name = prim_obj.cast<py::str>();
        primitives_.push_back(std::make_shared<PrimitivePy>(prim_name));
      } else {
        MS_LOG(EXCEPTION) << "Parameter of Prim::__init__ must be Primitive_ type or Prim name, please check input.";
      }
    }
    // Default using the first prim to build target
    matched_prim_ = primitives_[0];
  }
  MS_DECLARE_PARENT(Prim, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;
  PrimitivePyPtr matched_primitive() { return matched_prim_; }
  void reset() override {
    // Init before reset
    MS_EXCEPTION_IF_NULL(matched_prim_);
    matched_prim_ = primitives_[0];
  }

 private:
  vector<PrimitivePyPtr> primitives_;
  string name_;
  PrimitivePyPtr matched_prim_{nullptr};
};

class Call : public Pattern {
 public:
  Call() { unique_name_ = std::to_string(g_id_++); }
  ~Call() = default;
  Call(PatternPtr prim_pattern, vector<PatternPtr> inputs) {
    // NOTE: should_replace is ignored in this case, since each sub-pattern has its own setting
    prim_pattern_ = prim_pattern;
    unique_name_ = std::to_string(g_id_++) + "Call_" + prim_pattern->unique_name();
    inputs_ = inputs;
  }
  Call(py::object prim_obj, vector<PatternPtr> inputs) {
    if (py::isinstance<PrimitivePyAdapter>(prim_obj)) {
      auto prim_adapter = prim_obj.cast<PrimitivePyAdapterPtr>();
      prim_ = std::make_shared<PrimitivePy>(prim_obj, prim_adapter);
    } else if (py::isinstance<py::str>(prim_obj)) {
      std::string prim_name = prim_obj.cast<py::str>();
      prim_ = std::make_shared<PrimitivePy>(prim_name);
    } else {
      MS_LOG(EXCEPTION) << "Parameter of Call::__init__ must be Primitive_ type or Prim name, please check input.";
    }
    unique_name_ = std::to_string(g_id_++) + "Call_" + prim_->ToString();
    inputs_ = inputs;
  }
  MS_DECLARE_PARENT(Call, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;
  PrimitivePtr prim_value() { return prim_; }
  PatternPtr prim_pattern() { return prim_pattern_; }

 private:
  PatternPtr prim_pattern_ = nullptr;
  PrimitivePtr prim_ = nullptr;
  vector<string> types_;
  string name_;
};

class OneOf : public Pattern {
 public:
  OneOf() { unique_name_ = std::to_string(g_id_++); }
  ~OneOf() = default;
  explicit OneOf(vector<PatternPtr> patterns) : patterns_(patterns) {
    unique_name_ = std::to_string(g_id_++) + "OneOf";
    for (auto &iter : patterns) {
      unique_name_ = unique_name_ + "_" + iter->unique_name();
    }
  }
  MS_DECLARE_PARENT(OneOf, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;

 private:
  vector<PatternPtr> patterns_;
};

class NoneOf : public Pattern {
 public:
  NoneOf() { unique_name_ = std::to_string(g_id_++); }
  ~NoneOf() = default;
  explicit NoneOf(vector<PatternPtr> patterns) : patterns_(patterns) {
    unique_name_ = std::to_string(g_id_++) + "NoneOf";
    for (auto &iter : patterns) {
      unique_name_ = unique_name_ + "_" + iter->unique_name();
    }
  }
  MS_DECLARE_PARENT(NoneOf, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;

 private:
  vector<PatternPtr> patterns_;
};

class Any : public Pattern {
 public:
  Any() { unique_name_ = std::to_string(g_id_++) + "_Any"; }
  ~Any() = default;
  MS_DECLARE_PARENT(Any, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;
};

class NewTensor : public Pattern {
 public:
  NewTensor() { unique_name_ = std::to_string(g_id_++); }
  ~NewTensor() = default;
  explicit NewTensor(tensor::TensorPtr input_tensor) : input_tensor_(input_tensor) {
    unique_name_ = std::to_string(g_id_++) + "NewTensor";
  }
  MS_DECLARE_PARENT(NewTensor, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override {
    MS_LOG(EXCEPTION) << "Find NewTensor in pattern, NewTensor should only appear in the target.\n";
  }
  tensor::TensorPtr input_tensor() { return input_tensor_; }

 private:
  tensor::TensorPtr input_tensor_;
};

class NewParameter : public Pattern {
 public:
  NewParameter() { unique_name_ = std::to_string(g_id_++); }
  explicit NewParameter(string para_name, tensor::TensorPtr default_tensor, bool requires_grad, bool layerwise_parallel)
      : para_name_(para_name), requires_grad_(requires_grad), layerwise_parallel_(layerwise_parallel) {
    unique_name_ = std::to_string(g_id_++) + "NewParameter_" + para_name;
    default_tensor_ = std::make_shared<tensor::Tensor>(*default_tensor.get());
    built_ = false;
  }
  ~NewParameter() = default;
  MS_DECLARE_PARENT(NewParameter, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override {
    MS_LOG(EXCEPTION) << "Find NewParameter in pattern, NewParameter should only appear in the target.\n";
  }
  string para_name() { return para_name_; }
  tensor::TensorPtr default_tensor() { return default_tensor_; }
  bool requires_grad() { return requires_grad_; }
  bool layerwise_parallel() { return layerwise_parallel_; }
  bool built() { return built_; }
  void set_built(bool built) { built_ = built; }
  void reset() override { built_ = false; }
  bool should_last() { return last_across_passes_; }
  void set_last(bool last) { last_across_passes_ = last; }

 private:
  string para_name_;
  bool requires_grad_;
  bool layerwise_parallel_;
  bool last_across_passes_{false};
  bool built_;
  tensor::TensorPtr default_tensor_;
};

class Imm : public Pattern {
 public:
  Imm() { unique_name_ = std::to_string(g_id_++); }
  explicit Imm(int value) : value_(value) { unique_name_ = std::to_string(g_id_++) + "Imm_" + std::to_string(value); }
  ~Imm() = default;
  MS_DECLARE_PARENT(Imm, Pattern);
  MatchResultPtr match(const AnfNodePtr &node) override;
  int value() { return value_; }

 private:
  int64_t value_;
};

class MatchResult {
 public:
  MatchResult() {}
  ~MatchResult() = default;
  void add_entry(PatternPtr pattern, AnfNodePtr node) { match_result_[pattern] = node; }
  const PatternNodeMap &result() { return match_result_; }
  AnfNodePtr get_node(const PatternPtr &pattern);
  void merge(const MatchResultPtr &other_result);
  void clear() { match_result_.clear(); }
  void dump() {
    MS_LOG(DEBUG) << "match_result_.size: " + std::to_string(match_result_.size()) + "\n";
    for (auto &iter : match_result_) {
      MS_LOG(DEBUG) << "Pattern : " + iter.first->unique_name() + " , node : " + iter.second->ToString() + "\n";
    }
  }

 private:
  PatternNodeMap match_result_;
};
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PATTERN_H_
