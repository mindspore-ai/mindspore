/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <functional>

#include "include/common/expander/core/node.h"
#include "include/common/expander/core/emitter.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace expander {
namespace bprop {
class BpropIRBuilder;

using BpropIRBuilderFunc = std::function<NodePtrList(BpropIRBuilder *)>;
struct BpropHandle {
  BpropIRBuilderFunc func;
  mindspore::HashSet<size_t> unused_inputs;
};

class BpropIRBuilder : public Emitter {
 public:
  BpropIRBuilder(const std::string &name, const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer)
      : Emitter(func_graph, infer, std::make_shared<Scope>(std::string("Bprop/grad") + name)), name_(name) {}

  /// \brief Run irbuilder to generate a graph
  NodePtrList Run(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
                  const BpropHandle &handle, const std::string &instance_name);

  ValuePtr GetAttr(const std::string &attr) const;
  template <typename S>
  S GetAttr(const std::string &attr) const {
    return GetValue<S>(GetAttr(attr));
  }
  const mindspore::HashMap<std::string, ValuePtr> &GetAttrs() const { return *attrs_ptr_; }
  NodePtr GetInput(size_t i) const;
  const NodePtrList &GetInputs() const { return *inputs_ptr_; }

  NodePtrList BroadcastGradientArgs(const NodePtr &s0, const NodePtr &s1, size_t shift = 0);

  // For node that has single output
  ShapeVector GetShape(const NodePtr &node) const { return node->shape(); }
  size_t GetRank(const NodePtr &node) const { return GetShape(node).size(); }
  // For node that has multiple outputs
  std::vector<ShapeVector> GetShapes(const NodePtr &node) const { return node->shapes(); }
  TypePtr GetDtype(const NodePtr &node) const { return node->dtype(); }
  TypeId GetDtypeId(const NodePtr &node) const { return GetDtype(node)->type_id(); }
  ValuePtr GetAttr(const NodePtr &node, const std::string &attr) const;
  int64_t GetSize(const NodePtr &node) const;
  NodePtr DynSize(const NodePtr &node, const TypePtr &type) { return Cast(DynSize(node), type); }
  NodePtr DynSize(const NodePtr &node, TypeId type_id) { return Cast(DynSize(node), type_id); }
  NodePtr DynSize(const NodePtr &node);
  NodePtr Range(const NodePtr &limit) { return Range(Tensor(0, kInt64), limit, Tensor(1, kInt64)); }
  NodePtr Range(const NodePtr &start, const NodePtr &limit, const NodePtr &delta, int64_t max_len = 1000000) {
    return Emit("Range", {start, limit, delta}, {{"maxlen", MakeValue(max_len)}});
  }

  NodePtr SequenceToTensor(const NodePtr &node, const TypePtr &dtype = kInt64);
  NodePtr TensorToSequence(const NodePtr &node, const AbstractBasePtr &abs, const TypePtr &dtype = kInt64);
  NodePtr SequenceSetItem(const NodePtr &node, const NodePtr &index, const NodePtr &value);
  NodePtr SequenceSlice(const NodePtr &node, const NodePtr &start, const NodePtr &stop, const NodePtr &step);

  std::string name() const { return name_; }
  std::string GetTargetFromContext() const;
  bool IsGraphMode() const;

  // Tensor getitem by a single integer number on the outermost axis.
  NodePtr TensorGetItem(const NodePtr &node, int64_t idx);

  // get a tensor slice like python.
  // case 1: x[...,2]   => StridedSlice(x, {{-1,{2}}})
  // case 2: x[2, ..., 1:3]   => StridedSlice(x, {{0,{2}}, {-1,{1,3}}})
  // case 3: x[..., 0:3:2, 0::2, :]   => StridedSlice(x, {{-3,{0,3,2}}, {-2,{0,LLONG_MAX,2}}})
  NodePtr StridedSlice(const NodePtr &x, const std::map<int64_t, std::vector<int64_t>> &slices);
  std::string GetInstanceName() const { return instance_name_; }
  NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) { return Emit("TanhGrad", {y, dy}); }
  virtual NodePtr OutZeros(const NodePtr &node) { return ZerosLike(node); }

 protected:
  std::string name_;
  std::string instance_name_;
  const NodePtrList *inputs_ptr_{nullptr};
  const mindspore::HashMap<std::string, ValuePtr> *attrs_ptr_{nullptr};
};

class BpropIRBuilderFactory {
 public:
  static BpropIRBuilderFactory &Instance() {
    static BpropIRBuilderFactory instance{};
    return instance;
  }

  const BpropHandle *GetBuilder(const std::string &name) const {
    auto iter = registry_.find(name);
    return (iter == registry_.end()) ? nullptr : &(iter->second);
  }

  void RegBuilder(const std::string &name, const BpropIRBuilderFunc &func) { registry_[name].func = func; }
  void RegUnusedInputs(const std::string &name, const mindspore::HashSet<size_t> &unused) {
    registry_[name].unused_inputs = unused;
  }

 private:
  HashMap<std::string, BpropHandle> registry_;
};

class BpropIRBuilderRegHelper {
 public:
  explicit BpropIRBuilderRegHelper(const std::string &name) : name_(name) {}
  ~BpropIRBuilderRegHelper() = default;
  const BpropIRBuilderRegHelper &SetBody(const BpropIRBuilderFunc &func) const {
    BpropIRBuilderFactory::Instance().RegBuilder(name_, func);
    return *this;
  }
  const BpropIRBuilderRegHelper &SetUnusedInputs(const std::initializer_list<size_t> &unused_inputs) const {
    BpropIRBuilderFactory::Instance().RegUnusedInputs(name_, unused_inputs);
    return *this;
  }

 private:
  std::string name_;
};

#define BPROP_EXPANDER_JOIN(x, y) x##y
#define BPROP_EXPANDER_UNIQUE_NAME(prefix, cnt) BPROP_EXPANDER_JOIN(prefix, cnt)
#define REG_BPROP_BUILDER(name) \
  const BpropIRBuilderRegHelper BPROP_EXPANDER_UNIQUE_NAME(g_bprop, __COUNTER__) = BpropIRBuilderRegHelper(name)
#define BODYFUNC(v) [](BpropIRBuilder * v) -> NodePtrList

#ifdef _MSC_VER
#define REG_BPROP_BUILDERS_BEGIN(func_name) void Reg##func_name() {
#define REG_BPROP_BUILDERS_END }
#else
#define REG_BPROP_BUILDERS_BEGIN(func_name)
#define REG_BPROP_BUILDERS_END
#endif
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_BPROP_IRBUILDER_H_
