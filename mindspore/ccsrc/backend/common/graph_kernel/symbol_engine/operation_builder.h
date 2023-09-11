/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATION_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATION_BUILDER_H_
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>

#include "ir/anf.h"
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"
#include "utils/hash_map.h"

namespace mindspore::graphkernel::symbol {
class SymbolCache {
 public:
  void SetShape(const AnfNodePtr &node, const SymbolPtr &symbol) { node_shape_map_[RealNode(node)] = symbol; }
  void SetValue(const AnfNodePtr &node, const SymbolPtr &symbol) { node_value_map_[RealNode(node)] = symbol; }
  SymbolPtr GetShape(const AnfNodePtr &node) const {
    auto iter = node_shape_map_.find(RealNode(node));
    return iter == node_shape_map_.end() ? nullptr : iter->second;
  }
  SymbolPtr GetValue(const AnfNodePtr &node) const {
    auto iter = node_value_map_.find(RealNode(node));
    return iter == node_value_map_.end() ? nullptr : iter->second;
  }
  InputSymbolPtr GetInput(const AnfNodePtr &para) const {
    auto iter = input_index_.find(RealNode(para));
    return iter == input_index_.end() ? nullptr : inputs_[iter->second];
  }

  void InitInputs(const AnfNodePtrList &parameters);
  bool UpdateInputs(const AbstractBasePtrList &inputs);

  /// \brief bind nodes in cache. after that, queries with `ref_node` will get the result of `real_node`.
  void BindNode(const AnfNodePtr &ref_node, const AnfNodePtr &real_node) { node_binder_[ref_node] = real_node; }
  const AnfNodePtr &RealNode(const AnfNodePtr &ref_node) const {
    auto iter = node_binder_.find(ref_node);
    return (iter != node_binder_.end()) ? iter->second : ref_node;
  }

 protected:
  HashMap<AnfNodePtr, SymbolPtr> node_shape_map_;
  HashMap<AnfNodePtr, SymbolPtr> node_value_map_;
  InputSymbolPtrList inputs_;
  HashMap<AnfNodePtr, size_t> input_index_;
  HashMap<AnfNodePtr, AnfNodePtr> node_binder_;
};
namespace ops::builders {
class OperationBuilder;
using InferFunc = std::function<SymbolPtr(OperationBuilder *)>;
enum class DependOn { kShape, kValue, kNone };
struct OperationBuilderInfo {
  InferFunc build_shape_func{nullptr};
  InferFunc build_value_func{nullptr};
  std::vector<DependOn> build_shape_depend;
  std::vector<DependOn> build_value_depend;
};

class OperationBuilder {
 public:
  OperationBuilder(OperationEmitter *emitter, SymbolCache *cache, const OperationBuilderInfo &info)
      : emitter_(emitter), cache_(cache), symbol_builder_info_(info) {}
  ~OperationBuilder() = default;
  inline SymbolPtr BuildShape(const CNodePtr &cnode) {
    if (symbol_builder_info_.build_shape_func == nullptr) {
      return nullptr;
    }
    cnode_ = cnode;
    return symbol_builder_info_.build_shape_func(this);
  }
  inline SymbolPtr BuildValue(const CNodePtr &cnode) {
    if (symbol_builder_info_.build_value_func == nullptr) {
      return nullptr;
    }
    cnode_ = cnode;
    return symbol_builder_info_.build_value_func(this);
  }

  SymbolPtr Emit(const OpPtr &op) const { return emitter_->Emit(op); }
  const CNodePtr &cnode() const { return cnode_; }
  SymbolPtr RealShape(const AnfNodePtr &node) const;
  SymbolPtr RealValue(const AnfNodePtr &node) const;
  AnfNodePtr GetInput(size_t i) const { return cnode_->input(i); }
  SymbolPtr GetAttr(const std::string &attr_name) const;

 protected:
  OperationEmitter *emitter_;
  SymbolCache *cache_;
  CNodePtr cnode_;
  const OperationBuilderInfo &symbol_builder_info_;
};
using OperationBuilderPtr = std::unique_ptr<OperationBuilder>;

class OperationBuilderRegistry {
 public:
  inline static bool HasBuilder(const std::string &name) {
    return OperationBuilderRegistry::Instance().builders_.count(name) > 0;
  }
  inline static OperationBuilderPtr GetBuilder(const std::string &name, OperationEmitter *e, SymbolCache *cache) {
    const auto &builders = OperationBuilderRegistry::Instance().builders_;
    auto iter = builders.find(name);
    if (iter == builders.end()) {
      return nullptr;
    }
    return std::make_unique<OperationBuilder>(e, cache, iter->second);
  }

  inline static OperationBuilderRegistry &Instance() {
    static OperationBuilderRegistry ins{};
    return ins;
  }

  class RegHelper {
   public:
    explicit RegHelper(const std::string &name) { builder_ = OperationBuilderRegistry::Instance().NewBuilder(name); }
    RegHelper &SetBuildShape(const std::initializer_list<DependOn> &depends, const InferFunc &func) {
      builder_->build_shape_depend = depends;
      builder_->build_shape_func = func;
      return *this;
    }
    RegHelper &SetBuildShape(const InferFunc &func) {
      builder_->build_shape_func = func;
      return *this;
    }
    RegHelper &SetBuildValue(const std::initializer_list<DependOn> &depends, const InferFunc &func) {
      builder_->build_value_depend = depends;
      builder_->build_value_func = func;
      return *this;
    }
    RegHelper &SetBuildValue(const InferFunc &func) {
      builder_->build_value_func = func;
      return *this;
    }
    OperationBuilderInfo *builder_;
  };

 private:
  OperationBuilderInfo *NewBuilder(const std::string &name) { return &builders_[name]; }
  std::unordered_map<std::string, OperationBuilderInfo> builders_;
};
}  // namespace ops::builders
using ops::builders::OperationBuilderRegistry;

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define REG_OP_SYMBOL_BUILDER(name) \
  static const auto UNIQUE_NAME(g_ob_, __COUNTER__) = OperationBuilderRegistry::RegHelper(name)
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATION_BUILDER_H_
