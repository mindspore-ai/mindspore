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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_BPROP_IRBUILDER_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_BPROP_IRBUILDER_H_

#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <functional>

#include "common/graph_kernel/bprop/expander/node.h"
#include "common/graph_kernel/bprop/expander/emitter.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace expander {
namespace bprop {
using DoutUser = std::vector<std::pair<CNodePtr, int>>;
class BpropIRBuilder : public Emitter {
 public:
  BpropIRBuilder(const std::string &name, const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer)
      : Emitter(func_graph, infer), name_(name) {}

  /// \brief Run irbuilder to generate a graph
  bool Run(const NodePtrList &inputs, const DAttr &attrs, std::vector<CNodePtr> *outputs, DoutUser *dout_user);

  ValuePtr GetAttr(const std::string &attr) const;
  template <typename S>
  S GetAttr(const std::string &attr) const {
    return GetValue<S>(GetAttr(attr));
  }
  const DAttr &GetAttrs() const { return *attrs_ptr_; }
  NodePtr GetInput(size_t i) const;
  const NodePtrList &GetInputs() const { return *inputs_ptr_; }

  ShapeVector GetShape(const NodePtr &node) const;
  TypePtr GetDtype(const NodePtr &node) const;
  TypeId GetDtypeId(const NodePtr &node) const { return GetDtype(node)->type_id(); }
  ValuePtr GetAttr(const NodePtr &node, const std::string &attr) const;

  std::string name() const { return name_; }

  void DumpResult(const std::vector<CNodePtr> &outputs, const DoutUser &dout_user) const;

 protected:
  void FindDoutUsers(const std::vector<CNodePtr> &outputs, DoutUser *dout_user) const;

  std::string name_;
  const NodePtrList *inputs_ptr_{nullptr};
  const DAttr *attrs_ptr_{nullptr};
};
using BpropIRBuilderPtr = std::shared_ptr<BpropIRBuilder>;

using BpropIRBuilderFunc = std::function<NodePtrList(const BpropIRBuilder *)>;
class BpropIRBuilderFactory {
 public:
  static BpropIRBuilderFactory &Instance() {
    static BpropIRBuilderFactory instance{};
    return instance;
  }
  const BpropIRBuilderFunc &GetBuilder(const std::string &name) { return builders()[name]; }
  void RegBuilder(const std::string &name, const BpropIRBuilderFunc &func) { builders()[name] = func; }
  bool HasOp(const std::string &name) const { return builders().count(name) != 0; }

 private:
  HashMap<std::string, BpropIRBuilderFunc> &builders() const {
    static HashMap<std::string, BpropIRBuilderFunc> builder_map;
    return builder_map;
  }
};

class BpropIRBuilderRegHelper {
 public:
  explicit BpropIRBuilderRegHelper(const std::string &name) : name_(name) {}
  ~BpropIRBuilderRegHelper() = default;
  const BpropIRBuilderRegHelper &SetBody(const BpropIRBuilderFunc &func) const {
    BpropIRBuilderFactory::Instance().RegBuilder(name_, func);
    return *this;
  }

 private:
  std::string name_;
};

#define BPROP_EXPANDER_JOIN(x, y) x##y
#define BPROP_EXPANDER_UNIQUE_NAME(prefix, cnt) BPROP_EXPANDER_JOIN(prefix, cnt)
#define REG_BPROP_BUILDER(name) \
  const BpropIRBuilderRegHelper BPROP_EXPANDER_UNIQUE_NAME(g_bprop, __COUNTER__) = BpropIRBuilderRegHelper(name)
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_BPROP_BPROP_IRBUILDER_H_
