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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_EXPANDER_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_EXPANDER_H_

#include <memory>
#include <stack>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "ir/anf.h"
#include "ir/cell.h"
#include "ir/func_graph.h"
#include "ir/value.h"

namespace mindspore {
namespace expander {
class COMMON_EXPORT PackNode : public Value {
 public:
  explicit PackNode(const AnfNodePtr node) : node_(node) {}
  PackNode() = default;
  virtual ~PackNode() = default;
  MS_DECLARE_PARENT(PackNode, Value);

  py::object GetShape() const;
  py::object GetDtype() const;
  py::object GetValue() const;

  AnfNodePtr Get() const { return node_; }
  AbstractBasePtr ToAbstract() override { return node_->abstract(); }
  bool operator==(const Value &other) const override { return other.isa<PackNode>() && &other == this; }

 private:
  AnfNodePtr node_;
};
using PackNodePtr = std::shared_ptr<PackNode>;

class PackExpander {
 public:
  static std::shared_ptr<PackExpander> Instance() {
    static const std::shared_ptr<PackExpander> instance = std::make_shared<PackExpander>();
    return instance;
  }
  PackExpander() = default;
  ~PackExpander() = default;

  py::object BeginGraph(const py::object &obj, const abstract::AbstractBasePtrList &inputs);
  FuncGraphPtr EndGraph(const py::object &obj, const py::object &output);
  py::object BeginSubGraph(const py::object &obj, const py::args &inputs);
  py::object EndSubGraph(const py::object &obj, const py::object &output);

  py::object Emit(const py::object &prim, const py::args &inputs) const;
  static bool is_pynative_mode;

  bool SetMixedPrecision(const py::object &obj);
  void RecoverMixedPrecision();

 private:
  AnfNodePtr EmitCNode(const PrimitivePtr &prim, const AnfNodePtrList &cnode_inputs) const;
  AnfNodePtr ConvertInput(const py::object &arg) const;
  AnfNodePtr CNodeInfer(const CNodePtr &cnode) const;
  py::object ConvertCNodeToPython(const AnfNodePtr &node) const;
  py::object ConvertAbstractToParameter(const AbstractBasePtr &abs) const;
  void SetMixedPrecisionFlagToGraph() const;

  std::stack<FuncGraphPtr> graphs_;
  std::stack<AnfNodePtr> func_graph_node_;
  std::stack<MixedPrecisionType> mix_precision_types_;
};

void RegPackExpanderPy(const py::module *m);
}  // namespace expander
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_PACK_EXPANDER_H_
