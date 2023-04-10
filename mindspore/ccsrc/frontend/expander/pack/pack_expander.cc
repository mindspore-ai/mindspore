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
#include <algorithm>
#include "ir/tensor.h"
#include "pybind_api/ir/primitive_py.h"
#include "abstract/ops/primitive_infer_map.h"
#include "pipeline/jit/parse/data_converter.h"
#include "frontend/expander/pack/pack_expander.h"

namespace mindspore {
namespace expander {
py::object PackNode::GetShape() const {
  auto base = node_->abstract()->BuildShape();
  auto shape = base->cast<abstract::ShapePtr>();
  const ShapeVector &shape_vector = shape->shape();
  auto ret = py::tuple(shape_vector.size());
  for (size_t i = 0; i < shape_vector.size(); ++i) {
    ret[i] = shape_vector[i];
  }
  return ret;
}

py::object PackNode::GetDtype() const {
  auto base = node_->abstract()->BuildType();
  if (base->isa<TensorType>()) {
    base = base->cast<TensorTypePtr>()->element();
  }
  return py::cast(base);
}

py::object PackNode::GetValue() const { return py::none(); }

py::object PackExpander::BeginGraph(const abstract::AbstractBasePtrList &inputs) {
  py::tuple outputs(inputs.size());
  graph_ = std::make_shared<FuncGraph>();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto param = graph_->add_parameter();
    param->set_abstract(inputs[i]);
    outputs[i] = py::cast(std::make_shared<PackNode>(param));
  }
  return outputs;
}

FuncGraphPtr PackExpander::EndGraph(const py::object &output) const {
  auto node = ConvertInput(output);
  MS_EXCEPTION_IF_NULL(node);
  graph_->set_output(node);
  return graph_;
}

py::object PackExpander::Emit(const py::object &prim, const py::args &inputs) const {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &adapter = prim.cast<PrimitivePyAdapterPtr>();
  auto prim_py = std::make_shared<PrimitivePy>(prim, adapter);
  AnfNodePtrList cnode_inputs = {NewValueNode(prim_py)};
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto node = ConvertInput(inputs[i]);
    MS_EXCEPTION_IF_NULL(node);
    cnode_inputs.emplace_back(node);
  }
  auto cnode = EmitCNode(cnode_inputs);
  auto ret = std::make_shared<PackNode>(cnode);
  return py::cast(ret);
}

AnfNodePtr PackExpander::EmitCNode(const AnfNodePtrList &cnode_inputs) const {
  auto cnode = graph_->NewCNode(cnode_inputs);
  AbstractBasePtr abs;
  auto prim = GetCNodePrimitive(cnode);
  auto found = abstract::GetPrimitiveInferImpl(prim);
  if (found.has_value() && found.value().IsImplInferShapeAndType()) {
    AbstractBasePtrList abs_list;
    (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(abs_list),
                         [](const AnfNodePtr &node) {
                           const auto &abs = node->abstract();
                           if (abs == nullptr) {
                             MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(), node->ToString() + " has no abstract");
                             return node->cast<ValueNodePtr>()->value()->ToAbstract();
                           }
                           return abs;
                         });
    abs = found.value().InferShapeAndType(nullptr, prim, abs_list);
    cnode->set_abstract(abs);
  } else {
    MS_LOG(WARNING) << "donot found infer:" << prim->name();
  }
  return cnode;
}

AnfNodePtr PackExpander::ConvertInput(const py::object &arg) const {
  auto IsTensorTuple = [arg]() -> bool {
    if (!py::isinstance<py::tuple>(arg)) return false;
    py::tuple tuple = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (py::hasattr(tuple[i], "pack_node") || py::isinstance<tensor::Tensor>(tuple[i])) {
        return true;
      }
    }
    return false;
  };
  if (IsTensorTuple()) {
    AnfNodePtrList cnode_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    py::tuple tuple = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < tuple.size(); ++i) {
      cnode_inputs.emplace_back(ConvertInput(tuple[i]));
    }
    return EmitCNode(cnode_inputs);
  }
  // value
  if (py::hasattr(arg, "pack_node")) {
    py::object node = py::getattr(arg, "pack_node");
    return node.cast<std::shared_ptr<PackNode>>()->Get();
  } else {
    auto val = parse::data_converter::PyDataToValue(arg);
    MS_EXCEPTION_IF_NULL(val);
    auto node = NewValueNode(val);
    MS_EXCEPTION_IF_NULL(node);
    node->set_abstract(val->ToAbstract());
    return node;
  }
}

void RegPackExpanderPy(const py::module *m) {
  (void)py::class_<PackNode, std::shared_ptr<PackNode>>(*m, "PackNode")
    .def("get_shape", &PackNode::GetShape, "get value")
    .def("get_dtype", &PackNode::GetDtype, "get value")
    .def("get_value", &PackNode::GetValue, "get value");

  (void)py::class_<PackExpander, std::shared_ptr<PackExpander>>(*m, "PackExpander")
    .def_static("get_instance", &PackExpander::Instance, "PackExpander get_instance.")
    .def("emit", &PackExpander::Emit, "emit op in current graph");
}
}  // namespace expander
}  // namespace mindspore
