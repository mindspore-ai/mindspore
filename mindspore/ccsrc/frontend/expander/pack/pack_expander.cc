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
#include "frontend/operator/composite/do_signature.h"
#include "include/common/utils/convert_utils_py.h"
#include "backend/common/pass/convert_const_input_to_attr.h"

namespace mindspore {
namespace expander {
namespace {
AbstractBasePtr GetAbstract(const AnfNodePtr &node) {
  const auto &abs = node->abstract();
  if (abs == nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(), node->ToString() + " has no abstract");
    return node->cast<ValueNodePtr>()->value()->ToAbstract();
  }
  return abs;
}
}  // namespace
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

py::object PackNode::GetValue() const {
  if (node_->abstract()->BuildValue() != kValueAny) {
    return ValueToPyData(node_->abstract()->BuildValue());
  }
  return py::none();
}

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
  AnfNodePtrList cnode_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto node = ConvertInput(inputs[i]);
    MS_EXCEPTION_IF_NULL(node);
    cnode_inputs.emplace_back(node);
  }
  auto cnode = EmitCNode(prim_py, cnode_inputs);
  auto ret = std::make_shared<PackNode>(cnode);
  return py::cast(ret);
}

void PackExpander::CNodeInfer(const CNodePtr &cnode) const {
  auto prim = GetCNodePrimitive(cnode);
  auto found = abstract::GetPrimitiveInferImpl(prim);
  auto cnode_inputs = cnode->inputs();
  for (const auto &node : cnode_inputs) {
    if (node->isa<CNode>() && node->abstract() == nullptr) {
      CNodeInfer(node->cast<CNodePtr>());
    }
  }
  if (found.has_value() && found.value().IsImplInferShapeAndType()) {
    AbstractBasePtrList abs_list;
    (void)std::transform(cnode_inputs.cbegin() + 1, cnode_inputs.cend(), std::back_inserter(abs_list), GetAbstract);
    auto abs = found.value().InferShapeAndType(nullptr, prim, abs_list);
    cnode->set_abstract(abs);
  } else {
    MS_LOG(WARNING) << "donot found infer:" << prim->name();
  }
}

AnfNodePtr PackExpander::EmitCNode(const PrimitivePtr &prim, const AnfNodePtrList &cnode_inputs) const {
  AbstractBasePtrList abs_list;
  (void)std::transform(cnode_inputs.cbegin(), cnode_inputs.cend(), std::back_inserter(abs_list), GetAbstract);
  auto node = mindspore::prim::GenerateCNode(graph_, prim->name(), prim, abs_list, cnode_inputs);
  auto cnode = node->cast<CNodePtr>();
  CNodeInfer(cnode);
  return node;
}

AnfNodePtr PackExpander::ConvertInput(const py::object &arg) const {
  auto IsTensorTuple = [arg]() -> bool {
    if (!py::isinstance<py::tuple>(arg)) return false;
    py::tuple tuple = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < tuple.size(); ++i) {
      if (py::hasattr(tuple[i], PY_ATTR_PACK_NODE) || py::isinstance<tensor::Tensor>(tuple[i])) {
        return true;
      }
    }
    return false;
  };
  if (IsTensorTuple()) {
    AnfNodePtrList cnode_inputs;
    py::tuple tuple = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < tuple.size(); ++i) {
      cnode_inputs.emplace_back(ConvertInput(tuple[i]));
    }
    return EmitCNode(prim::kPrimMakeTuple, cnode_inputs);
  }
  // value
  if (py::hasattr(arg, PY_ATTR_PACK_NODE)) {
    py::object node = py::getattr(arg, PY_ATTR_PACK_NODE);
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
