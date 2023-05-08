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
#include "frontend/expander/pack/pack_expander.h"

#include <utility>
#include <algorithm>
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/stub_tensor.h"

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

inline bool IsPackTensor(const py::object &arg) {
  return py::hasattr(arg, stub::PY_ATTR_STUB) && py::isinstance<PackNode>(py::getattr(arg, stub::PY_ATTR_STUB));
}

inline bool IsHasValue(const ValuePtr &value) {
  return (value != nullptr && !value->isa<ValueAny>() && !value->isa<None>());
}

template <typename T>
bool IsTensorSequence(const py::object &arg) {
  if (!py::isinstance<T>(arg)) {
    return false;
  }
  T seq = py::cast<T>(arg);
  for (size_t i = 0; i < seq.size(); ++i) {
    if (IsPackTensor(seq[i]) || py::isinstance<tensor::Tensor>(seq[i])) {
      return true;
    }
  }
  return false;
}

std::pair<AbstractBasePtr, ValuePtr> InferShapeAndValue(const PrimitivePtr &prim, const AbstractBasePtrList abs_list,
                                                        const bool &need_infer_value) {
  auto found = abstract::GetFrontendPrimitiveInferImpl(prim);
  AbstractBasePtr infer_res = nullptr;
  ValuePtr val = nullptr;
  // C++ InferShape and InferValue.
  if (found.has_value()) {
    if (need_infer_value && found->IsImplInferValue()) {
      val = found->InferValue(prim, abs_list);
    } else if (found->IsImplInferShapeAndType()) {
      infer_res = found->InferShapeAndType(nullptr, prim, abs_list);
      val = infer_res->BuildValue();
    }
  }
  // Python InferShape and InferValue.
  if ((infer_res == nullptr || need_infer_value) && !IsHasValue(val)) {
    auto prim_py = prim->cast<PrimitivePyPtr>();
    auto py_infer_args = PreparePyInputs(abs_list);
    if (infer_res == nullptr) {
      auto py_infer_result = prim_py->RunInfer(py_infer_args);
      infer_res = abstract::PyInferRes2Abstract(prim_py, py_infer_result);
    }
    MS_EXCEPTION_IF_NULL(infer_res);
    if (need_infer_value && py::hasattr(prim_py->GetPyObj(), PY_PRIM_METHOD_INFER_VALUE)) {
      py::tuple py_vals(py_infer_args.size());
      for (size_t i = 0; i < py_infer_args.size(); ++i) {
        py_vals[i] = py_infer_args[i][ATTR_VALUE];
      }
      py::object py_ret = prim_py->RunInferValue(py_vals);
      if (!py::isinstance<py::none>(py_ret)) {
        bool converted = parse::ConvertData(py_ret, &val, false, infer_res->BuildType());
        if (!converted) {
          MS_LOG(EXCEPTION) << "Convert data failed";
        }
      }
    }
  }
  return {infer_res, val};
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
    outputs[i] = ConvertAbstractToParameter(inputs[i]);
  }
  return outputs;
}

FuncGraphPtr PackExpander::EndGraph(const py::object &output) {
  auto node = ConvertInput(output);
  MS_EXCEPTION_IF_NULL(node);
  graph_->set_output(node);
  auto graph = graph_;
  graph_.reset();
  return graph;
}

py::object PackExpander::ConvertCNodeToPython(const AnfNodePtr &node) const {
  auto type = node->abstract()->BuildType();
  if (type->isa<Tuple>()) {
    size_t len = type->cast<TuplePtr>()->size();
    py::tuple tuple_node(len);
    for (size_t i = 0; i < len; ++i) {
      auto cnode = EmitCNode(prim::kPrimTupleGetItem, {node, NewValueNode(SizeToLong(i))});
      tuple_node[i] = ConvertCNodeToPython(cnode);
    }
    return tuple_node;
  } else if (type->isa<TensorType>()) {
    auto ret = std::make_shared<PackNode>(node);
    return py::cast(ret);
  } else {
    auto val = node->abstract()->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    return ValueToPyData(val);
  }
}

py::object PackExpander::ConvertAbstractToParameter(const AbstractBasePtr &abs) const {
  if (IsHasValue(abs->BuildValue())) {
    auto val = abs->BuildValue();
    graph_->AddNode(NewValueNode(val));
    return ValueToPyData(val);
  } else if (abs->isa<abstract::AbstractSequence>()) {
    size_t len = abs->cast<abstract::AbstractSequencePtr>()->size();
    py::tuple tuple_node(len);
    for (size_t i = 0; i < len; ++i) {
      tuple_node[i] = ConvertAbstractToParameter(abs->cast<abstract::AbstractSequencePtr>()->elements()[i]);
    }
    return tuple_node;
  } else {
    if (!abs->isa<abstract::AbstractTensor>()) {
      MS_LOG(WARNING) << "input should be Tensor, but get " << abs->ToString();
    }
    auto param = graph_->add_parameter();
    param->set_abstract(abs);
    auto ret = std::make_shared<PackNode>(param);
    return py::cast(ret);
  }
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
  return ConvertCNodeToPython(cnode);
}

AnfNodePtr PackExpander::CNodeInfer(const CNodePtr &cnode) const {
  auto prim = GetCNodePrimitive(cnode);
  auto cnode_inputs = cnode->inputs();

  AbstractBasePtrList abs_list(cnode_inputs.size() - 1);
  bool need_infer_value = true;

  for (size_t i = 1; i < cnode_inputs.size(); ++i) {
    auto node = cnode_inputs[i];
    if (node->isa<CNode>() && node->abstract() == nullptr) {
      node = CNodeInfer(node->cast<CNodePtr>());
      if (node->isa<ValueNode>()) {
        cnode->set_input(i, node);
      }
    }
    abs_list[i - 1] = GetAbstract(node);
    auto value = abs_list[i - 1]->BuildValue();
    need_infer_value &= IsHasValue(value);
  }
  auto [infer_res, val] = InferShapeAndValue(prim, abs_list, need_infer_value);
  if (IsHasValue(val)) {
    graph_->DropNode(cnode);
    auto node = NewValueNode(val);
    node->set_abstract(val->ToAbstract());
    return node;
  }
  cnode->set_abstract(infer_res);
  return cnode;
}

AnfNodePtr PackExpander::EmitCNode(const PrimitivePtr &prim, const AnfNodePtrList &cnode_inputs) const {
  AbstractBasePtrList abs_list;
  (void)std::transform(cnode_inputs.cbegin(), cnode_inputs.cend(), std::back_inserter(abs_list), GetAbstract);
  auto node = mindspore::prim::GenerateCNode(graph_, prim->name(), prim, abs_list, cnode_inputs);
  auto cnode = node->cast<CNodePtr>();
  node = CNodeInfer(cnode);
  return node;
}

AnfNodePtr PackExpander::ConvertInput(const py::object &arg) const {
  auto ConvertSqeuenceInput = [&](const auto &tuple) -> AnfNodePtr {
    AnfNodePtrList cnode_inputs;
    for (size_t i = 0; i < tuple.size(); ++i) {
      cnode_inputs.emplace_back(ConvertInput(tuple[i]));
    }
    return EmitCNode(prim::kPrimMakeTuple, cnode_inputs);
  };
  if (IsTensorSequence<py::tuple>(arg)) {
    return ConvertSqeuenceInput(py::cast<py::tuple>(arg));
  }
  if (IsTensorSequence<py::list>(arg)) {
    return ConvertSqeuenceInput(py::cast<py::list>(arg));
  }
  // value
  if (IsPackTensor(arg)) {
    py::object node = py::getattr(arg, stub::PY_ATTR_STUB);
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
