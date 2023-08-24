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
#include <string>
#include <unordered_map>
#include "mindspore/core/ops/sequence_ops.h"
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace expander {
namespace {
void AddDebugInfo(const FuncGraphPtr &fg, const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object name = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MS_CLASS_NAME, obj);
  auto cls_name = "TR-" + py::cast<std::string>(name);
  fg->debug_info()->set_name(cls_name);
}

void UpdateRecomputeScope(const FuncGraphPtr &fg, const py::object &obj) {
  py::object scope_str =
    python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_SCOPE_NAME, obj);
  if (!py::isinstance<py::none>(scope_str)) {
    auto scope_name = py::cast<std::string>(scope_str);
    if (scope_name.find("recompute_") == 0) {
      parse::UpdateRecomputeScope(fg);
    }
  }
}

void GraphDropout(const FuncGraphPtr &fg) {
  auto nodes = fg->TopoSort(fg->get_return());
  std::unordered_map<AnfNodePtr, bool> node_map;
  auto order_list = fg->order_list();
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      node_map[node] = true;
    }
  }
  for (auto &node : order_list) {
    if (node->isa<CNode>() && node != fg->get_return() && node_map.find(node) == node_map.end()) {
      fg->DropNode(node);
    }
  }
}

void UpdateFuncGraphInBegin(const FuncGraphPtr &fg, const py::object &obj) {
  if (!obj.is_none()) {
    AddDebugInfo(fg, obj);
    parse::data_converter::SetFuncGraphByCellObj(fg, obj);
    parse::UpdateFuncGraphFlags(obj, fg, true);
  }
}

void UpdateFuncGraphInEnd(const FuncGraphPtr &fg, const py::object &obj) {
  if (!obj.is_none()) {
    UpdateRecomputeScope(fg, obj);
  }
  GraphDropout(fg);
}

AbstractBasePtr GetAbstract(const AnfNodePtr &node) {
  const auto &abs = node->abstract();
  if (abs == nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(), node->ToString() + " has no abstract");
    auto node_abs = node->cast<ValueNodePtr>()->value()->ToAbstract();
    node->set_abstract(node_abs);
    return node_abs;
  }
  return abs;
}

inline bool IsPackTensor(const py::object &arg) { return py::hasattr(arg, "__pack__"); }

inline bool IsHasValue(const ValuePtr &value) { return (value != nullptr && !value->isa<ValueAny>()); }

bool IsTensorSequence(const py::object &arg) {
  if (!py::isinstance<py::tuple>(arg) && !py::isinstance<py::list>(arg)) {
    return false;
  }
  py::tuple seq = py::cast<py::tuple>(arg);
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
    py::gil_scoped_acquire acquire;
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

bool PackExpander::is_pynative_mode;

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

py::object PackExpander::BeginGraph(const py::object &obj, const abstract::AbstractBasePtrList &inputs) {
  py::tuple outputs(inputs.size());
  auto graph = std::make_shared<FuncGraph>();
  UpdateFuncGraphInBegin(graph, obj);
  graphs_.push(graph);
  for (size_t i = 0; i < inputs.size(); ++i) {
    outputs[i] = ConvertAbstractToParameter(inputs[i]);
  }
  if (is_pynative_mode) {
    // only in pynative, for parse::ResolveParameterObj need
    parse::Parser::UpdateTopFuncGraph(graph);
    (void)std::for_each(graph->parameters().begin(), graph->parameters().end(),
                        [](const AnfNodePtr &param) { param->cast<ParameterPtr>()->set_is_top_graph_param(true); });
  }
  return outputs;
}

py::object PackExpander::BeginSubGraph(const py::object &obj, const py::args &inputs) {
  auto up_graph = graphs_.top();
  auto graph = std::make_shared<FuncGraph>();
  UpdateFuncGraphInBegin(graph, obj);
  graphs_.push(graph);
  AnfNodePtrList node_inputs = {NewValueNode(graph)};
  auto args = py::cast<py::tuple>(inputs);
  py::tuple outputs(inputs.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto node = ConvertInput(args[i]);
    MS_EXCEPTION_IF_NULL(node);
    (void)node_inputs.emplace_back(node);
    outputs[i] = ConvertAbstractToParameter(node->abstract()->Clone());
  }
  auto node = up_graph->NewCNodeInOrder(node_inputs);
  func_graph_node_.push(node);
  return outputs;
}

FuncGraphPtr PackExpander::EndGraph(const py::object &obj, const py::object &output) {
  auto node = ConvertInput(output);
  MS_EXCEPTION_IF_NULL(node);
  auto graph = graphs_.top();
  graphs_.pop();
  graph->set_output(node);
  UpdateFuncGraphInEnd(graph, obj);
  return graph;
}

py::object PackExpander::EndSubGraph(const py::object &obj, const py::object &output) {
  auto func_node = func_graph_node_.top();
  func_graph_node_.pop();
  auto fg = EndGraph(obj, output);
  func_node->set_abstract(fg->output()->abstract());
  return ConvertCNodeToPython(func_node);
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
  auto val = abs->BuildValue();
  if (IsHasValue(val)) {
    if (!is_pynative_mode) {
      auto param = graphs_.top()->add_parameter();
      param->set_abstract(abs);
    }
    return ValueToPyData(val);
  } else if (abs->isa<abstract::AbstractSequence>()) {
    if (!is_pynative_mode) {
      auto param = graphs_.top()->add_parameter();
      param->set_abstract(abs);
      return ConvertCNodeToPython(param);
    } else {
      auto abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
      py::tuple tuple_node(abs_seq.size());
      for (size_t i = 0; i < abs_seq.size(); ++i) {
        tuple_node[i] = ConvertAbstractToParameter(abs_seq[i]);
      }
      return tuple_node;
    }
  } else {
    if (!abs->isa<abstract::AbstractTensor>()) {
      MS_LOG(WARNING) << "Input should be Tensor, but get " << abs->ToString() << ".";
    }
    auto param = graphs_.top()->add_parameter();
    param->set_abstract(abs);
    auto ret = std::make_shared<PackNode>(param);
    return py::cast(ret);
  }
}

py::object PackExpander::Emit(const py::object &prim, const py::args &inputs) const {
  MS_EXCEPTION_IF_NULL(graphs_.top());
  auto prim_py = std::make_shared<PrimitivePy>(prim);
  AnfNodePtrList cnode_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto node = ConvertInput(inputs[i]);
    MS_EXCEPTION_IF_NULL(node);
    (void)cnode_inputs.emplace_back(node);
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
    graphs_.top()->DropNode(cnode);
    auto node = NewValueNode(val);
    node->set_abstract(val->ToAbstract());
    return node;
  }
  cnode->set_abstract(infer_res);
  return cnode;
}

AnfNodePtr PackExpander::EmitCNode(const PrimitivePtr &prim, const AnfNodePtrList &cnode_inputs) const {
  AbstractBasePtrList abs_list;
  {
    py::gil_scoped_release release;
    (void)std::transform(cnode_inputs.cbegin(), cnode_inputs.cend(), std::back_inserter(abs_list), GetAbstract);
  }
  auto node = mindspore::prim::GenerateCNode(graphs_.top(), prim->name(), prim, abs_list, cnode_inputs);
  {
    py::gil_scoped_release release;
    auto cnode = node->cast<CNodePtr>();
    node = CNodeInfer(cnode);
  }
  return node;
}

AnfNodePtr PackExpander::ConvertInput(const py::object &arg) const {
  if (IsTensorSequence(arg)) {
    py::tuple tuple = py::cast<py::tuple>(arg);
    AnfNodePtrList cnode_inputs;
    for (size_t i = 0; i < tuple.size(); ++i) {
      (void)cnode_inputs.emplace_back(ConvertInput(tuple[i]));
    }
    if (py::hasattr(arg, "__parameter_tuple__")) {
      return EmitCNode(prim::kPrimMakeTuple, std::move(cnode_inputs));
    }
    return EmitCNode(prim::kPrimMakeTuple, cnode_inputs);
  } else if (IsPackTensor(arg)) {
    py::object node = py::getattr(arg, stub::PY_ATTR_STUB);
    return node.cast<std::shared_ptr<PackNode>>()->Get();
  } else if (py::hasattr(arg, "__parameter__") && py::isinstance<tensor::MetaTensor>(arg)) {
    auto node = parse::ResolveParameterObj(graphs_.top(), arg);
    auto param_node = node->cast<ParameterPtr>();
    if (param_node->has_default()) {
      auto context = parallel::ParallelContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context);
      auto param_abs = pipeline::GetDefaultValueAbstract(param_node);
      context->ParallelParameterContextRestoreShape(graphs_.top(), param_node, param_abs);
      node->set_abstract(param_abs);
    }
    return node;
  } else {
    auto val = parse::data_converter::PyDataToValue(arg);
    MS_EXCEPTION_IF_NULL(val);
    auto node = NewValueNode(val);
    MS_EXCEPTION_IF_NULL(node);
    node->set_abstract(val->ToAbstract());
    return node;
  }
}

void PackExpander::SetMixedPrecisionFlagToGraph() const {
  auto mixed_type = mix_precision_types_.top();
  graphs_.top()->set_flag(GRAPH_FLAG_MIX_PRECISION_FP16, mixed_type == MixedPrecisionType::kFP16);
  graphs_.top()->set_flag(GRAPH_FLAG_MIX_PRECISION_FP32, mixed_type == MixedPrecisionType::kFP32);
  graphs_.top()->set_flag(GRAPH_FLAG_MIX_PRECISION_BF16, mixed_type == MixedPrecisionType::kBF16);
}

bool PackExpander::SetMixedPrecision(const py::object &obj) {
  if (!py::isinstance<Cell>(obj)) {
    return false;
  }
  auto cell = py::cast<CellPtr>(obj);
  MS_EXCEPTION_IF_NULL(cell);
  auto mixed_type = cell->GetMixedPrecisionType();
  if (mixed_type != MixedPrecisionType::kNotSet &&
      (mix_precision_types_.empty() || mixed_type != mix_precision_types_.top())) {
    mix_precision_types_.push(mixed_type);
    SetMixedPrecisionFlagToGraph();
    return true;
  }
  return false;
}

void PackExpander::RecoverMixedPrecision() {
  py::gil_scoped_release release;
  mix_precision_types_.pop();
  if (!mix_precision_types_.empty()) {
    return SetMixedPrecisionFlagToGraph();
  }
  graphs_.top()->erase_flag(GRAPH_FLAG_MIX_PRECISION_FP16);
  graphs_.top()->erase_flag(GRAPH_FLAG_MIX_PRECISION_FP32);
  graphs_.top()->erase_flag(GRAPH_FLAG_MIX_PRECISION_BF16);
}

void RegPackExpanderPy(const py::module *m) {
  (void)py::class_<PackNode, std::shared_ptr<PackNode>>(*m, "PackNode")
    .def("get_shape", &PackNode::GetShape, "get shape")
    .def("get_dtype", &PackNode::GetDtype, "get dtype")
    .def("get_value", &PackNode::GetValue, "get value");

  (void)py::class_<PackExpander, std::shared_ptr<PackExpander>>(*m, "PackExpander")
    .def_static("get_instance", &PackExpander::Instance, "PackExpander get_instance.")
    .def("emit", &PackExpander::Emit, "emit op in current graph")
    .def("begin_subgraph", &PackExpander::BeginSubGraph, "begin subgraph in current graph")
    .def("end_subgraph", &PackExpander::EndSubGraph, "end subgraph in current graph")
    .def("set_mixed_precision", &PackExpander::SetMixedPrecision, "set mixed precision by python cell.")
    .def("recover_mixed_precision", &PackExpander::RecoverMixedPrecision, "recover mixed precision.");
}
}  // namespace expander
}  // namespace mindspore
