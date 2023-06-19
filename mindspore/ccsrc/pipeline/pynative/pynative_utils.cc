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
#include "pipeline/pynative/pynative_utils.h"
#include <algorithm>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "pybind_api/ir/primitive_py.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/stub_tensor.h"
#include "frontend/expander/bprop/bprop.h"

namespace mindspore {
namespace pynative {
namespace PyNativeAlgo {
namespace {
std::string GetObjIdFromPython(const py::handle &obj) {
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}

std::string GetIdForPyTupleOrList(const py::handle &obj) {
  auto p_list = py::cast<py::tuple>(obj);
  string prefix = py::isinstance<py::tuple>(obj) ? "Tuple<" : "List<";
  if (p_list.empty()) {
    prefix = "Empty:";
  } else {
    for (size_t i = 0; i < p_list.size(); ++i) {
      prefix += PyParser::GetIdByPyObj(p_list[i]) + ":";
    }
  }
  prefix.pop_back();
  prefix += ">";
  return prefix;
}

std::string GetFnInfoByPyObj(const py::object &obj) {
  std::string fn_info = obj.attr("__module__").cast<std::string>();
  fn_info += "_" + obj.attr("__name__").cast<std::string>();
  fn_info += "_" + obj.attr("__code__").attr("co_filename").cast<std::string>();
  fn_info += "_" + py::str(obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  if (py::hasattr(obj, "__warpped__")) {
    auto warpped_obj = obj.attr("__warpped__");
    fn_info += "_" + warpped_obj.attr("__name__").cast<std::string>();
    fn_info += "_" + warpped_obj.attr("__code__").attr("co_filename").cast<std::string>();
    fn_info += "_" + py::str(warpped_obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  }
  return fn_info;
}

void AddDynInputsSizesAttr(const FrontendOpRunInfoPtr &op_run_info) {
  if (op_run_info->base_op_run_info.dyn_input_sizes.empty()) {
    return;
  }
  op_run_info->op_grad_info->op_prim->set_attr(kAttrDynInputSizes,
                                               MakeValue(op_run_info->base_op_run_info.dyn_input_sizes));
}
}  // namespace

AbstractBasePtr Common::SetAbstractValueToAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kValueAny);
  } else if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    for (const auto &elem : abs_seq->elements()) {
      (void)SetAbstractValueToAnyValue(elem);
    }
  } else if (abs->isa<abstract::AbstractDictionary>()) {
    const auto &abs_dic = abs->cast<abstract::AbstractDictionaryPtr>();
    for (const auto &elem : abs_dic->elements()) {
      (void)SetAbstractValueToAnyValue(elem.first);
      (void)SetAbstractValueToAnyValue(elem.second);
    }
  }
  return abs;
}

AnfNodePtr Common::ConvertValueSequenceToMakeTuple(const ValueNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &v = node->value();
  if (!v->isa<ValueSequence>()) {
    return node;
  }
  auto value_sequence = v->cast<ValueSequencePtr>();
  if (!node->abstract()->isa<abstract::AbstractSequence>() &&
      (node->abstract()->cast<abstract::AbstractSequencePtr>()->size() != value_sequence->size())) {
    MS_LOG(EXCEPTION) << "Get wrong matched abs " << node->abstract()->ToString() << " and value "
                      << value_sequence->ToString();
  }

  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &value : value_sequence->value()) {
    MS_EXCEPTION_IF_NULL(value);
    auto value_node = NewValueNode(value);
    auto abs = Common::SetAbstractValueToAnyValue(value->ToAbstract());
    value_node->set_abstract(abs);
    auto tuple_node = ConvertValueSequenceToMakeTuple(value_node, func_graph);
    (void)inputs.emplace_back(tuple_node);
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  auto make_tuple_node = func_graph->NewCNode(inputs);
  make_tuple_node->set_abstract(node->abstract());
  return make_tuple_node;
}

std::string Common::GetIdByValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>()->id();
  } else if (v->isa<stub::StubNode>()) {
    return GetIdByValue(v->cast<stub::StubNodePtr>()->WaitValue());
  } else if (v->isa<Cell>()) {
    return v->cast<CellPtr>()->id();
  } else if (v->isa<mindspore::Type>()) {
    auto type_ptr = v->cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  } else if (v->isa<StringImm>()) {
    return "S" + v->cast<StringImmPtr>()->value();
  } else if (v->isa<BoolImm>()) {
    return "B" + std::to_string(v->cast<BoolImmPtr>()->value());
  } else if (v->isa<IntegerImm>()) {
    return "I" + std::to_string(v->cast<Int64ImmPtr>()->value());
  } else if (v->isa<FloatImm>()) {
    return "F" + std::to_string(v->cast<FP32ImmPtr>()->value());
  } else if (v->isa<None>()) {
    return "None";
  } else if (v->isa<Ellipsis>()) {
    return "Ellipsis";
  } else if (v->isa<ValueSequence>()) {
    auto p_list = v->cast<ValueSequencePtr>();
    string prefix = v->isa<ValueTuple>() ? "Tuple<" : "List<";
    if (p_list->size() == 0) {
      prefix = "Empty:";
    } else {
      for (size_t i = 0; i < p_list->size(); ++i) {
        prefix += GetIdByValue(p_list->value()[i]) + ":";
      }
    }
    prefix.pop_back();
    prefix += ">";
    return prefix;
  }
  MS_LOG(DEBUG) << "Get type " << v->ToString();
  return v->ToString();
}

bool Common::ValueHasDynamicShape(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return value->cast<tensor::TensorPtr>()->base_shape_ptr() != nullptr;
  } else if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return std::any_of(value_seq->value().begin(), value_seq->value().end(),
                       [](const ValuePtr &elem) { return ValueHasDynamicShape(elem); });
  }
  return false;
}

bool Common::IsTensor(const ValuePtr &v, bool include_sequence) {
  MS_EXCEPTION_IF_NULL(v);
  if (include_sequence) {
    if (v->isa<tensor::Tensor>() || v->isa<tensor::MetaSparseTensor>()) {
      return true;
    } else if (v->isa<ValueSequence>()) {
      auto v_seq = v->cast<ValueSequencePtr>();
      if (v_seq->size() == 0) {
        return false;
      }
      // SpareTensor have scalar index, so just check have csr tensor
      if (v_seq->value().front()->isa<tensor::MetaSparseTensor>()) {
        return true;
      }
      // All value are tensor
      return std::all_of(v_seq->value().begin(), v_seq->value().end(),
                         [](const ValuePtr &e) { return PyNativeAlgo::Common::IsTensor(e, true); });
    } else {
      return false;
    }
  }
  return v->isa<tensor::Tensor>() || v->isa<tensor::MetaSparseTensor>();
}

bool Common::IsControlFlowGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

ValuePtr Common::FilterSensValues(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<tensor::COOTensor>() || value->isa<tensor::CSRTensor>()) {
    return value;
  } else if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> value_list;
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (auto &filter_value : value_seq->value()) {
      if (FilterSensValues(filter_value) != nullptr) {
        (void)value_list.emplace_back(filter_value);
      }
    }
    return std::make_shared<ValueTuple>(value_list);
  } else {
    MS_LOG(DEBUG) << "Value type: " << value->ToString();
    return nullptr;
  }
}

tensor::TensorPtr Common::GetTensorFromParam(const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(param_node);
  auto param = param_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  if (!param->has_default()) {
    return nullptr;
  }
  auto default_value = param->default_param();
  MS_EXCEPTION_IF_NULL(default_value);
  auto tensor_value = default_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_value);
  return tensor_value;
}

std::shared_ptr<PyNativeExecutor> Common::GetPyNativeExecutor() {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  return executor;
}

void Common::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR(filename, graph);
  }
#endif
}

TypeId Common::GetTypeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    return GetTypeFromAbstract(abs_seq->elements().front());
  }
  const auto &type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  return common::AnfAlgo::GetOutputInferDataType(type, 0);
}

ShapeVector Common::GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get abstract sequence";
  }
  auto shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_ptr = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  return shape_ptr->shape();
}

ValuePtr Common::CreatOutputTensorValueByAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto type_id = pynative::PyNativeAlgo::Common::GetTypeFromAbstract(abs);
  if (abs->isa<abstract::AbstractMonad>()) {
    return std::make_shared<tensor::Tensor>(0);
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> out;
    if (!abs_seq->elements().front()->isa<abstract::AbstractTensor>()) {
      MS_LOG(EXCEPTION) << "Get non tensor output";
    }
    for (size_t i = 0; i < abs_seq->size(); ++i) {
      (void)out.emplace_back(std::make_shared<tensor::Tensor>(type_id, GetShapeFromAbstract(abs_seq->elements()[i])));
    }
    return std::make_shared<ValueTuple>(out);
  }
  return std::make_shared<tensor::Tensor>(type_id, GetShapeFromAbstract(abs));
}

void Common::ReplaceCNodeWithValueNode(const FuncGraphPtr &bprop_graph) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  if (bprop_graph->used_forward_nodes().empty()) {
    return;
  }
  auto mng = MakeManager({bprop_graph}, false);
  auto tr = mng->Transact();
  for (const auto &forward_node : bprop_graph->used_forward_nodes()) {
    auto cnode = forward_node->cast<CNodePtr>();
    auto v_node = cnode->forward().first;
    MS_EXCEPTION_IF_NULL(v_node);
    bprop_graph->AddValueNode(v_node);
    MS_LOG(DEBUG) << "Replace " << forward_node->DebugString() << " by value node " << v_node->DebugString();
    auto converted_node = ConvertValueSequenceToMakeTuple(v_node, bprop_graph);
    (void)tr.Replace(forward_node, converted_node);
  }
  tr.Commit();
  bprop_graph->ClearUsedForwardNodes();
  PyNativeAlgo::Common::DumpGraphIR("replace_cnode_with_valuenode.ir", bprop_graph);
}

ValuePtr StubNodeToValueInner(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue();
  } else if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    if (!values.empty() && utils::isa<Scalar>(values[0])) {
      return v;
    }
    ValuePtrList value_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(value_list),
                         [](const ValuePtr &value) { return StubNodeToValueInner(value); });
    if (utils::isa<ValueTuple>(v)) {
      return std::make_shared<ValueTuple>(value_list);
    } else if (utils::isa<ValueList>(v)) {
      return std::make_shared<ValueList>(value_list);
    } else {
      MS_LOG(EXCEPTION) << "Not support ValueSequence " << v->ToString();
    }
  } else {
    return v;
  }
}

void Common::StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    op_run_info->op_grad_info->input_value[i] = StubNodeToValueInner(op_run_info->op_grad_info->input_value[i]);
  }
}

void Common::GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                 const std::string &device_target, bool is_dynamic_shape,
                                 mindspore::HashSet<size_t> *input_to_attr_index) {
  if (op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_index);
    return;
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return;
  }

  auto reg_info =
    opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, device_target, is_dynamic_shape);
  if (reg_info == nullptr) {
    return;
  } else {
    MS_EXCEPTION_IF_NULL(input_to_attr_index);
    for (auto &iter : reg_info->input_attr_map()) {
      (void)input_to_attr_index->insert(iter.first);
    }
  }
}

ValueNodePtr Common::CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  auto v_node = NewValueNode(v);
  if (abs == nullptr) {
    v_node->set_abstract(SetAbstractValueToAnyValue(v->ToAbstract()));
  } else {
    v_node->set_abstract(abs);
  }
  return v_node;
}

tensor::TensorPtr Common::CreateFakeTensorWithoutDeviceAddress(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  //  auto t = ObjectPool::GetInstance().GetObj<tensor::Tensor>(ObjType::kTensor);
  //  t->set_id(tensor->id());
  //  t->MetaTensor::set_shape(tensor->shape());
  //  t->set_auto_grad_meta_data(tensor->auto_grad_meta_data());
  auto t = std::make_shared<tensor::Tensor>(*tensor);
  if (tensor->is_parameter()) {
    t->set_param_info(tensor->param_info());
  }
  t->set_device_address(nullptr);
  return t;
}

ValuePtr Common::CreateFakeValueWithoutDeviceAddress(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &v_t = value->cast<tensor::TensorPtr>();
    auto t = std::make_shared<tensor::Tensor>(*v_t);
    if (v_t->is_parameter()) {
      t->set_param_info(v_t->param_info());
    }
    t->set_device_address(nullptr);
    return t;
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    ValuePtrList value_list;
    (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(value_list),
                         [](const ValuePtr &elem) { return CreateFakeValueWithoutDeviceAddress(elem); });
    return std::make_shared<ValueTuple>(value_list);
  } else if (value->isa<stub::StubNode>()) {
    const auto &stub_node = value->cast<stub::StubNodePtr>();
    return CreateFakeValueWithoutDeviceAddress(stub_node->WaitValue());
  } else {
    return value;
  }
}

TensorGradType Common::SetValueGradInfo(const ValuePtr &value, const TopCellInfoPtr &top_cell,
                                        TensorGradType grad_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &tensor_value = value->cast<tensor::TensorPtr>();
    if (tensor_value->auto_grad_meta_data() != nullptr) {
      return tensor_value->auto_grad_meta_data()->grad_type();
    }
    const auto &auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    if (tensor_value->is_parameter() && grad_type != TensorGradType::kInput) {
      grad_type = TensorGradType::kParameter;
    }
    auto_grad_meta_data->set_grad_type(grad_type);
    tensor_value->set_auto_grad_meta_data(auto_grad_meta_data);
    if (top_cell != nullptr && PyNativeAlgo::Common::IsParam(grad_type)) {
      top_cell->AddParamGradInfo(tensor_value, auto_grad_meta_data);
    }
    return grad_type;
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    TensorGradType ret_type = grad_type;
    for (const auto &v : value_seq) {
      auto ret = SetValueGradInfo(v, top_cell, grad_type);
      if (PyNativeAlgo::Common::IsParam(ret)) {
        ret_type = ret;
      }
    }
    return ret_type;
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, top_cell, grad_type);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, top_cell, grad_type);
  }
  return grad_type;
}

TensorGradType Common::SetTensorGradInfo(const tensor::TensorPtr &tensor, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->auto_grad_meta_data() != nullptr) {
    const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
    return auto_grad_meta_data->grad_type();
  }
  tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
  const auto &auto_grad_meta_data = tensor->auto_grad_meta_data();
  // Set weight tensor grad type
  if (tensor->is_parameter()) {
    auto_grad_meta_data->set_grad_type(TensorGradType::kParameter);
    if (top_cell != nullptr) {
      top_cell->AddParamGradInfo(tensor, auto_grad_meta_data);
    }
    return TensorGradType::kParameter;
  }
  // Is a constant input tensor, but not constant scalar value
  return TensorGradType::kConstant;
}

void Common::SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &original_params = func_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  bool need_add_input_abs = op_run_info->op_grad_info->input_abs.empty();
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      op_run_info->op_grad_info->input_value_grad_type[i] =
        SetValueGradInfo(op_run_info->op_grad_info->input_value[i], nullptr, TensorGradType::kConstant);
      if (need_add_input_abs) {
        (void)op_run_info->op_grad_info->input_abs.emplace_back(original_params[i]->abstract());
      }
      continue;
    }
    // Must weight param
    const auto &param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = PyNativeAlgo::Common::GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    (void)op_run_info->op_grad_info->input_value.emplace_back(tensor_value);
    (void)op_run_info->op_grad_info->input_value_grad_type.emplace_back(
      PyNativeAlgo::Common::SetTensorGradInfo(tensor_value, nullptr));
    (void)op_run_info->op_grad_info->input_abs.emplace_back(param->abstract());
    MS_LOG(DEBUG) << "Set graph weight parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

std::string PyParser::GetIdByPyObj(const py::object &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    return obj.cast<tensor::TensorPtr>()->id();
  } else if (IsStubTensor(obj)) {
    return ConvertStubTensor(obj)->id();
  } else if (py::isinstance<Cell>(obj)) {
    return obj.cast<CellPtr>()->id();
  } else if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = obj.cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  } else if (py::isinstance<py::str>(obj)) {
    return "S" + obj.cast<std::string>();
  } else if (py::isinstance<py::bool_>(obj)) {
    return "B" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::int_>(obj)) {
    return "I" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::float_>(obj)) {
    return "F" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::none>(obj)) {
    return "None";
  } else if (py::isinstance<py::ellipsis>(obj)) {
    return "Ellipsis";
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    return GetIdForPyTupleOrList(obj);
  } else if (py::isinstance<py::function>(obj)) {
    return GetFnInfoByPyObj(obj);
  }
  // For id with value and obj can be the same
  if (py::isinstance<tensor::CSRTensor>(obj) || py::isinstance<tensor::COOTensor>(obj) ||
      py::isinstance<tensor::RowTensor>(obj)) {
    return DataConvert::PyObjToValue(obj)->ToString();
  }
  return GetObjIdFromPython(obj);
}

void PyParser::SetPrim(const FrontendOpRunInfoPtr &op_run_info, const py::object &prim_arg) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &adapter = prim_arg.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(prim_arg);
    adapter->set_attached_primitive(prim);
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->signatures = prim->signatures();
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs, bool stub) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  op_run_info->op_grad_info->input_value.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->op_grad_info->input_value[i] = PyNativeAlgo::DataConvert::PyObjToValue(op_inputs[i], stub);
  }
  PrepareOpGradInfo(op_run_info);
}

void PyParser::PrepareOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) {
  // Do some prepare for grad
  if (!op_run_info->requires_grad) {
    return;
  }
  // kIndex1 is for add output
  op_run_info->input_unused_in_bprop.resize(op_run_info->input_size + kIndex1, false);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size, TensorGradType::kConstant);
  if (!op_run_info->is_ms_function_input) {
    const auto &unused_inputs = BpropExpander().GetUnusedInputs(op_run_info->op_grad_info->op_prim->name());
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      op_run_info->input_unused_in_bprop[i] = (unused_inputs.find(i) != unused_inputs.end());
    }
    // Set out used
    op_run_info->input_unused_in_bprop[op_run_info->input_size] =
      unused_inputs.find(op_run_info->input_size) != unused_inputs.end();
  }
}

py::object DataConvert::ValueToPyObj(const ValuePtr &v) { return ValueToPyData(v); }

void SetAdapterTensorAttr(const py::object &obj) {
  if (py::hasattr(obj, PYTHON_ADAPTER_TENSOR)) {
    // In PyNative mode, AdapterTensor is treated as ms.Tensor.
    py::setattr(obj, PYTHON_ADAPTER_TENSOR, py::bool_(false));
  } else if (py::isinstance<py::tuple>(obj)) {
    auto obj_tuple = obj.cast<py::tuple>();
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      SetAdapterTensorAttr(obj_tuple[i]);
    }
  } else if (py::isinstance<py::list>(obj)) {
    auto obj_list = obj.cast<py::list>();
    for (size_t i = 0; i < obj_list.size(); ++i) {
      SetAdapterTensorAttr(obj_list[i]);
    }
  }
}

ValuePtr DataConvert::PyObjToValue(const py::object &obj, bool stub) {
  SetAdapterTensorAttr(obj);
  ValuePtr converted_ret;
  if (stub) {
    converted_ret = parse::data_converter::PyDataToStubNode(obj);
  } else {
    converted_ret = parse::data_converter::PyDataToValue(obj);
  }
  if (converted_ret == nullptr) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  return converted_ret;
}

ValuePtr DataConvert::BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::TensorPtr>(value)) {
    auto t = utils::cast<tensor::TensorPtr>(value);
    if (requires_grad) {
      t->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
      t->auto_grad_meta_data()->set_grad_type(TensorGradType::kOpOutput);
    }
    ret = t;
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref, requires_grad, is_out_sequence);
  } else if (utils::isa<int>(value)) {
    ret = MakeValue(utils::cast<int>(value));
  } else if (utils::isa<float>(value)) {
    ret = MakeValue(utils::cast<float>(value));
  } else if (utils::isa<double>(value)) {
    ret = MakeValue(utils::cast<double>(value));
  } else if (utils::isa<bool>(value)) {
    ret = MakeValue(utils::cast<bool>(value));
  } else {
    MS_LOG(EXCEPTION) << "value is not support type " << value.ToString();
  }
  return ret;
}

ValuePtr DataConvert::VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence) {
  MS_EXCEPTION_IF_NULL(vec_ref);

  size_t value_size = vec_ref.size();
  if (value_size == 1 && !is_out_sequence) {
    return BaseRefToValue(vec_ref[0], requires_grad, is_out_sequence);
  }
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i], requires_grad, is_out_sequence);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::FlattenValueSeqArg(const ValuePtr &v, std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(flatten_v);
  if (v->isa<tensor::Tensor>()) {
    (void)flatten_v->emplace_back(v);
  } else if (v->isa<ValueSequence>()) {
    const auto &v_vec = v->cast<ValueSequencePtr>()->value();
    for (const auto &elem : v_vec) {
      FlattenValueSeqArg(elem, flatten_v);
    }
  }
}

void DataConvert::FlattenArgs(const std::vector<ValuePtr> &v_vec, std::vector<ValuePtr> *flatten_v, bool has_sens) {
  MS_EXCEPTION_IF_NULL(flatten_v);
  size_t input_size = has_sens ? v_vec.size() - 1 : v_vec.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto &v = v_vec[i];
    MS_EXCEPTION_IF_NULL(v);
    MS_LOG(DEBUG) << "Get v is " << v->ToString();
    if (PyNativeAlgo::Common::IsTensor(v)) {
      (void)flatten_v->emplace_back(v);
    }
  }
  if (has_sens) {
    if (PyNativeAlgo::Common::IsTensor(v_vec[input_size])) {
      (void)flatten_v->emplace_back(v_vec[input_size]);
    } else if (v_vec[input_size]->isa<ValueSequence>()) {
      FlattenValueSeqArg(v_vec[input_size], flatten_v);
    }
  }
}

bool DataConvert::RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                               size_t input_index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->input_to_attr.empty()) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(v);
  if (op_run_info->input_to_attr.find(input_index) == op_run_info->input_to_attr.end()) {
    return false;
  }
  const auto &input_names_value = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return false;
  }
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "The input index: " << input_index << " is larger than the input names vector size!";
  }
  const auto &input_name = input_names_vec[input_index];
  if (v->isa<tensor::Tensor>()) {
    auto tensor = v->cast<tensor::TensorPtr>();
    if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
      return false;
    }
  }
  (void)op_run_info->op_grad_info->op_prim->AddAttr(input_name, v);
  return true;
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  ValuePtrList fake_tensor_list;
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kOpOutput;
  }
  for (const auto &v : value_seq->value()) {
    if (!v->isa<tensor::Tensor>()) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    int64_t tensor_mask = kParameterDataTensorMask;
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_parameter()) {
      tensor_mask = kParameterWeightTensorMask;
    }
    if (op_run_info->requires_grad) {
      auto grad_type = Common::SetTensorGradInfo(tensor, top_cell);
      if (PyNativeAlgo::Common::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kParameter;
      }
      if (op_run_info->input_unused_in_bprop[index]) {
        (void)fake_tensor_list.emplace_back(Common::CreateFakeTensorWithoutDeviceAddress(tensor));
      }
    }
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
  }
  if (op_run_info->requires_grad && op_run_info->input_unused_in_bprop[index]) {
    op_run_info->op_grad_info->input_value[index] = std::make_shared<ValueTuple>(fake_tensor_list);
  }
  if (!op_run_info->base_op_run_info.dyn_input_sizes.empty()) {
    int64_t elem_size = SizeToLong(value_seq->size());
    if (op_run_info->base_op_run_info.dyn_input_sizes.size() != op_run_info->input_size) {
      for (size_t i = op_run_info->base_op_run_info.dyn_input_sizes.size(); i < index; ++i) {
        (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
      }
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(elem_size);
    } else {
      op_run_info->base_op_run_info.dyn_input_sizes[index] = elem_size;
    }
  } else {
    for (size_t i = 0; i < index; ++i) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
    (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(SizeToLong(value_seq->size()));
  }
}

void DataConvert::ConvertValueTupleToTensor(const FrontendOpRunInfoPtr &op_run_info,
                                            const ValueSequencePtr &value_seq) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  ValueTuplePtr value_tuple;
  if (value_seq->isa<ValueList>()) {
    value_tuple = std::make_shared<ValueTuple>(value_seq->value());
  } else {
    value_tuple = value_seq->cast<ValueTuplePtr>();
  }
  MS_EXCEPTION_IF_NULL(value_tuple);
  auto tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor_ptr);
}

void DataConvert::ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                                   const TopCellInfoPtr &top_cell, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(map_tensor);
  constexpr int input_num = 1;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(map_tensor);
  const auto it = op_run_info->base_op_run_info.input_mask.end();
  (void)op_run_info->base_op_run_info.input_mask.insert(it, input_num, kParameterWeightTensorMask);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] =
      Common::SetTensorGradInfo(op_run_info->base_op_run_info.input_tensor.back(), top_cell);
  }
}

void DataConvert::ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                               const tensor::CSRTensorPtr &csr_tensor, const TopCellInfoPtr &top_cell,
                                               size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(csr_tensor);
  constexpr int input_num = 3;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }

  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_mask.end();
  (void)op_run_info->base_op_run_info.input_mask.insert(it, input_num, kParameterDataTensorMask);
  op_run_info->op_grad_info->op_prim->set_attr("is_csr", MakeValue(true));
  op_run_info->op_grad_info->op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kOpOutput;
    for (int i = 0; i < input_num; ++i) {
      auto iter = op_run_info->base_op_run_info.input_tensor.rbegin() + i;
      auto grad_type = Common::SetTensorGradInfo(*iter, top_cell);
      if (PyNativeAlgo::Common::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kParameter;
      }
    }
  }
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    std::vector<int64_t> axis = {};
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(std::make_shared<tensor::Tensor>(axis, kInt64));
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::Tensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, index, top_cell);
  } else {
    ConvertValueTupleToTensor(op_run_info, value_seq);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
  }
}

void DataConvert::ConvertValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                                       const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  tensor::TensorPtr tensor_ptr = nullptr;
  int64_t tensor_mask = kParameterDataTensorMask;
  if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), top_cell, index);
    return;
  } else if (v->isa<tensor::Tensor>()) {
    tensor_ptr = v->cast<tensor::TensorPtr>();
    if (tensor_ptr->is_parameter()) {
      tensor_mask = kParameterWeightTensorMask;
    }
    if (op_run_info->requires_grad) {
      op_run_info->op_grad_info->input_value_grad_type[index] = Common::SetTensorGradInfo(tensor_ptr, top_cell);
      if (op_run_info->input_unused_in_bprop[index]) {
        op_run_info->op_grad_info->input_value[index] = Common::CreateFakeTensorWithoutDeviceAddress(tensor_ptr);
      }
    }
  } else if (v->isa<FloatImm>()) {
    double input_value = v->cast<FP32ImmPtr>()->value();
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<BoolImm>()) {
    tensor_ptr = std::make_shared<tensor::Tensor>(v->cast<BoolImmPtr>()->value(), kBool);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<IntegerImm>()) {
    int64_t input = v->cast<Int64ImmPtr>()->value();
    if (op_run_info->op_grad_info->op_prim->name() == prim::kPrimCSRReduceSum->name()) {
      op_run_info->op_grad_info->op_prim->set_attr("axis", MakeValue(input));
      return;
    }
    tensor_ptr = std::make_shared<tensor::Tensor>(input, kInt64);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<Type>()) {
    int64_t type_id = v->cast<TypePtr>()->type_id();
    tensor_ptr = std::make_shared<tensor::Tensor>(type_id, kInt64);
    tensor_ptr->set_user_data(kTensorValueIsType, v);
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<StringImm>()) {
    auto value_string = GetValue<std::string>(v);
    const ShapeVector shape = {1, SizeToLong(value_string.size())};
    tensor_ptr = std::make_shared<tensor::Tensor>(kObjectTypeString, shape, value_string.data(), value_string.size());
    tensor_mask = kValueNodeTensorMask;
  } else if (v->isa<ValueSequence>()) {
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), index, top_cell);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), top_cell, index);
    return;
  } else if (v->isa<None>() || v->isa<Monad>()) {
    return;
  } else if (v->isa<parse::InterpretedObject>()) {
    MS_EXCEPTION(TypeError) << "Not support for " << v->ToString();
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
}

void ReplaceReduceAxis(const FrontendOpRunInfoPtr &op_run_info) {
  if (!common::AnfAlgo::IsReduceOp(op_run_info->base_op_run_info.op_name)) {
    return;
  }
  const auto &input_tensors = op_run_info->base_op_run_info.input_tensor;
  constexpr size_t kReduceOpInputNum = 2;
  if (input_tensors.size() < kReduceOpInputNum) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size " << input_tensors.size() << " of Op "
                      << op_run_info->base_op_run_info.op_name;
  }

  const auto &axis_shape = input_tensors[1]->shape();
  // 2nd input tensor is {}, means reduce all axis.
  if (axis_shape.size() == 1 && axis_shape[0] == 0) {
    auto size = input_tensors[0]->shape().size();
    std::vector<int64_t> axis;
    for (size_t i = 0; i < size; ++i) {
      axis.push_back(SizeToLong(i));
    }
    op_run_info->base_op_run_info.input_tensor[1] = std::make_shared<tensor::Tensor>(axis);
  }
}

void ReplaceValueNodeWithParameter(const FrontendOpRunInfoPtr &op_run_info) {
  if (!op_run_info->base_op_run_info.use_dynamic_shape_process) {
    return;
  }

  auto replace_tensor_mask = [](const FrontendOpRunInfoPtr &op_run_info) {
    std::replace_if(
      op_run_info->base_op_run_info.input_mask.begin(), op_run_info->base_op_run_info.input_mask.end(),
      [](auto mask) { return mask == kValueNodeTensorMask; }, kParameterDataTensorMask);
  };

  // value to parameter(onehot)
  if (op_run_info->base_op_run_info.device_target != kAscendDevice) {
    replace_tensor_mask(op_run_info);
  }
}

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target,
                                 const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);

  (void)op_run_info->base_op_run_info.input_tensor.reserve(op_run_info->input_size);
  (void)op_run_info->base_op_run_info.input_mask.reserve(op_run_info->input_size);
  // Get input tensors.
  op_run_info->op_grad_info->op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
    // convert const input to attr
    if (RunOpConvertConstInputToAttr(op_run_info, input_object, index)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    ConvertValueToTensor(op_run_info, input_object, index, top_cell);
    // -1 indicates input_object is not a dynInput
    if (!op_run_info->base_op_run_info.dyn_input_sizes.empty() && !input_object->isa<ValueSequence>()) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
  }
  op_run_info->op_grad_info->op_prim->EndRecordAddAttr();
  ReplaceValueNodeWithParameter(op_run_info);
  ReplaceReduceAxis(op_run_info);
  AddDynInputsSizesAttr(op_run_info);
}
}  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
