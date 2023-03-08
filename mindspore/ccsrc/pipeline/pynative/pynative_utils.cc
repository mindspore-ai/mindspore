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
#include "backend/common/optimizer/helper.h"
#include "backend/common/optimizer/op_adaptation_info_factory.h"
#include "pybind_api/ir/primitive_py.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace pynative {
namespace PyNativeAlgo {
namespace {
void ClonePrim(const FrontendOpRunInfoPtr &op_run_info) {
  // Clone a new prim
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto new_prim = std::make_shared<PrimitivePy>(*(op_run_info->op_prim->cast<PrimitivePyPtr>()));
  op_run_info->op_prim = new_prim;
  MS_EXCEPTION_IF_NULL(new_prim->adapter());
  if (new_prim->adapter()->attached_primitive() == nullptr) {
    new_prim->adapter()->set_attached_primitive(new_prim);
  }
}

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
}  // namespace

AbstractBasePtr Common::SetAbstractValueToAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kAnyValue);
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

std::string Common::GetIdByValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>()->id();
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

void Common::SetForwardOutputFlag(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    auto tensor_value = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_value);
    tensor_value->set_is_forward_output(true);
  } else if (v->isa<ValueSequence>()) {
    auto v_seq = v->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(v_seq);
    (void)std::for_each(v_seq->value().begin(), v_seq->value().end(),
                        [](const ValuePtr &elem) { return SetForwardOutputFlag(elem); });
  }
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
  auto mng = MakeManager({bprop_graph}, false);
  auto tr = mng->Transact();
  for (const auto &forward_node : bprop_graph->used_forward_nodes()) {
    auto cnode = forward_node->cast<CNodePtr>();
    auto v_node = cnode->forward().first;
    bprop_graph->AddValueNode(v_node);
    MS_LOG(DEBUG) << "Replace " << forward_node->DebugString() << " by value node " << v_node;
    tr.Replace(forward_node, v_node);
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
    ValuePtrList value_list;
    std::transform(values.begin(), values.end(), std::back_inserter(value_list),
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
    op_run_info->input_value[i] = StubNodeToValueInner(op_run_info->input_value[i]);
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
    prim = std::make_shared<PrimitivePy>(prim_arg, adapter);
    adapter->set_attached_primitive(prim);
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  op_run_info->op_prim = prim;
  op_run_info->signatures = prim->signatures();
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs, bool stub) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->input_value.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->input_value[i] = PyNativeAlgo::DataConvert::PyObjToValue(op_inputs[i], stub);
  }
}

py::object DataConvert::ValueToPyObj(const ValuePtr &v) { return ValueToPyData(v); }

ValuePtr DataConvert::PyObjToValue(const py::object &obj, bool stub) {
  // In PyNative mode, AdapterTensor is treated as ms.Tensor.
  if (py::hasattr(obj, PYTHON_ADAPTER_TENSOR) && py::getattr(obj, PYTHON_ADAPTER_TENSOR).cast<bool>()) {
    py::setattr(obj, PYTHON_ADAPTER_TENSOR, py::bool_(false));
  }
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

ValuePtr DataConvert::PyObjToStubNode(const py::object &obj) {
  // In PyNative mode, AdapterTensor is treated as ms.Tensor.
  if (py::hasattr(obj, PYTHON_ADAPTER_TENSOR) && py::getattr(obj, PYTHON_ADAPTER_TENSOR).cast<bool>()) {
    py::setattr(obj, PYTHON_ADAPTER_TENSOR, py::bool_(false));
  }
  ValuePtr converted_ret = parse::data_converter::PyDataToStubNode(obj);
  if (converted_ret == nullptr) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  return converted_ret;
}

ValuePtr DataConvert::BaseRefToValue(const BaseRef &value) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::TensorPtr>(value)) {
    ret = utils::cast<tensor::TensorPtr>(value);
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref);
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

ValuePtr DataConvert::VectorRefToValue(const VectorRef &vec_ref) {
  MS_EXCEPTION_IF_NULL(vec_ref);
  size_t value_size = vec_ref.size();
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i]);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::FlattenTupleArg(const ValuePtr &v, std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(flatten_v);
  const auto &v_vec = v->cast<ValueSequencePtr>();
  size_t v_vec_size = v_vec->size();
  for (size_t i = 0; i < v_vec_size; ++i) {
    const auto &elem_v = v_vec->value()[i];
    MS_LOG(DEBUG) << "Get elem_v is " << v->ToString();
    if (elem_v->isa<ValueSequence>()) {
      FlattenTupleArg(elem_v, flatten_v);
    } else if (PyNativeAlgo::Common::IsTensor(elem_v)) {
      (void)flatten_v->emplace_back(elem_v);
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
      FlattenTupleArg(v_vec[input_size], flatten_v);
    }
  }
}

bool DataConvert::RunOpConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                               size_t input_index, const PrimitivePtr &op_prim,
                                               const mindspore::HashSet<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(op_prim);
  if (input_attrs.find(input_index) == input_attrs.end()) {
    return false;
  }
  const auto &input_names_value = op_prim->GetAttr(kAttrInputNames);
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
  (void)op_prim->AddAttr(input_name, v);
  return true;
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           const PrimitivePtr &op_prim, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  MS_EXCEPTION_IF_NULL(op_prim);
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
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
  }
  if (op_prim->HasAttr(kAttrDynInputSizes)) {
    int64_t elem_size = SizeToLong(value_seq->size());
    auto dyn_v = GetValue<const std::vector<int64_t>>(op_prim->GetAttr(kAttrDynInputSizes));
    if (dyn_v.size() != op_run_info->input_size) {
      for (size_t i = dyn_v.size(); i < index; ++i) {
        (void)dyn_v.emplace_back(-1);
      }
      (void)dyn_v.emplace_back(elem_size);
      (void)op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
    } else {
      if (dyn_v[index] != elem_size) {
        dyn_v[index] = elem_size;
        op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
      }
    }
  } else {
    std::vector<int64_t> dyn_v;
    for (size_t i = 0; i < index; ++i) {
      (void)dyn_v.emplace_back(-1);
    }
    (void)dyn_v.emplace_back(SizeToLong(value_seq->size()));
    op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
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
                                   const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(map_tensor);
  constexpr int input_num = 1;
  const auto input_names = op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(map_tensor);
  const auto it = op_run_info->base_op_run_info.input_mask.end();
  (void)op_run_info->base_op_run_info.input_mask.insert(it, input_num, kParameterWeightTensorMask);
}

void DataConvert::ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                               const tensor::CSRTensorPtr &csr_tensor, const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(csr_tensor);
  constexpr int input_num = 3;
  const auto input_names = op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_mask.end();
  (void)op_run_info->base_op_run_info.input_mask.insert(it, input_num, kParameterDataTensorMask);
  op_prim->set_attr("is_csr", MakeValue(true));
  op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            const PrimitivePtr &op_prim, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  MS_EXCEPTION_IF_NULL(op_prim);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    std::vector<int64_t> axis = {};
    (void)op_run_info->base_op_run_info.input_tensor.emplace_back(std::make_shared<tensor::Tensor>(axis, kInt64));
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::Tensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, op_prim, index);
  } else {
    ConvertValueTupleToTensor(op_run_info, value_seq);
    (void)op_run_info->base_op_run_info.input_mask.emplace_back(kValueNodeTensorMask);
  }
}

void DataConvert::ConvertValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                                       const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(op_prim);
  tensor::TensorPtr tensor_ptr = nullptr;
  int64_t tensor_mask = kParameterDataTensorMask;
  if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), op_prim);
    return;
  } else if (v->isa<tensor::Tensor>()) {
    tensor_ptr = v->cast<tensor::TensorPtr>();
    if (tensor_ptr->is_parameter()) {
      tensor_mask = kParameterWeightTensorMask;
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
    if (op_prim->name() == prim::kPrimCSRReduceSum->name()) {
      op_prim->set_attr("axis", MakeValue(input));
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
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), op_prim, index);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), op_prim);
    return;
  } else if (v->isa<None>() || v->isa<Monad>()) {
    return;
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_tensor.emplace_back(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_mask.emplace_back(tensor_mask);
}

bool DataConvert::NeedConvertConstInputToAttr(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target,
                                              mindspore::HashSet<size_t> *input_to_attr_ptr) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_to_attr_ptr);
  if (op_run_info->base_op_run_info.op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    const PrimitivePtr &op_prim = op_run_info->op_prim;
    MS_EXCEPTION_IF_NULL(op_prim);
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_ptr);
    return !input_to_attr_ptr->empty();
  }

  // Ascend const input to attr move to AscendVmOpAdapter
  if (device_target == kAscendDevice) {
    return false;
  }

  auto reg_info = opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(
    op_run_info->base_op_run_info.op_name, device_target,
    op_run_info->base_op_run_info.has_dynamic_output || op_run_info->base_op_run_info.use_dynamic_shape_process);
  if (reg_info == nullptr) {
    return false;
  } else {
    for (auto &iter : reg_info->input_attr_map()) {
      (void)input_to_attr_ptr->insert(iter.first);
    }
  }
  return !input_to_attr_ptr->empty();
}

void ReplaceValueNodeWithParameter(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target) {
  if (!op_run_info->base_op_run_info.use_dynamic_shape_process) {
    return;
  }

  auto replace_tensor_mask = [](const FrontendOpRunInfoPtr &op_run_info) {
    const auto &tensor_masks = op_run_info->base_op_run_info.input_mask;
    std::vector<int64_t> new_masks;
    std::transform(tensor_masks.begin(), tensor_masks.end(), std::back_inserter(new_masks), [](int64_t tensor_mask) {
      return tensor_mask == kValueNodeTensorMask ? kParameterDataTensorMask : tensor_mask;
    });
    op_run_info->base_op_run_info.input_mask = new_masks;
  };

  if (device_target == kAscendDevice) {
    auto reg_info = opt::OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(
      op_run_info->base_op_run_info.op_name, device_target, true);
    if (reg_info != nullptr) {
      auto no_need_input_to_attr = reg_info->need_tbe_check_supported();
      if (no_need_input_to_attr) {
        replace_tensor_mask(op_run_info);
      }
    }
  } else {
    replace_tensor_mask(op_run_info);
  }
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

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  mindspore::HashSet<size_t> input_to_attr = {};
  bool need_convert_input_to_attr = NeedConvertConstInputToAttr(op_run_info, device_target, &input_to_attr);
  MS_LOG(DEBUG) << "Need convert input to addr " << need_convert_input_to_attr;
  if (need_convert_input_to_attr) {
    // Prim may be changed attr
    ClonePrim(op_run_info);
  }
  const auto &op_prim = op_run_info->op_prim;

  // Get input tensors.
  op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->input_value[index];
    // convert const input to attr
    if (need_convert_input_to_attr &&
        RunOpConvertConstInputToAttr(op_run_info, input_object, index, op_prim, input_to_attr)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    ConvertValueToTensor(op_run_info, input_object, index, op_prim);
    // -1 indicates input_object is not a dynInput
    if (op_prim->HasAttr(kAttrDynInputSizes)) {
      if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE)) {
        // Like addn, prim define in python, but number of inputs change, so the value of kAttrDynInputSizes
        // changed too. In async, do opgrad may be not complete.
        ClonePrim(op_run_info);
      }
      if (!input_object->isa<ValueSequence>()) {
        auto dyn_v = GetValue<const std::vector<int64_t>>(op_prim->GetAttr(kAttrDynInputSizes));
        (void)dyn_v.emplace_back(-1);
        op_prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_v));
      }
    }
  }
  op_prim->EndRecordAddAttr();
  ReplaceValueNodeWithParameter(op_run_info, device_target);
  ReplaceReduceAxis(op_run_info);
}
}  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
