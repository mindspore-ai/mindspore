/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "include/pynative/utils/pynative_utils.h"

#include <string>
#include <utility>
#include <memory>
#include <algorithm>
#include <vector>
#include <set>

#include "ir/map_tensor.h"
#include "ir/tensor_new.h"
#include "primitive/sparse_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/pass_manager/op_adaptation_info_factory.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/utils/utils.h"
#include "include/utils/convert_utils_py.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/pynative/common_utils.h"
#include "include/utils/stub_tensor.h"
#include "pynative/backward/jit_grad/jit_grad.h"
#include "primitive/sequence_op_name.h"
#include "primitive/structure_ops.h"
#include "primitive/other_ops.h"
#include "pynative/utils/predict_out_type_map.h"
#include "include/pynative/utils/pyboost/auto_generate/contiguous.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/pynative/abstract_converter.h"
#include "include/utils/operator/primitive_utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "pynative/backward/grad_utils.h"
#include "include/utils/tensor_py.h"
#include "mindspore/ccsrc/include/utils/pynative/py_parse.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "include/utils/pynative/storage_py.h"
#include "ops_utils/op_utils.h"

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
// for simply infer (simple infer will push abs in bprop queue)
AbstractConverter kGradAbstractConverter;
const std::set<std::string> kVmOperators = {"InsertGradientOf", "StopGradient", "HookBackward", "CellBackwardHook"};
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

tensor::TensorPtr GetContiguousTensor(const tensor::TensorPtr &input_tensor, device::DeviceType device_target,
                                      bool requires_grad) {
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_target);
  auto contiguous_tensor = contiguous_op->Call(input_tensor);
  if (requires_grad) {
    contiguous_op->CreateOutputSimpleInfo();
    const auto &contiguous_run_info = std::make_shared<FrontendOpRunInfo>();
    contiguous_run_info->requires_grad = true;
    auto real_output = AutoGradUtil::MakeOutput(true, contiguous_op);
    AutoGradUtil::SetInferOutputToGrad(contiguous_run_info->op_grad_info, contiguous_op);
    PyBoost::UpdateStubOutput(contiguous_op, contiguous_run_info->stub_output, contiguous_op->output_abs(),
                              real_output);
    contiguous_run_info->base_op_run_info.device_target = device_target;
    contiguous_run_info->input_size = 1;
    contiguous_run_info->base_op_run_info.op_name = ops::kNameContiguous;
    contiguous_run_info->op_grad_info->op_prim = prim::kPrimContiguous;

    contiguous_run_info->op_grad_info->input_value = {input_tensor};
    contiguous_run_info->op_grad_info->out_value = real_output;
    PyBoost::DoGrad(contiguous_op, contiguous_run_info->op_grad_info, contiguous_run_info->async_status);
  }
  return contiguous_tensor;
}

bool IsMindSporeDtype(PyObject *obj) {
  PyObject *ms_module = PyImport_ImportModule("mindspore");
  if (ms_module && ms_module != Py_None) {
    PyObject *type_class = PyObject_GetAttrString(ms_module, "Type");
    Py_DECREF(ms_module);
    if (type_class && type_class != Py_None) {
      if (PyObject_IsInstance(obj, type_class)) {
        Py_DECREF(type_class);
        return true;
      }
      Py_DECREF(type_class);
    }
  }
  return false;
}

std::string BuildSequenceTypeString(PyObject *obj, bool is_tuple) {
  std::stringstream ss;
  ss << (is_tuple ? "Tuple<" : "List<");
  Py_ssize_t size = is_tuple ? PyTuple_Size(obj) : PyList_Size(obj);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject *item = is_tuple ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
    if (i == 0) {
      ss << PyParser::BuildPyObjectInputTypeString(item);
    } else {
      ss << ", " << PyParser::BuildPyObjectInputTypeString(item);
    }
  }
  ss << ">";
  return ss.str();
}

std::string GetDefaultObjectTypeName(PyObject *obj) {
  std::stringstream ss;
  PyObject *obj_type = PyObject_Str(PyObject_Type(obj));
  ss << PyUnicode_AsUTF8(obj_type);
  Py_DECREF(obj_type);
  return ss.str();
}
}  // namespace

std::string Common::GetIdByValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>()) {
    return "T" + std::to_string(v->cast<tensor::TensorPtr>()->id());
  }
  if (v->isa<stub::StubNode>()) {
    return GetIdByValue(v->cast<stub::StubNodePtr>()->WaitValue());
  }
  if (v->isa<Cell>()) {
    return v->cast<CellPtr>()->id();
  }
  if (v->isa<mindspore::Type>()) {
    auto type_ptr = v->cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  }
  if (v->isa<StringImm>()) {
    return "S" + v->cast<StringImmPtr>()->value();
  }
  if (v->isa<BoolImm>()) {
    return "B" + std::to_string(v->cast<BoolImmPtr>()->value());
  }
  if (v->isa<IntegerImm>()) {
    return "I" + std::to_string(v->cast<Int64ImmPtr>()->value());
  }
  if (v->isa<FloatImm>()) {
    return "F" + std::to_string(v->cast<FP32ImmPtr>()->value());
  }
  if (v->isa<None>()) {
    return "None";
  }
  if (v->isa<Ellipsis>()) {
    return "Ellipsis";
  }
  if (v->isa<ValueSequence>()) {
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

void Common::SplitString(const std::string &str, std::vector<std::string> *id_vec) {
  constexpr char colon_delim = ':';
  constexpr char angle_bracket_left_delim = '<';
  constexpr char angle_bracket_right_delim = '>';
  auto paren_pos = str.find_first_of(angle_bracket_left_delim);
  if (paren_pos == std::string::npos) {
    MS_LOG(EXCEPTION) << "Get wrong str " << str;
  }
  size_t str_size = str.size();
  const auto &sub_str = str.substr(paren_pos + 1, str_size - paren_pos - 2);
  MS_LOG(DEBUG) << "Ori str " << str << ", get sub str " << sub_str;
  size_t begin = 0;
  size_t angle_bracket_left = 0;
  size_t angle_bracket_right = 0;
  size_t sub_str_size = sub_str.size();
  for (size_t i = 0; i < sub_str_size; ++i) {
    switch (sub_str[i]) {
      case colon_delim:
        if (i != 0 && angle_bracket_left == angle_bracket_right) {
          (void)id_vec->emplace_back(sub_str.substr(begin, i - begin));
          begin = i + 1;
          angle_bracket_left = 0;
          angle_bracket_right = 0;
        }
        break;
      case angle_bracket_left_delim:
        ++angle_bracket_left;
        break;
      case angle_bracket_right_delim:
        ++angle_bracket_right;
        break;
      default: {
      }
    }
  }
  if (angle_bracket_left == angle_bracket_right) {
    (void)id_vec->emplace_back(sub_str.substr(begin, sub_str_size - begin));
  }
}

bool Common::IsTensor(const ValuePtr &v, bool include_sequence) {
  MS_EXCEPTION_IF_NULL(v);
  if (include_sequence) {
    if (v->isa<tensor::MetaSparseTensor>() || v->isa<tensor::Tensor>()) {
      return true;
    }
    if (v->isa<ValueSequence>()) {
      auto v_seq = v->cast<ValueSequencePtr>();
      if (v_seq->size() == 0) {
        MS_LOG(DEBUG) << "Get empty value sequence";
        return false;
      }
      // SpareTensor have scalar index, so just check have csr tensor
      if (v_seq->value().front()->isa<tensor::MetaSparseTensor>()) {
        return true;
      }
      // All value are tensor
      return std::all_of(v_seq->value().begin(), v_seq->value().end(),
                         [](const ValuePtr &e) { return IsTensor(e, true); });
    }
    MS_LOG(DEBUG) << "Get value " << v->ToString();
    return false;
  }
  MS_LOG(DEBUG) << "Get value " << v->ToString();
  return v->isa<tensor::Tensor>() || v->isa<tensor::MetaSparseTensor>();
}

bool Common::IsControlFlowGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

ValuePtr Common::FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>() || value->isa<tensor::COOTensor>() || value->isa<tensor::CSRTensor>()) {
    return value;
  }
  if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> value_list;
    auto value_seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_seq);
    for (auto &filter_value : value_seq->value()) {
      if (auto t = FilterSensValues(filter_value, dict_convert_to_tuple); t != nullptr) {
        (void)value_list.emplace_back(t);
      }
    }
    return std::make_shared<ValueTuple>(value_list);
  }
  if (value->isa<ValueDictionary>()) {
    if (dict_convert_to_tuple) {
      return FilterSensValues(DataConvert::ConvertValueDictToValueTuple(value), dict_convert_to_tuple);
    }
    return value;
  }
  MS_LOG(DEBUG) << "Value type: " << value->ToString();
  return nullptr;
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

const std::shared_ptr<PyNativeExecutor> &Common::GetPyNativeExecutor() {
  const auto &executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  return executor;
}

ValuePtr Common::StubNodeToValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue();
  }
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    bool has_stub =
      std::any_of(values.begin(), values.end(), [](const auto &v) { return utils::isa<stub::StubNode>(v); });
    if (!has_stub) {
      return v;
    }
    ValuePtrList value_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(value_list),
                         [](const ValuePtr &value) { return StubNodeToValue(value); });
    if (utils::isa<ValueTuple>(v)) {
      return std::make_shared<ValueTuple>(value_list);
    }
    if (utils::isa<ValueList>(v)) {
      return std::make_shared<ValueList>(value_list);
    }
    MS_LOG(EXCEPTION) << "Value not support ValueSequence " << v->ToString();
  } else {
    return v;
  }
}

void Common::StubNodeToValue(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info);
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    op_run_info->op_grad_info->input_value[i] = StubNodeToValue(op_run_info->op_grad_info->input_value[i]);
    // Contiguous tensor in Backend RunOp.
  }
}

tensor::TensorPtr Common::StubNodeToTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue()->cast<tensor::TensorPtr>();
  }
  if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>();
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor, but got " << v->ToString();
}

tensor::TensorPtr Common::ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous, bool requires_grad,
                                                  bool is_inplace) {
  const auto &tensor = StubNodeToTensor(v);
  MS_EXCEPTION_IF_NULL(tensor);
  if (!need_contiguous || tensor->storage_info() == nullptr) {
    return tensor;
  }
  auto device_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  const auto &device_target = device_address->GetDeviceType();
  static bool op_plugin_loaded = ops::GetOpPluginHandle() != nullptr;
  if (device_target == device::DeviceType::kAscend || (op_plugin_loaded && device_target == device::DeviceType::kCPU)) {
    return tensor;
  }
  if (is_inplace) {
    const auto &storage_info = tensor->storage_info();
    if (storage_info->storage_offset == 0 && storage_info->is_contiguous) {
      return tensor;
    } else {
      MS_LOG(WARNING) << "Tensor on CPU with storage_offset " << storage_info->storage_offset
                      << " and is_contiguous: " << storage_info->is_contiguous
                      << " cannot use CPU inplace copy normally";
    }
  }

  return GetContiguousTensor(tensor, device_target, requires_grad);
}

std::optional<tensor::TensorPtr> Common::ConvertStubNodeToTensor(const std::optional<ValuePtr> &v, bool need_contiguous,
                                                                 bool requires_grad, bool is_inplace) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToTensor(v.value(), need_contiguous, requires_grad, is_inplace));
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad,
                                                  bool is_inplace) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad, is_inplace](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad, is_inplace);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad,
                                                  bool is_inplace) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad, is_inplace](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad, is_inplace);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

std::optional<ValueTuplePtr> Common::ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                 bool need_contiguous, bool requires_grad,
                                                                 bool is_inplace) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToValueTuple(v.value(), need_contiguous, requires_grad, is_inplace));
}

ValueNodePtr Common::CreateValueNodeByValue(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  auto v_node = NewValueNode(v);
  if (abs == nullptr) {
    v_node->set_abstract(CommonUtils::SetAbstractValueToAnyValue(v->ToAbstract()));
  } else {
    v_node->set_abstract(abs);
  }
  return v_node;
}

void Common::ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (const auto &tensor : tensors) {
    tensor->set_device_address(nullptr);
    tensor->set_storage_info(nullptr);
  }
}

void Common::SetOutputUsedInBpropGraph(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &v_t = value->cast<tensor::TensorPtr>();
    v_t->set_used_in_bprop_graph(true);
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (const auto &v : value_seq->value()) {
      SetOutputUsedInBpropGraph(v);
    }
  }
  if (value->isa<stub::StubNode>()) {
    const auto &stub_node = value->cast<stub::StubNodePtr>();
    return SetOutputUsedInBpropGraph(stub_node->WaitValue());
  }
  if (value->isa<ValueDictionary>()) {
    auto dic_v = value->cast<ValueDictionaryPtr>();
    for (const auto &v : dic_v->value()) {
      SetOutputUsedInBpropGraph(v.second);
    }
  }
}

ValuePtr Common::CreateFakeValueWithoutDeviceAddress(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    const auto &v_t = value->cast<tensor::TensorPtr>();
    auto t = std::make_shared<tensor::Tensor>(*v_t);
    if (v_t->is_parameter()) {
      t->set_param_info(v_t->param_info());
    }
    // For view tensor, we need keep storage info for do grad.
    if (v_t->storage_info() != nullptr) {
      t->set_storage_info(v_t->storage_info());
    }
    t->set_device_address(nullptr);
    t->set_used_in_bprop_graph(false);
    return t;
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    ValuePtrList value_list;
    (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(value_list),
                         [](const ValuePtr &elem) { return CreateFakeValueWithoutDeviceAddress(elem); });
    return std::make_shared<ValueTuple>(value_list);
  }
  if (value->isa<stub::StubNode>()) {
    const auto &stub_node = value->cast<stub::StubNodePtr>();
    return CreateFakeValueWithoutDeviceAddress(stub_node->WaitValue());
  }
  if (value->isa<ValueDictionary>()) {
    auto dic_v = value->cast<ValueDictionaryPtr>();
    std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
    for (const auto &v : dic_v->value()) {
      (void)key_values.emplace_back(v.first, CreateFakeValueWithoutDeviceAddress(v.second));
    }
    return std::make_shared<ValueDictionary>(key_values);
  }
  return value;
}

void Common::SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &original_params = func_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  bool need_add_input_abs = op_run_info->op_grad_info->input_abs.empty();
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      op_run_info->op_grad_info->input_value_grad_type[i] =
        AutoGradUtil::SetValueGradInfo(op_run_info->op_grad_info->input_value[i], InputType::kConstant);
      if (need_add_input_abs) {
        (void)op_run_info->op_grad_info->input_abs.emplace_back(original_params[i]->abstract());
      }
      continue;
    }
    // Must weight param
    // Parameters current used in inner graph, and no used in outer graph
    const auto &param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    (void)op_run_info->op_grad_info->input_value.emplace_back(tensor_value);
    (void)op_run_info->op_grad_info->input_value_grad_type.emplace_back(AutoGradUtil::SetTensorGradInfo(tensor_value));
    (void)op_run_info->op_grad_info->input_abs.emplace_back(param->abstract());
    MS_LOG(DEBUG) << "Set graph weight parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

bool Common::IsHookNeedSaveInputs(const PrimitivePyPtr &prim) {
  if (prim->hook_type() == HookType::kCustomOpBprop || prim->hook_type() == HookType::kCellCustomBprop) {
    return true;
  }
  return false;
}

bool Common::IsVmOp(const std::string &op_name) { return kVmOperators.find(op_name) != kVmOperators.end(); }

std::vector<int64_t> Common::BuildShape(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->BuildShape();
  if (base_shape->isa<abstract::NoShape>()) {
    return {};
  }
  auto shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape->shape();
}

void Common::ClearRes() { kGradAbstractConverter.clear(); }

std::string PyParser::GetIdByPyObj(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return "T" + std::to_string(tensor::ConvertToTensor(obj)->id());
  }
  if (py::isinstance<Cell>(obj)) {
    return obj.cast<CellPtr>()->id();
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = obj.cast<mindspore::TypePtr>();
    return "Type:" + type_ptr->ToString();
  }
  if (py::isinstance<py::str>(obj)) {
    return "S" + obj.cast<std::string>();
  }
  if (py::isinstance<py::bool_>(obj)) {
    return "B" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::int_>(obj)) {
    return "I" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::float_>(obj)) {
    return "F" + py::str(obj).cast<std::string>();
  }
  if (py::isinstance<py::none>(obj)) {
    return "None";
  }
  if (py::isinstance<py::ellipsis>(obj)) {
    return "Ellipsis";
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    return GetIdForPyTupleOrList(obj);
  }
  if (py::isinstance<py::function>(obj)) {
    return GetFnInfoByPyObj(obj);
  }
  // For id with value and obj can be the same
  if (py::isinstance<tensor::CSRTensor>(obj) || py::isinstance<tensor::COOTensor>(obj) ||
      py::isinstance<tensor::RowTensor>(obj)) {
    return parse::data_converter::PyObjToValue(obj)->ToString();
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
  prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  op_run_info->signatures = prim->signatures();
  op_run_info->base_op_run_info.py_prim_id_ = adapter->id();
}

std::string PyParser::BuilidPyInputTypeString(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return "Tensor";
  }
  // bool must before int, because bool is a special int
  if (py::isinstance<py::bool_>(obj)) {
    return "bool";
  }
  if (py::isinstance<py::int_>(obj)) {
    return "int";
  }
  if (py::isinstance<py::float_>(obj)) {
    return "float";
  }
  if (py::isinstance<py::str>(obj)) {
    return "string";
  }
  if (py::isinstance<py::none>(obj)) {
    return "None";
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    return "mindspore.dtype";
  }

  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    std::stringstream ss;
    ss << (py::isinstance<py::tuple>(obj) ? "Tuple<" : "List<");
    auto tuple = py::cast<py::tuple>(obj);
    for (size_t i = 0; i < tuple.size(); i++) {
      if (i == 0) {
        ss << BuilidPyInputTypeString(tuple[i]);
      } else {
        ss << ", " << BuilidPyInputTypeString(tuple[i]);
      }
    }
    ss << ">";
    return ss.str();
  }

  std::stringstream ss;
  ss << obj.get_type();
  return ss.str();
}

std::string PyParser::BuildPyObjectInputTypeString(PyObject *obj) {
  if (tensor::IsPyObjectTensorPy(obj)) {
    return "Tensor";
  }
  if (IsPyObjectStoragePy(obj)) {
    return "Storage";
  }
  // bool must before int, because bool is a special int
  if (PyBool_Check(obj)) {
    return "bool";
  }
  if (PyLong_Check(obj)) {
    return "int";
  }
  if (PyFloat_Check(obj)) {
    return "float";
  }
  if (PyUnicode_Check(obj)) {
    return "string";
  }
  if (obj == Py_None) {
    return "None";
  }

  if (IsMindSporeDtype(obj)) {
    return "mindspore.dtype";
  }

  auto is_tuple = PyTuple_Check(obj);
  auto is_list = PyList_Check(obj);
  if (is_tuple || is_list) {
    return BuildSequenceTypeString(obj, is_tuple);
  }

  return GetDefaultObjectTypeName(obj);
}

void PyParser::PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx) {
  auto const &op_arg = op_def->args_[idx];
  bool is_suppport_tensor_cast = std::any_of(op_arg.cast_dtype_.begin(), op_arg.cast_dtype_.end(),
                                             [](const auto &type) { return type == ops::DT_TENSOR; });
  if (is_suppport_tensor_cast) {
    auto tensor = py_parse::ConvertTensorValue(op_inputs[idx]);
    auto PrintVectorFunc = [](const ShapeVector &shape) -> std::string {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) {
          ss << ", " << shape[i];
        } else {
          ss << shape[i];
        }
      }
      ss << "]";
      return ss.str();
    };
    if (tensor != nullptr) {
      MS_EXCEPTION(TypeError) << "For " << op_def->name_ << ", the " << idx << "'th input is a Tensor whose shape is "
                              << PrintVectorFunc(tensor->shape()) << " and dtype is ["
                              << TypeIdToString(tensor->data_type()) << "], which can not be converted to "
                              << prim::OpDTypeToString(op_arg.arg_dtype_) << ".";
    }
  }
  std::vector<std::string> op_type_list;
  for (size_t index = 0; index < op_inputs.size(); ++index) {
    (void)op_type_list.emplace_back(BuilidPyInputTypeString(op_inputs[index]));
  }
  PyNativeExecutor::GetInstance()->ClearRes();
  MS_EXCEPTION(TypeError) << prim::BuildOpErrorMsg(op_def, op_type_list);
}

void PyParser::PrintTypeCastErrorForPyObject(const ops::OpDefPtr &op_def, PyObject *op_inputs, size_t idx) {
  // op_inputs should be py::list
  auto const &op_arg = op_def->args_[idx];
  bool is_suppport_tensor_cast = std::any_of(op_arg.cast_dtype_.begin(), op_arg.cast_dtype_.end(),
                                             [](const auto &type) { return type == ops::DT_TENSOR; });
  if (is_suppport_tensor_cast) {
    PyObject *item = PyList_GetItem(op_inputs, idx);
    auto tensor = py_parse::ConvertPyObjectTensorValue(item);
    auto PrintVectorFunc = [](const ShapeVector &shape) -> std::string {
      std::stringstream ss;
      ss << "[";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) {
          ss << ", " << shape[i];
        } else {
          ss << shape[i];
        }
      }
      ss << "]";
      return ss.str();
    };
    if (tensor != nullptr) {
      MS_EXCEPTION(TypeError) << "For " << op_def->name_ << ", the " << idx << "'th input is a Tensor whose shape is "
                              << PrintVectorFunc(tensor->shape()) << " and dtype is ["
                              << TypeIdToString(tensor->data_type()) << "], which can not be converted to "
                              << prim::OpDTypeToString(op_arg.arg_dtype_) << ".";
    }
  }
  std::vector<std::string> op_type_list;
  Py_ssize_t inputs_size = (op_inputs && op_inputs != Py_None) ? PyList_Size(op_inputs) : 0;
  for (Py_ssize_t index = 0; index < inputs_size; ++index) {
    PyObject *item = PyList_GetItem(op_inputs, index);
    (void)op_type_list.emplace_back(BuildPyObjectInputTypeString(item));
  }
  PyNativeExecutor::GetInstance()->ClearRes();
  MS_EXCEPTION(TypeError) << prim::BuildOpErrorMsg(op_def, op_type_list);
}

inline ValuePtr ConvertScalarToTensor(const ValuePtr &value) {
  auto fp32_imm = value->cast<FP32ImmPtr>();
  if (fp32_imm != nullptr) {
    return tensor::from_scalar(fp32_imm->value());
  }

  auto bool_imm = value->cast<BoolImmPtr>();
  if (bool_imm != nullptr) {
    return tensor::from_scalar(bool_imm->value());
  }

  auto int64_imm = value->cast<Int64ImmPtr>();
  if (int64_imm != nullptr) {
    return tensor::from_scalar(int64_imm->value());
  }

  MS_LOG(EXCEPTION) << "Unsupported type: " << value->ToString();
}

inline ValuePtr ConvertBySignature(const py::object &obj, const FrontendOpRunInfoPtr &op_run_info, size_t index) {
  if (op_run_info->signatures.size() <= index) {
    return nullptr;
  }

  if (op_run_info->signatures[index].dtype != SignatureEnumDType::kDTypeEmptyDefaultValue) {
    auto convert_func = py_parse::GetConverterByType(static_cast<int32_t>(ops::DT_NUMBER));
    MS_EXCEPTION_IF_NULL(convert_func);
    return convert_func(obj);
  }
  return nullptr;
}

void ParseOpInputByOpDef(const ops::OpDefPtr &op_def, const py::list &op_inputs, bool stub,
                         const FrontendOpRunInfoPtr &op_run_info) {
  size_t input_size = op_inputs.size();
  if (input_size != op_def->args_.size()) {
    MS_LOG(EXCEPTION) << "For Operator[" << op_def->name_ << "], the inputs number should be " << op_def->args_.size()
                      << " but got " << op_inputs.size() << ".";
  }
  (void)op_run_info->op_grad_info->input_value.resize(input_size);
  for (size_t i = 0; i < op_def->args_.size(); i++) {
    auto const &op_arg = op_def->args_[i];
    op_run_info->none_init_inputs_num += static_cast<size_t>(!op_arg.as_init_arg_);

    // Optional argument is valid for None as input.
    if (op_arg.is_optional_ && py::isinstance<py::none>(op_inputs[i])) {
      op_run_info->op_grad_info->input_value[i] = kNone;
      continue;
    }

    ValuePtr value = nullptr;
    py_parse::OpDefConvertFunc convert_func = py_parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
    MS_EXCEPTION_IF_NULL(convert_func);
    value = convert_func(op_inputs[i]);
    if (value != nullptr) {
      op_run_info->op_grad_info->input_value[i] = value;
      continue;
    }

    // type cast has lower priority then signature cast
    if (!op_arg.cast_dtype_.empty()) {
      for (auto cast_dtype : op_arg.cast_dtype_) {
        convert_func = py_parse::GetConverterByType(py_parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
        MS_EXCEPTION_IF_NULL(convert_func);
        value = convert_func(op_inputs[i]);
        if (value != nullptr) {
          if (value->isa<tensor::Tensor>()) {
            value->cast<tensor::TensorPtr>()->set_source_type(cast_dtype);
          }
          op_run_info->op_grad_info->input_value[i] = value;
          op_run_info->source_type[i] = cast_dtype;
          break;
        }
      }
    }

    if (value == nullptr) {
      PyParser::PrintTypeCastError(op_def, op_inputs, i);
    }
  }
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs, bool stub) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  op_run_info->source_type.resize(op_run_info->input_size);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);

  auto op_def = mindspore::ops::GetOpDef(op_run_info->base_op_run_info.op_name);
  if (op_def == nullptr) {
    op_run_info->op_grad_info->input_value.resize(op_run_info->input_size);
    op_run_info->none_init_inputs_num = op_run_info->input_size;
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      op_run_info->op_grad_info->input_value[i] = parse::data_converter::PyObjToValue(op_inputs[i], stub);
    }
  } else {
    op_run_info->none_init_inputs_num = 0;
    ParseOpInputByOpDef(op_def, op_inputs, stub, op_run_info);
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
    if (tensor->unsafe_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
      return false;
    }
  }
  (void)op_run_info->op_grad_info->op_prim->AddAttr(input_name, v);
  return true;
}

ValuePtr DataConvert::ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs) {
  if (values.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "tensors size should not be empty!";
  }
  if (values.size() == kSizeOne && !abs->isa<abstract::AbstractSequence>()) {
    return values[kIndex0];
  }
  return std::make_shared<ValueTuple>(values);
}

ValuePtrList DataConvert::TensorListToValueList(const tensor::TensorPtrList &tensor_list) {
  ValuePtrList output_values;
  output_values.reserve(tensor_list.size());
  (void)std::transform(tensor_list.begin(), tensor_list.end(), std::back_inserter(output_values),
                       [](const TensorPtr &tensor) -> ValuePtr {
                         if (tensor == nullptr) return kNone;
                         return tensor;
                       });
  return output_values;
}

PyboostOpRunInfoPtr PyBoost::Init_Pyboost(const PrimitivePtr &prim) {
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &op_run_info = std::make_shared<PyboostOpRunInfo>();
  op_run_info->op_prim = prim;
  pynative_executor->StoreAsyncStatus(op_run_info);
  forward_executor->InitOpRunInfo(op_run_info);
  return op_run_info;
}

FrontendOpRunInfoPtr PyBoost::Init(const PrimitivePtr &prim) {
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  pynative_executor->StoreAsyncStatus(op_run_info);
  forward_executor->InitOpRunInfo(op_run_info);
  return op_run_info;
}

void PyBoost::UpdateStubOutput(const kernel::pyboost::OpPtr &op, const stub::StubNodePtr &stub_output,
                               const AbstractBasePtr &abstract, const ValuePtr &real_out) {
  MS_EXCEPTION_IF_NULL(op);
  if (stub_output == nullptr || stub_output->isa<stub::NoneTypeNode>()) {
    return;
  }
  if (MS_UNLIKELY(op->output_value_simple_info() != nullptr)) {
    stub_output->SetValueSimpleInfo(op->output_value_simple_info());
  } else {
    MS_EXCEPTION_IF_NULL(abstract);
    auto success = stub_output->SetAbstract(abstract);
    if (!success) {
      MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                              << PredictOutTypeByName(op->primitive()->name()) << ", infer type is "
                              << abstract->BuildType() << ", the name of operator is [" << op->primitive()->name()
                              << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
    }
    MS_LOG(DEBUG) << "Update StubNode abstract " << abstract->ToString();
  }
  stub_output->SetValue(real_out);
}

PrimitivePtr PyBoost::ConvertPrimitive(const py::object &obj) {
  const auto &adapter = obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);

  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
#ifndef ENABLE_TEST
    // Custom operator's infer type and backpropagation are defined on the Python side.
    if (adapter->name() != kCustomExtOpName && adapter->name() != ops::kNameCellBackwardHook) {
      return std::make_shared<Primitive>(adapter->name(), adapter->attrs());
    }
    prim = std::make_shared<PrimitivePy>(obj);
    adapter->set_attached_primitive(prim);
#else
    prim = std::make_shared<PrimitivePy>(obj);
    adapter->set_attached_primitive(prim);
#endif
  }
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  prim->EnableSharedMutex();
  return prim;
}

py::object PyBoost::RunPyFunction(const PrimitivePtr &prim, const py::list &args) {
  py::tuple wrap_args(kIndex3);
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (!prim_py->HasPyObj()) {
      MS_LOG(EXCEPTION) << "Prim has not python obj!";
    }
    wrap_args[kIndex0] = prim_py->GetPyObj();
  } else {
    wrap_args[kIndex0] = std::make_shared<PrimitivePyAdapter>(prim->name());
  }
  wrap_args[kIndex1] = prim->name();
  wrap_args[kIndex2] = args;
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  return pynative_executor->RunOpStub(wrap_args);
}

void PyBoost::DoGrad(const kernel::pyboost::OpPtr &op, const OpGradInfoPtr &grad_info,
                     const AsyncStatus &async_status) {
  static const std::string kDoGradName = "DoGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);

  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();
  if (op->output_value_simple_info() == nullptr) {
    MS_LOG(EXCEPTION) << "The simple info of " << op->primitive()->name() << " infer is null";
  }
  MarkPyBoostInputs(grad_info);
  if (op->clone_tensor() != nullptr) {
    grad_info->input_value[0] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(grad_info->input_value[0]);
    grad_info->clone_value = op->clone_tensor();
  }
  forward->ForwardOpGradImpl(grad_info, async_status);
}

void PyBoost::DoGrad(const OpGradInfoPtr &grad_info, const AsyncStatus &async_status) {
  static const std::string kDoGradName = "DoGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);

  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();
  MarkPyBoostInputs(grad_info);
  forward->ForwardOpGradImpl(grad_info, async_status);
}

void PyBoost::MarkSideEffect(PyObject *arg) {
  if (tensor::IsTensorPy(arg)) {
    tensor::PyType<tensor::TensorPy> *tensor = reinterpret_cast<tensor::PyType<tensor::TensorPy> *>(arg);
    tensor->value.set_has_side_effect(true);
    return;
  }
  if (PyTuple_Check(arg)) {
    Py_ssize_t tup_size = PyTuple_Size(arg);
    for (Py_ssize_t i = 0; i < tup_size; ++i) {
      MarkSideEffect(PyTuple_GetItem(arg, i));
    }
  }
}

void PyBoost::MarkPyBoostInputs(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  size_t input_size = op_grad_info->input_value.size();
  op_grad_info->input_value_grad_type.resize(input_size);
  for (size_t index = 0; index < input_size; ++index) {
    const auto &v = op_grad_info->input_value[index];
    if (v->isa<tensor::Tensor>()) {
      op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(v->cast<tensor::TensorPtr>());
    } else if (v->isa<ValueSequence>()) {
      const auto &value_sequence = v->cast<ValueSequencePtr>();
      const auto &tuple_inputs = value_sequence->value();
      if (!tuple_inputs.empty() && tuple_inputs[0]->isa<tensor::Tensor>()) {
        op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
        for (const auto &elem : tuple_inputs) {
          auto grad_type = AutoGradUtil::SetTensorGradInfo(elem->cast<tensor::TensorPtr>());
          if (AutoGradUtil::IsParam(grad_type)) {
            op_grad_info->input_value_grad_type[index] = InputType::kParameter;
          }
        }
      }
    } else if (v->isa<tensor::MapTensor>()) {
      op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(v->cast<tensor::MapTensorPtr>());
    } else if (v->isa<tensor::CSRTensor>()) {
      const auto &csr_tensor = v->cast<tensor::CSRTensorPtr>();
      auto fn = [&op_grad_info, index](const auto &csr_tensor_input) {
        auto grad_type = AutoGradUtil::SetTensorGradInfo(csr_tensor_input);
        if (AutoGradUtil::IsParam(grad_type)) {
          op_grad_info->input_value_grad_type[index] = InputType::kParameter;
        }
      };
      op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
      fn(csr_tensor->GetIndptr());
      fn(csr_tensor->GetIndices());
      fn(csr_tensor->GetValues());
    }
  }
}

void PyBoost::BumpVersionAsync(tensor::Version version) {
  const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
  if (forward->enable_async()) {
    const auto task = [version]() mutable { version.BumpVersion(); };
    const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
    bprop_queue->Push(std::make_shared<BpropTask>(task));
  } else {
    version.BumpVersion();
  }
}

void PyBoost::UpdateVersionAsync(const autograd::ViewAutoGradMetaDataPtr &view_meta, const tensor::Version &version) {
  const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
  if (forward->enable_async()) {
    const auto task = [view_meta, version]() { view_meta->set_version_attr(version.current_version()); };
    const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
    bprop_queue->Push(std::make_shared<BpropTask>(task));
  } else {
    view_meta->set_version_attr(version.current_version());
  }
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
  }
  for (const auto &v : value_seq->value()) {
    if (!v->isa<tensor::Tensor>()) {
      MS_LOG(DEBUG) << "Get value " << v->ToString() << " in tensor tuple, op name "
                    << op_run_info->base_op_run_info.op_name;
      (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
      (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
      continue;
    }
    InputType input_type = InputType::kInput;
    auto tensor = v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      auto grad_type = AutoGradUtil::SetTensorGradInfo(tensor);
      if (AutoGradUtil::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kParameter;
      }
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(input_type);
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

ValuePtr DataConvert::ConvertValueDictToValueTuple(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  const auto &dic_v = v->cast<ValueDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(dic_v);
  std::vector<ValuePtr> v_list;
  (void)std::transform(dic_v->value().begin(), dic_v->value().end(), std::back_inserter(v_list),
                       [](const std::pair<ValuePtr, ValuePtr> &elem) { return elem.second; });
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::ConvertMapTensor(const FrontendOpRunInfoPtr &op_run_info, const tensor::MapTensorPtr &map_tensor,
                                   size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(map_tensor);
  constexpr int input_num = 1;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(map_tensor);
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kParameter);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(map_tensor);
  }
}

void DataConvert::ConvertCSRTensorToTensorList(const FrontendOpRunInfoPtr &op_run_info,
                                               const tensor::CSRTensorPtr &csr_tensor, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(csr_tensor);
  constexpr int input_num = 3;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }

  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kInput);
  op_run_info->op_grad_info->op_prim->set_attr("is_csr", MakeValue(true));
  op_run_info->op_grad_info->op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
    for (int i = 0; i < input_num; ++i) {
      auto iter = op_run_info->base_op_run_info.expanded_input_values.rbegin() + i;
      auto grad_type = AutoGradUtil::SetTensorGradInfo((*iter)->cast<tensor::TensorPtr>());
      if (AutoGradUtil::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kParameter;
      }
    }
  }
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::Tensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, index);
  } else {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
  }
}

void DataConvert::MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  tensor::TensorPtr tensor_ptr = nullptr;
  InputType input_type = InputType::kInput;
  if (v->isa<tensor::Tensor>()) {
    tensor_ptr = v->cast<tensor::TensorPtr>();
    if (tensor_ptr->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      op_run_info->op_grad_info->input_value_grad_type[index] = AutoGradUtil::SetTensorGradInfo(tensor_ptr);
    }
  } else if (v->isa<BoolImm>() || v->isa<FloatImm>() || v->isa<Type>() || v->isa<StringImm>() || v->isa<None>()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  } else if (v->isa<IntegerImm>()) {
    if (op_run_info->base_op_run_info.op_name == prim::kPrimCSRReduceSum->name()) {
      int64_t input = v->cast<Int64ImmPtr>()->value();
      op_run_info->op_grad_info->op_prim->set_attr("axis", MakeValue(input));
      return;
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  } else if (v->isa<ValueSequence>()) {
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), index);
    return;
  } else if (v->isa<ValueDictionary>()) {
    auto v_dict = v->cast<ValueDictionaryPtr>();
    std::vector<ValuePtr> vec;
    vec.reserve(v_dict->value().size());
    for (const auto &kv : v_dict->value()) {
      (void)vec.emplace_back(kv.second);
    }
    ConvertTupleValueToTensor(op_run_info, std::make_shared<ValueTuple>(vec), index);
    return;
  } else if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), index);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), index);
    return;
  } else if (v->isa<Monad>()) {
    return;
  } else if (v->isa<parse::InterpretedObject>()) {
    MS_EXCEPTION(TypeError) << "Not support for " << v->ToString();
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(tensor_ptr);
  (void)op_run_info->base_op_run_info.input_types.emplace_back(input_type);
}

void ReplaceReduceAxis(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!common::AnfAlgo::IsReduceOp(op_run_info->base_op_run_info.op_name)) {
    return;
  }
  const auto &inputs = op_run_info->base_op_run_info.expanded_input_values;
  constexpr size_t kReduceOpInputNum = 2;
  if (inputs.size() < kReduceOpInputNum) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size " << inputs.size() << " of Op "
                      << op_run_info->base_op_run_info.op_name;
  }

  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info);
  const auto &op_prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(op_prim);
  if (op_prim->HasAttr(kAttrSkipMode) && GetValue<bool>(op_prim->GetAttr(kAttrSkipMode))) {
    return;
  }

  // 2nd input tensor is {} or nulltpr, means reduce all axis.
  bool reduce_all_axis = false;
  if (inputs[kIndex1]->isa<ValueSequence>()) {
    auto seq_size = inputs[1]->cast<ValueSequencePtr>()->size();
    reduce_all_axis = seq_size == 0;
  } else if (inputs[kIndex1]->isa<None>()) {
    reduce_all_axis = true;
  }
  if (reduce_all_axis) {
    auto size = inputs[0]->cast<tensor::TensorPtr>()->shape().size();
    // For example, input 0 is Tensor(shape=[], value=1), the axis to reduce is 0.
    std::vector<ValuePtr> axis = {std::make_shared<Int64Imm>(0)};
    for (size_t i = 1; i < size; ++i) {
      axis.push_back(std::make_shared<Int64Imm>(static_cast<int64_t>(i)));
    }
    op_run_info->base_op_run_info.expanded_input_values[1] = std::make_shared<ValueTuple>(axis);
  }
}

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);

  (void)op_run_info->base_op_run_info.expanded_input_values.reserve(op_run_info->input_size);
  (void)op_run_info->base_op_run_info.input_types.reserve(op_run_info->input_size);
  // Get input tensors.
  op_run_info->op_grad_info->op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
    // convert const input to attr
    if (RunOpConvertConstInputToAttr(op_run_info, input_object, index)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    MarkInputs(op_run_info, input_object, index);
    // -1 indicates input_object is not a dynInput
    if (!op_run_info->base_op_run_info.dyn_input_sizes.empty() && !input_object->isa<ValueSequence>()) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
  }
  op_run_info->op_grad_info->op_prim->EndRecordAddAttr();
  ReplaceReduceAxis(op_run_info);
  AddDynInputsSizesAttr(op_run_info);
}

void Common::DoGradInner(runtime::OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
  if (!kernel::pyboost::OpRunStatus::Get().RequireGrad()) {
    return;
  }
  MS_LOG(DEBUG) << "Begin DoGradInner";
  auto grad_info = std::make_shared<OpGradInfo>();
  grad_info->input_value = op_runner_info->inputs;
  bool is_out_sequence = op_runner_info->output_abs->isa<abstract::AbstractSequence>();
  grad_info->out_value = AutoGradUtil::VectorRefToValue(*op_outputs, true, is_out_sequence);
  grad_info->op_prim = op_runner_info->prim;
  grad_info->out_abs = op_runner_info->output_abs;
  grad_info->input_abs = op_runner_info->inputs_abs;
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();
  AsyncStatus status;
  PyBoost::MarkPyBoostInputs(grad_info);
  forward->ForwardOpGradImpl(grad_info, status);
  MS_LOG(DEBUG) << "End DoGradInner";
}

tensor::TensorPtr Common::GetTensorFromSparseTensor(const ValuePtr &val) {
  if (val->isa<tensor::Tensor>()) {
    return val->cast<tensor::TensorPtr>();
  } else if (val->isa<tensor::CSRTensor>()) {
    auto csr_tensor = val->cast<tensor::CSRTensorPtr>();
    return csr_tensor->GetValues();
  } else if (val->isa<tensor::COOTensor>()) {
    auto coo_tensor = val->cast<tensor::COOTensorPtr>();
    return coo_tensor->GetValues();
  }
  return nullptr;
}

void Common::WaitBprop() { return runtime::Pipeline::Get().WaitBpropStage(); }

tensor::TensorPtr Common::CaculateGradNorm(const tensor::TensorPtr &grad) {
  if (grad->Dtype()->type_id() == kNumberTypeBool) {
    return grad;
  }
  static constexpr const float norm_val = 2;
  kernel::pyboost::OpStatus status{false, DeviceManagerConf::GetInstance()->device_type()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
  return kernel::pyboost::norm(grad, std::make_shared<FP32Imm>(norm_val), std::nullopt,
                               std::make_shared<BoolImm>(false), std::nullopt);
}
}  // namespace PyNativeAlgo

void RegOpCall(const py::module *m) {
  (void)py::class_<OpCall, std::shared_ptr<OpCall>>(*m, "OpCall")
    .def(py::init<>())
    .def("__call__", &OpCall::operator())
    .def_property_readonly("name", &OpCall::name)
    .def("__str__", [](const std::shared_ptr<OpCall> &op_call) { return "<OpCall name=" + op_call->name() + ">"; })
    .def("__repr__", [](const std::shared_ptr<OpCall> &op_call) { return "<OpCall name=" + op_call->name() + ">"; });

  (void)py::class_<TensorOverloadCall, std::shared_ptr<TensorOverloadCall>>(*m, "TensorOverloadCall")
    .def(py::init<>())
    .def("__call__", &TensorOverloadCall::operator())
    .def_property_readonly("name", &TensorOverloadCall::name)
    .def("__str__",
         [](const std::shared_ptr<TensorOverloadCall> &op_call) { return "<OpCall name=" + op_call->name() + ">"; })
    .def("__repr__",
         [](const std::shared_ptr<TensorOverloadCall> &op_call) { return "<OpCall name=" + op_call->name() + ">"; });
}

void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task) {
  GilReleaseWithCheck no_gil;
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    MS_LOG(INFO) << "PyBoost sync run frontend task";
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}
}  // namespace pynative
}  // namespace mindspore
