/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "ops/sparse_ops.h"
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "pybind_api/ir/primitive_py.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/primfunc_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/utils/stub_tensor.h"
#include "frontend/expander/bprop/bprop.h"
#include "pipeline/pynative/grad/jit/jit_grad.h"
#include "ops/sequence_op_name.h"
#include "ops/structure_ops.h"
#include "ops/other_ops.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "kernel/pyboost/auto_generate/contiguous.h"

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

ValuePtr CreateNonTensorByAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto type_id = pynative::PyNativeAlgo::Common::GetTypeFromAbstract(abs);
  if (abs->isa<abstract::AbstractMonad>()) {
    return std::make_shared<tensor::Tensor>(0);
  }
  if (type_id == kMetaTypeNone) {
    return kNone;
  }
  if (type_id == kMetaTypeNull) {
    return kNull;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    ValuePtrList value_ptr_list;
    (void)std::transform(abs_seq.begin(), abs_seq.end(), std::back_inserter(value_ptr_list),
                         [](const abstract::AbstractBasePtr &elem) { return CreateNonTensorByAbstract(elem); });
    return std::make_shared<ValueTuple>(value_ptr_list);
  }
  if (type_id == kNumberTypeBool) {
    return MakeValue(true);
  } else if (type_id == kObjectTypeString) {
    return MakeValue("");
  } else if (type_id >= kNumberTypeInt && type_id <= kNumberTypeUInt64) {
    return MakeValue(static_cast<int64_t>(0));
  } else if (type_id >= kNumberTypeFloat && type_id <= kNumberTypeFloat64) {
    return MakeValue(static_cast<float>(0));
  } else if (type_id == kNumberTypeDouble) {
    return MakeValue(static_cast<double>(0));
  } else {
    MS_LOG(EXCEPTION) << "Get unsupported type " << type_id;
  }
}

void PlantTupleParam(const FuncGraphPtr &bprop_graph, const abstract::AbstractSequencePtr &abs_seq,
                     AnfNodePtrList *make_tuple, AnfNodePtrList *new_param) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  MS_EXCEPTION_IF_NULL(make_tuple);
  MS_EXCEPTION_IF_NULL(new_param);
  MS_EXCEPTION_IF_NULL(abs_seq);
  for (size_t i = 0; i < abs_seq->size(); ++i) {
    if (abs_seq->elements()[i]->isa<abstract::AbstractSequence>()) {
      PlantTupleParam(bprop_graph, abs_seq->elements()[i]->cast<abstract::AbstractSequencePtr>(), make_tuple,
                      new_param);
    } else if (abs_seq->elements()[i]->isa<abstract::AbstractTensor>()) {
      auto plant_param = bprop_graph->add_parameter();
      plant_param->set_abstract(abs_seq->elements()[i]);
      (void)make_tuple->emplace_back(plant_param);
      (void)new_param->emplace_back(plant_param);
    }
  }
}

const mindspore::HashSet<std::string> kNotRealOP{
  kMakeTupleOpName,
  kMakeListNewOpName,
  kTupleGetItemOpName,
  kStopGradientOpName,
  kUpdateStateOpName,
  kLoadOpName,
  kDependOpName,
  kReturnOpName,
  kNPUAllocFloatStatusOpName,
  kNPUGetFloatStatusOpName,
  kNPUClearFloatStatusOpName,
  kMirrorOperatorOpName,
  kSequenceSliceOpName,
  kSequenceMulOpName,
  kPyExecuteOpName,
};
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

std::string Common::GetCellId(const std::string &obj_id, const std::vector<std::string> &input_arg_id_vec,
                              const std::vector<ValuePtr> &input_arg_value_vec) {
  auto cell_id = obj_id;
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  const auto &forward = GetPyNativeExecutor()->forward_executor();
  for (size_t i = 0; i < input_arg_id_vec.size(); ++i) {
    const auto &arg_id = input_arg_id_vec[i];
    // Find in step process
    auto cache_abs = forward->GetNodeAbsById(arg_id);
    if (cache_abs != nullptr) {
      fn(cache_abs);
    } else {
      MS_EXCEPTION_IF_NULL(input_arg_value_vec[i]);
      fn(SetAbstractValueToAnyValue(input_arg_value_vec[i]->ToAbstract()));
    }
  }
  return cell_id;
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
                         [](const ValuePtr &e) { return IsTensor(e, true); });
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
      if (auto t = FilterSensValues(filter_value); t != nullptr) {
        (void)value_list.emplace_back(t);
      }
    }
    return std::make_shared<ValueTuple>(value_list);
  } else if (value->isa<ValueDictionary>()) {
    return FilterSensValues(DataConvert::ConvertValueDictToValueTuple(value));
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

const std::shared_ptr<PyNativeExecutor> &Common::GetPyNativeExecutor() {
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
  auto type_id = GetTypeFromAbstract(abs);
  if (abs->isa<abstract::AbstractMonad>()) {
    return std::make_shared<tensor::Tensor>(0);
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    std::vector<ValuePtr> out;
    if (!abs_seq->elements().front()->isa<abstract::AbstractTensor>()) {
      MS_LOG(DEBUG) << "Get non tensor output";
      return CreateNonTensorByAbstract(abs);
    }
    for (size_t i = 0; i < abs_seq->size(); ++i) {
      (void)out.emplace_back(std::make_shared<tensor::Tensor>(type_id, GetShapeFromAbstract(abs_seq->elements()[i])));
    }
    return std::make_shared<ValueTuple>(out);
  }
  if (!abs->isa<abstract::AbstractTensor>()) {
    MS_LOG(DEBUG) << "Get non tensor output";
    return CreateNonTensorByAbstract(abs);
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
  DumpGraphIR("replace_cnode_with_valuenode.ir", bprop_graph);
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
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info);
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    op_run_info->op_grad_info->input_value[i] = StubNodeToValueInner(op_run_info->op_grad_info->input_value[i]);
  }
}

TensorPtr Common::StubNodeToTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue()->cast<tensor::TensorPtr>();
  } else if (v->isa<tensor::Tensor>()) {
    return v->cast<tensor::TensorPtr>();
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor, but got " << v->ToString();
}

tensor::TensorPtr Common::ConvertToContiguousTensor(const tensor::TensorPtr &tensor, const std::string &device_target) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->storage_info() == nullptr) {
    return tensor;
  }

  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_target);
  return contiguous_op->Call(tensor);
}

TensorPtr Common::ConvertStubNodeToTensor(const ValuePtr &v, const std::string &device_target, bool need_contiguous) {
  const auto &tensor = StubNodeToTensor(v);
  if (!need_contiguous || device_target == kAscendDevice) {
    return tensor;
  }
  return ConvertToContiguousTensor(tensor, device_target);
}

std::optional<tensor::TensorPtr> Common::ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                 const std::string &device_target,
                                                                 bool need_contiguous) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToTensor(v.value(), device_target, need_contiguous));
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValuePtr &v, const std::string &device_target,
                                                  bool need_contiguous) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [&device_target, need_contiguous](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, device_target, need_contiguous);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

void Common::GetConstInputToAttr(const PrimitivePtr &op_prim, const std::string &op_name,
                                 const std::string &device_target, bool is_dynamic_shape,
                                 mindspore::HashSet<size_t> *input_to_attr_index) {
  if (op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    mindspore::HashSet<size_t> attr_indexes;
    PrimitiveReadLock read_lock(op_prim->shared_mutex());
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
  auto t = std::make_shared<tensor::Tensor>(*tensor);
  if (tensor->is_parameter()) {
    t->set_param_info(tensor->param_info());
  }
  t->set_device_address(nullptr);
  t->set_storage_info(nullptr);
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
    t->set_storage_info(nullptr);
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
    if (top_cell != nullptr && IsParam(grad_type)) {
      top_cell->AddParamGradInfo(tensor_value, auto_grad_meta_data);
    }
    return grad_type;
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    TensorGradType ret_type = grad_type;
    for (const auto &v : value_seq) {
      auto ret = SetValueGradInfo(v, top_cell, grad_type);
      if (IsParam(ret)) {
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
  } else if (value->isa<ValueDictionary>()) {
    const auto &dic_v = value->cast<ValueDictionaryPtr>()->value();
    for (const auto &v : dic_v) {
      SetValueGradInfo(v.second, top_cell, grad_type);
    }
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

void Common::SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph,
                                         const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &original_params = func_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  bool need_add_input_abs = op_run_info->op_grad_info->input_abs.empty();
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      op_run_info->op_grad_info->input_value_grad_type[i] =
        SetValueGradInfo(op_run_info->op_grad_info->input_value[i], top_cell, TensorGradType::kConstant);
      if (need_add_input_abs) {
        (void)op_run_info->op_grad_info->input_abs.emplace_back(original_params[i]->abstract());
      }
      continue;
    }
    // Must weight param
    const auto &param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    (void)op_run_info->op_grad_info->input_value.emplace_back(tensor_value);
    (void)op_run_info->op_grad_info->input_value_grad_type.emplace_back(SetTensorGradInfo(tensor_value, top_cell));
    (void)op_run_info->op_grad_info->input_abs.emplace_back(param->abstract());
    MS_LOG(DEBUG) << "Set graph weight parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

void Common::ProcessTupleParam(const FuncGraphPtr &bprop_graph, size_t position) {
  auto bprop_params = bprop_graph->parameters();
  auto target_param = bprop_params[position];
  MS_EXCEPTION_IF_NULL(target_param);
  const auto &target_abstract = target_param->abstract();
  MS_EXCEPTION_IF_NULL(target_abstract);
  if (!target_abstract->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get wrong param " << target_abstract->ToString();
  }
  MS_LOG(DEBUG) << "Process tuple param " << target_abstract->ToString();
  auto it = std::find(bprop_params.begin(), bprop_params.end(), target_param);
  it = bprop_params.erase(it);
  const auto &abs_seq = target_abstract->cast<abstract::AbstractSequencePtr>();
  AnfNodePtrList make_tuple{NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList new_param;
  PlantTupleParam(bprop_graph, abs_seq, &make_tuple, &new_param);
  (void)bprop_params.insert(it, new_param.begin(), new_param.end());
  bprop_graph->set_parameters(bprop_params);
  auto make_tuple_param = bprop_graph->NewCNode(make_tuple);
  make_tuple_param->set_abstract(target_abstract);
  auto manager = bprop_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto tr = manager->Transact();
  (void)tr.Replace(target_param, make_tuple_param);
  tr.Commit();
}

void Common::FreeFuncGraphForwardNodes(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->used_forward_nodes().empty()) {
    return;
  }
  for (const auto &node : func_graph->used_forward_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_forward(nullptr, "");
  }
  func_graph->ClearUsedForwardNodes();
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

std::pair<std::vector<std::string>, std::vector<ValuePtr>> PyParser::GetArgsIdAndValue(const py::args &args) {
  size_t arg_size = args.size();
  std::vector<std::string> input_arg_id_vec;
  std::vector<ValuePtr> input_arg_value_vec;
  input_arg_id_vec.reserve(arg_size);
  input_arg_value_vec.reserve(arg_size);
  for (size_t i = 0; i < arg_size; ++i) {
    (void)input_arg_value_vec.emplace_back(DataConvert::PyObjToValue(args[i]));
    (void)input_arg_id_vec.emplace_back(Common::GetIdByValue(input_arg_value_vec.back()));
  }
  return {input_arg_id_vec, input_arg_value_vec};
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

static std::string BuilidPyInputTypeString(const py::object &obj) {
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

  if (py::isinstance<mindspore::tensor::Tensor>(obj)) {
    return "Tensor";
  }

  if (IsStubTensor(obj)) {
    return "Tensor";
  }

  if (py::isinstance<py::tuple>(obj)) {
    std::stringstream ss;
    ss << "tuple<";
    auto tuple = obj.cast<py::tuple>();
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

  if (py::isinstance<py::list>(obj)) {
    std::stringstream ss;
    ss << "list<";
    auto list = obj.cast<py::list>();
    for (size_t i = 0; i < list.size(); i++) {
      if (i == 0) {
        ss << BuilidPyInputTypeString(list[i]);
      } else {
        ss << ", " << BuilidPyInputTypeString(list[i]);
      }
    }
    ss << ">";
    return ss.str();
  }

  std::stringstream ss;
  ss << obj.get_type();
  return ss.str();
}

inline ValuePtr ConvertScalarToTensor(const ValuePtr &value) {
  auto fp32_imm = value->cast<FP32ImmPtr>();
  if (fp32_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(fp32_imm->value());
  }

  auto bool_imm = value->cast<BoolImmPtr>();
  if (bool_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(bool_imm->value());
  }

  auto int64_imm = value->cast<Int64ImmPtr>();
  if (int64_imm != nullptr) {
    return std::make_shared<tensor::Tensor>(int64_imm->value());
  }

  MS_LOG(EXCEPTION) << "Unsupported type: " << value->ToString();
}

inline ValuePtr ConvertBySignature(const py::object &obj, const FrontendOpRunInfoPtr &op_run_info, size_t index) {
  if (op_run_info->signatures.size() <= index) {
    return nullptr;
  }

  if (op_run_info->signatures[index].dtype != SignatureEnumDType::kDTypeEmptyDefaultValue) {
    auto convert_func = parse::GetConverterByType(static_cast<int32_t>(ops::DT_NUMBER));
    MS_EXCEPTION_IF_NULL(convert_func);
    return convert_func(obj);
  }
  return nullptr;
}

inline void PrintTypeError(const ops::OpDefPtr &op_def, const py::list &op_inputs) {
  std::vector<std::string> op_type_list;
  for (size_t index = 0; index < op_inputs.size(); ++index) {
    (void)op_type_list.emplace_back(BuilidPyInputTypeString(op_inputs[index]));
  }
  MS_EXCEPTION(TypeError) << ops::BuildOpErrorMsg(op_def, op_type_list);
}

void ParseOpInputByOpDef(const ops::OpDefPtr &op_def, const py::list &op_inputs, bool stub,
                         const FrontendOpRunInfoPtr &op_run_info) {
  size_t input_size = op_inputs.size();
  if (input_size != op_def->args_.size()) {
    MS_LOG(EXCEPTION) << "For Operator[" << op_def->name_ << "], the inputs number should be " << op_def->args_.size()
                      << " but got " << op_inputs.size() << ".";
  }
  (void)op_run_info->op_grad_info->input_value.resize(input_size);
  parse::OpDefConvertFunc convert_func;
  for (size_t i = 0; i < op_def->args_.size(); i++) {
    auto const &op_arg = op_def->args_[i];
    op_run_info->none_init_inputs_num += static_cast<size_t>(!op_arg.as_init_arg_);

    // Optional argument is valid for None as input.
    if (op_arg.is_optional_ && py::isinstance<py::none>(op_inputs[i])) {
      op_run_info->op_grad_info->input_value[i] = kNone;
      continue;
    }

    ValuePtr value = nullptr;
    convert_func = parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
    MS_EXCEPTION_IF_NULL(convert_func);
    value = convert_func(op_inputs[i]);
    if (value != nullptr) {
      op_run_info->op_grad_info->input_value[i] = value;
      continue;
    }

    // type cast has lower priority then signature cast
    if (!op_arg.cast_dtype_.empty()) {
      for (auto cast_dtype : op_arg.cast_dtype_) {
        convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
        MS_EXCEPTION_IF_NULL(convert_func);
        value = convert_func(op_inputs[i]);
        if (value != nullptr) {
          op_run_info->op_grad_info->input_value[i] = value;
          op_run_info->source_type[i] = cast_dtype;
          break;
        }
      }
    }

    if (value == nullptr) {
      PrintTypeError(op_def, op_inputs);
    }
  }
}

void PyParser::ParseOpInputByPythonObj(const FrontendOpRunInfoPtr &op_run_info, const py::list &op_inputs, bool stub) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_size = op_inputs.size();
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  op_run_info->source_type.resize(op_run_info->input_size);
  auto op_def = mindspore::ops::GetOpDef(op_run_info->base_op_run_info.op_name);
  if (op_def == nullptr) {
    op_run_info->op_grad_info->input_value.resize(op_run_info->input_size);
    op_run_info->none_init_inputs_num = op_run_info->input_size;
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      op_run_info->op_grad_info->input_value[i] = DataConvert::PyObjToValue(op_inputs[i], stub);
    }
  } else {
    op_run_info->none_init_inputs_num = 0;
    ParseOpInputByOpDef(op_def, op_inputs, stub, op_run_info);
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
  if (!op_run_info->is_jit_input) {
    const auto &unused_inputs = BpropExpander::GetUnusedInputs(op_run_info->op_grad_info->op_prim->name());
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      op_run_info->input_unused_in_bprop[i] = (unused_inputs.find(i) != unused_inputs.end());
    }
    // Set out used
    op_run_info->input_unused_in_bprop[op_run_info->input_size] =
      unused_inputs.find(op_run_info->input_size) != unused_inputs.end();
  }
}

py::object DataConvert::ValueToPyObj(const ValuePtr &v) { return ValueToPyData(v); }

ValuePtr DataConvert::PyObjToValue(const py::object &obj, bool stub) {
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
  if (v_vec.empty()) {
    MS_LOG(EXCEPTION) << "For bprop graph input value size should be greatet than 0, but get empty.";
  }
  size_t input_size = has_sens ? v_vec.size() - 1 : v_vec.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto &v = v_vec[i];
    MS_EXCEPTION_IF_NULL(v);
    MS_LOG(DEBUG) << "Get v is " << v->ToString();
    (void)flatten_v->emplace_back(v);
  }
  if (has_sens) {
    if (Common::IsTensor(v_vec[input_size])) {
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

FrontendOpRunInfoPtr PyBoost::Init(const PrimitivePtr &prim, const py::list &args) {
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  pynative_executor->StoreAsyncStatus(op_run_info);
  forward_executor->InitOpRunInfo(op_run_info);
  return op_run_info;
}

void PyBoost::MakeOutputValue(const FrontendOpRunInfoPtr &op_run_info, const std::vector<TensorPtr> &outputs) {
  size_t size = outputs.size();
  if (size == kSizeOne) {
    op_run_info->real_out = outputs[0];
    return;
  }
  std::vector<ValuePtr> output_values(size);
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = outputs[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    output_values[i] = output_tensor;
  }
  op_run_info->real_out = std::make_shared<ValueTuple>(output_values);
}

void PyBoost::UpdateOutputTensorGradInfo(const std::vector<TensorPtr> &outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs[i];
    output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>());
    output_tensor->auto_grad_meta_data()->set_grad_type(TensorGradType::kOpOutput);
  }
}

void PyBoost::UpdateStubOutput(const FrontendOpRunInfoPtr &op_run_info, const AbstractBasePtr &abstract) {
  if (op_run_info->stub_output == nullptr) {
    return;
  }

  auto success = op_run_info->stub_output->SetAbstract(abstract);
  if (!success) {
    const auto &op_name = op_run_info->base_op_run_info.op_name;
    MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                            << PredictOutType(op_run_info) << ", infer type is " << abstract->BuildType()
                            << ", the name of operator is [" << op_name
                            << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
  }
  MS_LOG(DEBUG) << "Update StubNode abstract " << abstract->ToString();

  op_run_info->stub_output->SetValue(op_run_info->real_out);
}

void PyBoost::UpdateOpRunInfo(const kernel::pyboost::OpPtr &op, const vector<ValuePtr> &op_inputs,
                              const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(op_run_info);
  // Set result to python
  MakeOutputValue(op_run_info, op->outputs());
  UpdateStubOutput(op_run_info, op->output_abs());

  // Update op run info for auto grad
  if (op_run_info->requires_grad) {
    op_run_info->base_op_run_info.abstract = op->output_abs();
    op_run_info->op_grad_info->input_value = op_inputs;
    op_run_info->op_grad_info->input_abs = op->input_abs();
    op_run_info->op_grad_info->out_value = op_run_info->real_out;
    op_run_info->op_grad_info->out_abs = op->output_abs();
    UpdateOutputTensorGradInfo(op->outputs());
    op->set_grad_func([op_run_info]() { DoGrad(op_run_info); });
  }
}

PrimitivePtr PyBoost::ConvertPrimitive(const py::object &obj) {
  const auto &adapter = obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(obj);
    adapter->set_attached_primitive(prim);
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
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  return pynative_executor->RunOpStub(wrap_args);
}

void PyBoost::DoGrad(const FrontendOpRunInfoPtr &op_run_info) {
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();

  PyParser::PrepareOpGradInfo(op_run_info);
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
    DataConvert::MarkInputs(op_run_info, input_object, index, forward->grad()->top_cell());
  }
  forward->ForwardOpGradImpl(op_run_info);
}

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
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
      if (Common::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kParameter;
      }
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(tensor);
    (void)op_run_info->base_op_run_info.input_masks.emplace_back(tensor_mask);
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
                                   const TopCellInfoPtr &top_cell, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(map_tensor);
  constexpr int input_num = 1;
  const auto input_names = op_run_info->op_grad_info->op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(map_tensor);
  const auto it = op_run_info->base_op_run_info.input_masks.end();
  (void)op_run_info->base_op_run_info.input_masks.insert(it, input_num, kParameterWeightTensorMask);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = Common::SetTensorGradInfo(map_tensor, top_cell);
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

  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndptr());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetIndices());
  (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(csr_tensor->GetValues());
  const auto it = op_run_info->base_op_run_info.input_masks.end();
  (void)op_run_info->base_op_run_info.input_masks.insert(it, input_num, kParameterDataTensorMask);
  op_run_info->op_grad_info->op_prim->set_attr("is_csr", MakeValue(true));
  op_run_info->op_grad_info->op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kOpOutput;
    for (int i = 0; i < input_num; ++i) {
      auto iter = op_run_info->base_op_run_info.expanded_input_values.rbegin() + i;
      auto grad_type = Common::SetTensorGradInfo((*iter)->cast<tensor::TensorPtr>(), top_cell);
      if (Common::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = TensorGradType::kParameter;
      }
    }
  }
}

void DataConvert::ConvertValueTensorId(const ValuePtr &value, std::vector<std::string> *converted_tensor_id) {
  if (value->isa<tensor::Tensor>()) {
    (void)converted_tensor_id->emplace_back(value->cast<tensor::TensorPtr>()->id());
  } else if (value->isa<ValueSequence>()) {
    const auto &seq = value->cast<ValueSequencePtr>();
    for (const auto &val : seq->value()) {
      ConvertValueTensorId(val, converted_tensor_id);
    }
  }
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_masks.emplace_back(kValueNodeMask);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::Tensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, index, top_cell);
  } else {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_masks.emplace_back(kValueNodeMask);
  }
}

void DataConvert::MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                             const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  tensor::TensorPtr tensor_ptr = nullptr;
  int64_t input_mask = kParameterDataTensorMask;
  if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), top_cell, index);
    return;
  } else if (v->isa<tensor::Tensor>()) {
    tensor_ptr = v->cast<tensor::TensorPtr>();
    if (tensor_ptr->is_parameter()) {
      input_mask = kParameterWeightTensorMask;
    }
    if (op_run_info->requires_grad) {
      op_run_info->op_grad_info->input_value_grad_type[index] = Common::SetTensorGradInfo(tensor_ptr, top_cell);
    }
  } else if (v->isa<BoolImm>() || v->isa<FloatImm>() || v->isa<Type>() || v->isa<StringImm>() || v->isa<None>()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_masks.emplace_back(kValueNodeMask);
    return;
  } else if (v->isa<IntegerImm>()) {
    if (op_run_info->base_op_run_info.op_name == prim::kPrimCSRReduceSum->name()) {
      int64_t input = v->cast<Int64ImmPtr>()->value();
      op_run_info->op_grad_info->op_prim->set_attr("axis", MakeValue(input));
      return;
    }
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(v);
    (void)op_run_info->base_op_run_info.input_masks.emplace_back(kValueNodeMask);
    return;
  } else if (v->isa<ValueSequence>()) {
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), index, top_cell);
    return;
  } else if (v->isa<tensor::CSRTensor>()) {
    ConvertCSRTensorToTensorList(op_run_info, v->cast<tensor::CSRTensorPtr>(), top_cell, index);
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
  (void)op_run_info->base_op_run_info.input_masks.emplace_back(input_mask);
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

  auto seq = inputs[1]->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq);
  // 2nd input tensor is {}, means reduce all axis.
  if (seq->size() == 0) {
    auto size = inputs[0]->cast<tensor::TensorPtr>()->shape().size();
    // For example, input 0 is Tensor(shape=[], value=1), the axis to reduce is 0.
    std::vector<ValuePtr> axis = {std::make_shared<Int64Imm>(0)};
    for (size_t i = 1; i < size; ++i) {
      axis.push_back(std::make_shared<Int64Imm>(static_cast<int64_t>(i)));
    }
    op_run_info->base_op_run_info.expanded_input_values[1] = std::make_shared<ValueTuple>(axis);
  }
}

void DataConvert::GetInputTensor(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);

  (void)op_run_info->base_op_run_info.expanded_input_values.reserve(op_run_info->input_size);
  (void)op_run_info->base_op_run_info.input_masks.reserve(op_run_info->input_size);
  // Get input tensors.
  op_run_info->op_grad_info->op_prim->BeginRecordAddAttr();
  for (size_t index = 0; index < op_run_info->input_size; ++index) {
    const ValuePtr &input_object = op_run_info->op_grad_info->input_value[index];
    // convert const input to attr
    if (RunOpConvertConstInputToAttr(op_run_info, input_object, index)) {
      continue;
    }
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    MarkInputs(op_run_info, input_object, index, top_cell);
    // -1 indicates input_object is not a dynInput
    if (!op_run_info->base_op_run_info.dyn_input_sizes.empty() && !input_object->isa<ValueSequence>()) {
      (void)op_run_info->base_op_run_info.dyn_input_sizes.emplace_back(-1);
    }
  }
  op_run_info->op_grad_info->op_prim->EndRecordAddAttr();
  ReplaceReduceAxis(op_run_info);
  AddDynInputsSizesAttr(op_run_info);
}

bool GradCommon::IsRealOp(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    return false;
  }
  return kNotRealOP.find(prim->name()) == kNotRealOP.end();
}

void GradCommon::SetForward(const AnfNodePtrList &node_list) {
  for (const auto &cn : node_list) {
    auto out = Common::CreatOutputTensorValueByAbstract(cn->abstract());
    const auto &c_node = cn->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(c_node);
    c_node->set_forward(Common::CreateValueNodeByValue(out, cn->abstract()), "");
  }
}

void GradCommon::GetUsedCNodeInBpropGraph(const CNodePtr &cnode, const mindspore::HashSet<size_t> &unused_inputs,
                                          AnfNodePtrList *node_list) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(node_list);
  // Check input used in single op bprop graph. For example,
  // A = a * b;
  // B = A * c;
  // So, A can also replace by its output
  size_t input_num = cnode->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    if (unused_inputs.find(i) == unused_inputs.end() && cnode->input(i + 1)->isa<CNode>()) {
      // Input used by bprop graph, and it is a cnode have produce real output
      const auto &input_c = cnode->input(i + 1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_c);
      if (IsPrimitive(input_c, prim::kPrimMakeTuple)) {
        size_t tuple_input_num = input_c->size() - 1;
        for (size_t j = 0; j < tuple_input_num; ++j) {
          if (auto f_node = common::AnfAlgo::VisitKernel(input_c, j).first; f_node->isa<CNode>() && IsRealOp(f_node)) {
            (void)node_list->emplace_back(f_node);
          }
        }
      } else {
        if (auto f_node = common::AnfAlgo::VisitKernel(input_c, 0).first; f_node->isa<CNode>() && IsRealOp(f_node)) {
          (void)node_list->emplace_back(f_node);
        }
      }
    }
  }
  // Check output used in single op bprop graph
  if (unused_inputs.find(cnode->size() - 1) == unused_inputs.end()) {
    (void)node_list->emplace_back(cnode);
  }
}
}  // namespace PyNativeAlgo

void DispatchOp(const std::shared_ptr<FrontendTask> &task) {
  auto forward_executor = PyNativeExecutor::GetInstance()->forward_executor();
  // TODO(caifubi): enable cpu/gpu async.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto sync = context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  auto mode = context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (sync || device == kGPUDevice || device == kCPUDevice || mode == mindspore::kGraphMode) {
    runtime::OpExecutor::GetInstance().WaitAll();
    task->Run();
  } else {
    forward_executor->frontend_queue()->Push(task);
  }
}
}  // namespace pynative
}  // namespace mindspore
