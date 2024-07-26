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
#include "pybind_api/gil_scoped_long_running.h"
#include "pybind_api/ir/hook_py.h"
#include "utils/ms_context.h"
#include "ir/cell.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/primfunc_utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "include/common/utils/stub_tensor.h"
#include "frontend/expander/bprop/bprop.h"
#include "frontend/optimizer/environ_conversion.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "pipeline/pynative/grad/jit/jit_grad.h"
#include "ops/sequence_op_name.h"
#include "ops/structure_ops.h"
#include "ops/other_ops.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "kernel/pyboost/auto_generate/contiguous.h"
#include "runtime/pipeline/pipeline.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "include/common/pynative/abstract_converter.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"

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
static AbstractConverter kGradAbstractConverter;

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
  auto type_id = Common::GetTypeFromAbstract(abs);
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
  }
  if (type_id == kObjectTypeString) {
    return MakeValue("");
  }
  if (type_id >= kNumberTypeInt && type_id <= kNumberTypeUInt64) {
    return MakeValue(static_cast<int64_t>(0));
  }
  if (type_id >= kNumberTypeFloat && type_id <= kNumberTypeFloat64) {
    return MakeValue(static_cast<float>(0));
  }
  if (type_id == kNumberTypeDouble) {
    return MakeValue(static_cast<double>(0));
  }
  MS_LOG(EXCEPTION) << "Get unsupported type " << type_id;
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

tensor::BaseTensorPtr GetContiguousTensor(const tensor::BaseTensorPtr &input_tensor, const std::string &device_target,
                                          bool requires_grad) {
  auto contiguous_op = CREATE_PYBOOST_OP(Contiguous, device_target);
  auto contiguous_tensor = contiguous_op->Call(input_tensor);
  if (requires_grad) {
    contiguous_op->CreateOutputSimpleInfoForView();
    const auto &contiguous_run_info = std::make_shared<FrontendOpRunInfo>();
    contiguous_run_info->requires_grad = true;
    PyBoost::UpdateOpRunInfo(contiguous_op, contiguous_run_info);
    contiguous_run_info->base_op_run_info.device_target = device_target;
    contiguous_run_info->input_size = 1;
    contiguous_run_info->base_op_run_info.op_name = ops::kNameContiguous;
    contiguous_run_info->op_grad_info->op_prim = prim::kPrimContiguous;
    PyBoost::DoGrad(contiguous_op, contiguous_run_info, {input_tensor});
  }
  return contiguous_tensor;
}

void UnsetValueAbstractCache(const ValuePtr &value) {
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    tensor->set_abstract(std::weak_ptr<abstract::AbstractBase>());
  } else if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    tensor->set_abstract(std::weak_ptr<abstract::AbstractBase>());
  } else if (value->isa<ValueSequence>()) {
    const auto &seq = value->cast<ValueSequencePtr>();
    auto elements = seq->value();
    for (const auto &element : elements) {
      UnsetValueAbstractCache(element);
    }
  }
}

bool NeedToContiguous(const TensorStorageInfoPtr &storage_info, const device::DeviceAddressPtr &device_address) {
  MS_EXCEPTION_IF_NULL(storage_info);
  MS_EXCEPTION_IF_NULL(device_address);
  if (!storage_info->is_contiguous || storage_info->storage_offset > 0) {
    // Tensor is not contiguous, or offset is not zero. Need to contiguous or copy.
    return true;
  }
  if (!trans::FormatHelper::GetInstance().IsBaseFormatType(device_address->GetFormatEnum())) {
    // Special format need to contiguous
    return true;
  }
  return false;
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
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>()->id();
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
  if (value->isa<tensor::BaseTensor>()) {
    return value->cast<tensor::BaseTensorPtr>()->base_shape_ptr() != nullptr;
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
    if (v->isa<tensor::MetaSparseTensor>() || v->isa<tensor::BaseTensor>()) {
      return true;
    } else if (v->isa<ValueSequence>()) {
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
    } else {
      MS_LOG(DEBUG) << "Get value " << v->ToString();
      return false;
    }
  }
  MS_LOG(DEBUG) << "Get value " << v->ToString();
  return v->isa<tensor::BaseTensor>() || v->isa<tensor::MetaSparseTensor>();
}

bool Common::IsControlFlowGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  return !func_graph->func_graphs_used_total().empty();
}

ValuePtr Common::FilterSensValues(const ValuePtr &value, bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>() || value->isa<tensor::COOTensor>() || value->isa<tensor::CSRTensor>()) {
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

tensor::BaseTensorPtr Common::GetTensorFromParam(const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(param_node);
  auto param = param_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  if (!param->has_default()) {
    return nullptr;
  }
  auto default_value = param->default_param();
  MS_EXCEPTION_IF_NULL(default_value);
  auto tensor_value = default_value->cast<tensor::BaseTensorPtr>();
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

std::pair<TypePtr, TypeId> Common::GetTypeFromValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    return std::make_pair(v->cast<tensor::BaseTensorPtr>()->Dtype(), kObjectTypeTensorType);
  } else if (v->isa<ValueTuple>()) {
    return std::make_pair(v->type(), kObjectTypeTuple);
  } else if (v->isa<ValueList>()) {
    return std::make_pair(v->type(), kObjectTypeList);
  } else if (v->isa<None>()) {
    return std::make_pair(kTypeNone, kMetaTypeNone);
  } else {
    return std::make_pair(v->type(), v->type()->object_type());
  }
}

ShapeVector Common::GetShapeFromValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>()->shape_c();
  } else if (v->isa<ValueSequence>()) {
    const auto &v_seq = v->cast<ValueSequencePtr>()->value();
    ShapeVector plant_shape_vector;
    for (const auto &item : v_seq) {
      const auto &shape = GetShapeFromValue(item);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(plant_shape_vector),
                           [](int64_t s) { return s; });
    }
    return plant_shape_vector;
  } else {
    return ShapeVector{};
  }
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
  }
  if (utils::isa<ValueSequence>(v)) {
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
  auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
  kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->base_op_run_info.stream_id);
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    op_run_info->op_grad_info->input_value[i] = StubNodeToValueInner(op_run_info->op_grad_info->input_value[i]);
    if (!op_run_info->is_view_op) {
      op_run_info->op_grad_info->input_value[i] =
        ConvertToContiguousValue(op_run_info->op_grad_info->input_value[i], op_run_info->requires_grad);
    }
    kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);
    runtime::DeviceAddressUtils::CreateKernelTensor(op_run_info->op_grad_info->input_value[i]);
  }
}

tensor::BaseTensorPtr Common::StubNodeToTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (utils::isa<stub::StubNode>(v)) {
    auto stub = utils::cast<stub::StubNodePtr>(v);
    return stub->WaitValue()->cast<tensor::BaseTensorPtr>();
  }
  if (v->isa<tensor::BaseTensor>()) {
    return v->cast<tensor::BaseTensorPtr>();
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor, but got " << v->ToString();
}

ValuePtr Common::ConvertToContiguousValue(const ValuePtr &v, bool requires_grad) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);

    auto contiguous_tensor = ConvertToContiguousTensor(tensor, requires_grad);
    return contiguous_tensor;
  }
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    if (values.empty() || utils::isa<Scalar>(values[0])) {
      return v;
    }
    ValuePtrList value_list;
    (void)std::transform(
      values.begin(), values.end(), std::back_inserter(value_list),
      [requires_grad](const ValuePtr &value) { return ConvertToContiguousValue(value, requires_grad); });
    if (utils::isa<ValueTuple>(v)) {
      return std::make_shared<ValueTuple>(value_list);
    }
    if (utils::isa<ValueList>(v)) {
      return std::make_shared<ValueList>(value_list);
    }
    MS_LOG(EXCEPTION) << "Not support ValueSequence " << v->ToString();
  } else {
    return v;
  }
}

tensor::BaseTensorPtr Common::ConvertToContiguousTensor(const tensor::BaseTensorPtr &tensor, bool requires_grad) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->storage_info() == nullptr) {
    return tensor;
  }

  // Tensor with storage info, need covert to contiguous in no-view op.
  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  const auto &device_target = device_address->device_name();

  if (!NeedToContiguous(tensor->storage_info(), device_address)) {
    return tensor;
  }

  const auto &contiguous_tensor = GetContiguousTensor(tensor, device_target, requires_grad);
  MS_LOG(DEBUG) << "ConvertToContiguousValue, old tensor id:" << tensor->id()
                << ", new tensor id:" << contiguous_tensor->id();
  return contiguous_tensor;
}  // namespace PyNativeAlgo

tensor::BaseTensorPtr Common::ConvertStubNodeToTensor(const ValuePtr &v, bool need_contiguous, bool requires_grad) {
  const auto &tensor = StubNodeToTensor(v);
  MS_EXCEPTION_IF_NULL(tensor);
  if (!need_contiguous || tensor->storage_info() == nullptr) {
    return tensor;
  }

  auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  const auto &device_target = device_address->device_name();
  if (device_target == kAscendDevice) {
    return tensor;
  }

  if (!NeedToContiguous(tensor->storage_info(), device_address)) {
    return tensor;
  }
  return GetContiguousTensor(tensor, device_target, requires_grad);
}

std::optional<tensor::BaseTensorPtr> Common::ConvertStubNodeToTensor(const std::optional<ValuePtr> &v,
                                                                     bool need_contiguous, bool requires_grad) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToTensor(v.value(), need_contiguous, requires_grad));
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueListPtr &v, bool need_contiguous, bool requires_grad) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

ValueTuplePtr Common::ConvertStubNodeToValueTuple(const ValueTuplePtr &v, bool need_contiguous, bool requires_grad) {
  if (utils::isa<ValueSequence>(v)) {
    const auto &value_seq = utils::cast<ValueSequencePtr>(v);
    const auto &values = value_seq->value();
    std::vector<ValuePtr> tensor_list;
    (void)std::transform(values.begin(), values.end(), std::back_inserter(tensor_list),
                         [need_contiguous, requires_grad](const ValuePtr &value) {
                           return ConvertStubNodeToTensor(value, need_contiguous, requires_grad);
                         });
    return std::make_shared<ValueTuple>(tensor_list);
  }
  MS_LOG(EXCEPTION) << "It should be stub tensor sequence, but got " << v->ToString();
}

std::optional<ValueTuplePtr> Common::ConvertStubNodeToValueTuple(const std::optional<ValueTuplePtr> &v,
                                                                 bool need_contiguous, bool requires_grad) {
  if (!v.has_value()) {
    return std::nullopt;
  }
  return std::make_optional(ConvertStubNodeToValueTuple(v.value(), need_contiguous, requires_grad));
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
  }
  MS_EXCEPTION_IF_NULL(input_to_attr_index);
  for (auto &iter : reg_info->input_attr_map()) {
    (void)input_to_attr_index->insert(iter.first);
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
  return t;
}

void Common::ClearDeviceAddress(const ValuePtr &value) {
  std::vector<tensor::BaseTensorPtr> tensors;
  TensorValueToTensor(value, &tensors);
  for (const auto &tensor : tensors) {
    tensor->set_device_address(nullptr);
  }
}

void Common::SetOutputUsedInBpropGraph(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &v_t = value->cast<tensor::BaseTensorPtr>();
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

ValuePtr Common::CreateFakeValueWithoutDeviceAddress(const ValuePtr &value, bool is_force_create_fake) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &v_t = value->cast<tensor::BaseTensorPtr>();
    // If the tensor used in bprop graph, no need create fake value
    if (!is_force_create_fake && (v_t->is_parameter() || v_t->used_in_bprop_graph())) {
      return value;
    }
    auto t = std::make_shared<tensor::BaseTensor>(*v_t);
    if (v_t->is_parameter()) {
      t->set_param_info(v_t->param_info());
    }
    t->set_device_address(nullptr);
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

InputType Common::SetValueGradInfo(const ValuePtr &value, const TopCellInfoPtr &top_cell, InputType grad_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor_value = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor_value->auto_grad_meta_data();
    if (auto_grad_meta_data != nullptr) {
      if (auto_grad_meta_data->input_type() != InputType::kUnkown) {
        return auto_grad_meta_data->input_type();
      }
      MS_LOG(DEBUG) << "Set input type for tensor " << tensor_value->id();
    } else if (grad_type != InputType::kConstant || tensor_value->is_parameter()) {
      MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor_value->id();
      auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      tensor::RegisterHook::UpdateTensorBackwardHook(auto_grad_meta_data, tensor_value->id());
      tensor_value->set_auto_grad_meta_data(auto_grad_meta_data);
    }
    // Scalar tensor auto grad meta data is nullptr
    if (auto_grad_meta_data != nullptr) {
      if (tensor_value->is_parameter() && grad_type != InputType::kInput) {
        grad_type = InputType::kParameter;
      }
      auto_grad_meta_data->set_input_type(grad_type);
    }
    return grad_type;
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    InputType ret_type = grad_type;
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
      (void)SetValueGradInfo(v.second, top_cell, grad_type);
    }
  }
  return grad_type;
}

InputType Common::SetTensorGradInfo(const tensor::BaseTensorPtr &tensor, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr) {
    if (auto_grad_meta_data->input_type() != InputType::kUnkown) {
      return auto_grad_meta_data->input_type();
    }
    MS_LOG(DEBUG) << "Set input type for tensor " << tensor->id();
  } else if (tensor->is_parameter()) {
    MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor->id();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor::RegisterHook::UpdateTensorBackwardHook(auto_grad_meta_data, tensor->id());
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  // Set weight tensor grad type
  if (tensor->is_parameter()) {
    auto_grad_meta_data->set_input_type(InputType::kParameter);
    return InputType::kParameter;
  }
  return InputType::kConstant;
}

void Common::SetGraphInputAndWeightsInfo(const FrontendOpRunInfoPtr &op_run_info, const FuncGraphPtr &func_graph,
                                         const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &original_params = func_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  bool need_add_input_abs = op_run_info->op_grad_info->input_abs.empty();
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      op_run_info->op_grad_info->input_value_grad_type[i] =
        SetValueGradInfo(op_run_info->op_grad_info->input_value[i], top_cell, InputType::kConstant);
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
  const auto &abs_seq = target_abstract->cast<abstract::AbstractSequencePtr>();
  if (abs_seq->dynamic_len() && abs_seq->dynamic_len_element_abs() != nullptr) {
    return;
  }
  MS_LOG(DEBUG) << "Process tuple param " << target_abstract->ToString();
  auto it = std::find(bprop_params.begin(), bprop_params.end(), target_param);
  it = bprop_params.erase(it);
  AnfNodePtrList make_tuple{NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList new_param;
  PlantTupleParam(bprop_graph, abs_seq, &make_tuple, &new_param);
  (void)bprop_params.insert(it, new_param.begin(), new_param.end());
  bprop_graph->set_parameters(bprop_params);
  auto make_tuple_param = bprop_graph->NewCNode(make_tuple);
  make_tuple_param->set_abstract(target_abstract);
  auto manager = bprop_graph->manager();
  if (manager == nullptr) {
    manager = MakeManager({bprop_graph}, false);
  }
  MS_EXCEPTION_IF_NULL(manager);
  auto tr = manager->Transact();
  (void)tr.Replace(target_param, make_tuple_param);
  tr.Commit();
}

void Common::ProcessDictParam(const FuncGraphPtr &bprop_graph, size_t position) {
  auto bprop_params = bprop_graph->parameters();
  auto target_param = bprop_params[position];
  MS_EXCEPTION_IF_NULL(target_param);
  const auto &target_abstract = target_param->abstract();
  MS_EXCEPTION_IF_NULL(target_abstract);
  if (!target_abstract->isa<abstract::AbstractDictionary>()) {
    MS_LOG(EXCEPTION) << "Get wrong param " << target_abstract->ToString();
  }
  MS_LOG(DEBUG) << "Process Dict param " << target_abstract->ToString();
  auto it = std::find(bprop_params.begin(), bprop_params.end(), target_param);
  it = bprop_params.erase(it);
  const auto &abs_dict = target_abstract->cast<abstract::AbstractDictionaryPtr>();
  abstract::AbstractBasePtrList local_key_abs_inputs;
  abstract::AbstractBasePtrList local_value_abs_inputs;
  for (size_t i = 0; i < abs_dict->size(); ++i) {
    (void)local_key_abs_inputs.emplace_back(abs_dict->elements()[i].first);
    (void)local_value_abs_inputs.emplace_back(abs_dict->elements()[i].second);
  }
  auto key_param = bprop_graph->add_parameter();
  key_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
  auto value_param = bprop_graph->add_parameter();
  value_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
  auto key_it = bprop_params.insert(it, value_param);
  (void)bprop_params.insert(key_it, key_param);
  bprop_graph->set_parameters(bprop_params);
  auto dict_node = bprop_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), key_param, value_param});
  dict_node->set_abstract(abs_dict);
  auto manager = bprop_graph->manager();
  if (manager == nullptr) {
    manager = MakeManager({bprop_graph}, false);
  }
  auto tr = manager->Transact();
  (void)tr.Replace(target_param, dict_node);
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

size_t Common::GetValueSize(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::BaseTensor>() || v->isa<Scalar>()) {
    return 1;
  } else if (v->isa<ValueSequence>()) {
    auto seq = v->cast<ValueSequencePtr>();
    size_t output_size = 0;
    for (const auto &val : seq->value()) {
      output_size += GetValueSize(val);
    }
    return output_size;
  } else if (v->isa<ValueDictionary>()) {
    const auto &v_dict = v->cast<ValueDictionaryPtr>();
    size_t output_size = 0;
    for (const auto &val : v_dict->value()) {
      output_size += GetValueSize(val.second);
    }
    return output_size;
  }
  return 0;
}

ValuePtr Common::CreateTensorByConstantValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->type();
  if (Common::IsTensor(value, true) || value->isa<Number>() || value->isa<None>() ||
      (type != nullptr && type->isa<String>())) {
    return value;
  }
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else if (value->isa<ValueList>()) {
    tensor_ptr = opt::CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value()));
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple, but get type " << value->type_name()
                      << ", value " << value->ToString();
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  return tensor_ptr;
}

void AutoGrad::CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);

  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    tensor->set_abstract(abs);
    kGradAbstractConverter.CacheAbstract(abs);
  } else if (v->isa<ValueSequence>()) {
    const auto &value_seq = v->cast<ValueSequencePtr>();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq == nullptr) {
      MS_LOG(EXCEPTION) << "Abstract is not abstract sequence, get " << abs->ToString();
    }
    size_t value_size = value_seq->size();
    if (value_size != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Abstract size " << abs_seq->size() << " is not equal to value size " << value_size;
    }
    for (size_t i = 0; i < value_size; ++i) {
      CacheOutputAbstract(value_seq->value()[i], abs_seq->elements()[i]);
    }
  }
}

namespace {
void ConvertSimpleInferInfoToAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  // Get inputs abstract
  for (const auto &v : op_grad_info->input_value) {
    op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
  }

  // Get output abstract
  MS_EXCEPTION_IF_NULL(op_grad_info->output_value_simple_info);
  op_grad_info->out_abs = TransformValueSimpleInfoToAbstract(*op_grad_info->output_value_simple_info);

  // Set abstract to tensor
  AutoGrad::CacheOutputAbstract(op_grad_info->out_value, op_grad_info->out_abs);
  MS_LOG(DEBUG) << "Get output abstract " << op_grad_info->out_abs->ToString();
}
}  // namespace

void AutoGrad::CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  if (op_grad_info->output_value_simple_info != nullptr) {
    MS_LOG(DEBUG) << "Convert op " << op_grad_info->op_prim->name() << " simple infer info to abstract";
    ConvertSimpleInferInfoToAbstract(op_grad_info);
    return;
  }

  // View op input abs and output abs maybe nullptr
  if (MS_UNLIKELY(op_grad_info->input_abs.empty())) {
    // Get inputs abstract
    MS_LOG(DEBUG) << "Op " << op_grad_info->op_prim->name() << " inputs abstract not set, set it now";
    for (const auto &v : op_grad_info->input_value) {
      // For use abstract cache on tensor
      op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
    }
  }
  if (op_grad_info->out_abs == nullptr) {
    MS_LOG(EXCEPTION) << "Get output abs is nullptr";
  }
}

std::string PyParser::GetIdByPyObj(const py::object &obj) {
  if (py::isinstance<tensor::BaseTensor>(obj)) {
    return obj.cast<tensor::BaseTensorPtr>()->id();
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
    if (py::isinstance<py::list>(args[i])) {
      (void)input_arg_value_vec.emplace_back(DataConvert::PyObjToValue(py::cast<py::tuple>(args[i])));
    } else {
      (void)input_arg_value_vec.emplace_back(DataConvert::PyObjToValue(args[i]));
    }
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

std::string PyParser::BuilidPyInputTypeString(const py::object &obj) {
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

  if (py::isinstance<mindspore::tensor::BaseTensor>(obj)) {
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

void PyParser::PrintTypeCastError(const ops::OpDefPtr &op_def, const py::list &op_inputs, size_t idx) {
  auto const &op_arg = op_def->args_[idx];
  bool is_suppport_tensor_cast = std::any_of(op_arg.cast_dtype_.begin(), op_arg.cast_dtype_.end(),
                                             [](const auto &type) { return type == ops::DT_TENSOR; });
  if (is_suppport_tensor_cast) {
    auto tensor = parse::ConvertTensorValue(op_inputs[idx]);
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
                              << ops::EnumToString(op_arg.arg_dtype_) << ".";
    }
  }
  std::vector<std::string> op_type_list;
  for (size_t index = 0; index < op_inputs.size(); ++index) {
    (void)op_type_list.emplace_back(PyParser::BuilidPyInputTypeString(op_inputs[index]));
  }
  MS_EXCEPTION(TypeError) << ops::BuildOpErrorMsg(op_def, op_type_list);
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
    parse::OpDefConvertFunc convert_func = parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
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
      op_run_info->op_grad_info->input_value[i] = DataConvert::PyObjToValue(op_inputs[i], stub);
    }
  } else {
    op_run_info->none_init_inputs_num = 0;
    ParseOpInputByOpDef(op_def, op_inputs, stub, op_run_info);
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
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << ConvertPyObjToString(obj);
  }
  return converted_ret;
}

ValuePtr DataConvert::BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence, size_t op_index) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::BaseTensorPtr>(value)) {
    auto t = utils::cast<tensor::BaseTensorPtr>(value);
    if (requires_grad) {
      t->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(op_index, InputType::kOpOutput));
    }
    ret = t;
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref, requires_grad, is_out_sequence, op_index);
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

ValuePtr DataConvert::VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence,
                                       size_t op_index) {
  MS_EXCEPTION_IF_NULL(vec_ref);

  size_t value_size = vec_ref.size();
  if (value_size == 1 && !is_out_sequence) {
    return BaseRefToValue(vec_ref[0], requires_grad, is_out_sequence, op_index);
  }
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i], requires_grad, is_out_sequence, op_index);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void DataConvert::FlattenValueSeqArg(const ValuePtr &v, bool is_only_flatten_tensor_seq, bool is_filter_tensor,
                                     std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(flatten_v);
  MS_LOG(DEBUG) << "Get is only flatten tensor seq " << is_only_flatten_tensor_seq;
  if (v->isa<tensor::BaseTensor>()) {
    (void)flatten_v->emplace_back(v);
  } else if (v->isa<ValueSequence>()) {
    const auto &v_vec = v->cast<ValueSequencePtr>()->value();
    if (v_vec.empty() && !is_filter_tensor) {
      MS_LOG(DEBUG) << "Get empty tuple value";
      (void)flatten_v->emplace_back(v);
      MS_LOG(DEBUG) << "Get empty value sequence";
      return;
    }
    if (is_only_flatten_tensor_seq && !v_vec.front()->isa<tensor::BaseTensor>()) {
      (void)flatten_v->emplace_back(v);
    } else {
      for (const auto &elem : v_vec) {
        FlattenValueSeqArg(elem, is_only_flatten_tensor_seq, is_filter_tensor, flatten_v);
      }
    }
  } else if (is_only_flatten_tensor_seq) {
    if (v->isa<ValueDictionary>()) {
      auto dic_v = v->cast<ValueDictionaryPtr>();
      for (const auto &elem : dic_v->value()) {
        FlattenValueSeqArg(elem.second, is_only_flatten_tensor_seq, is_filter_tensor, flatten_v);
      }
    } else {
      (void)flatten_v->emplace_back(v);
    }
  } else if (!is_filter_tensor) {
    MS_LOG(DEBUG) << "Get not tensor value: " << v->ToString();
    (void)flatten_v->emplace_back(v);
  }
}

ValuePtrList DataConvert::FlattenTensorSeqInValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList outputs;
  FlattenValueSeqArg(v, true, false, &outputs);
  return outputs;
}

ValuePtrList DataConvert::FlattenTensorSeqInValueSeq(const ValuePtrList &v, bool only_flatten_tensor) {
  ValuePtrList outputs;
  for (const auto &item : v) {
    FlattenValueSeqArg(item, only_flatten_tensor, false, &outputs);
  }
  return outputs;
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
      MS_LOG(DEBUG) << "Get value tuple size " << v_vec[input_size]->cast<ValueSequencePtr>()->size();
      FlattenValueSeqArg(v_vec[input_size], false, false, flatten_v);
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
  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
      return false;
    }
  }
  (void)op_run_info->op_grad_info->op_prim->AddAttr(input_name, v);
  return true;
}

void DataConvert::TransformValueNodeBaseTensorToTensor(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &v = value_node->value();
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<tensor::BaseTensor>()) {
    return;
  }
  const auto &tensor = v->cast<tensor::BaseTensorPtr>();
  value_node->set_value(std::make_shared<tensor::Tensor>(*tensor));
}

FrontendOpRunInfoPtr PyBoost::Init(const PrimitivePtr &prim, const py::list &args) {
  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  prim->EnableSharedMutex();
  op_run_info->op_grad_info->op_prim = prim;
  op_run_info->base_op_run_info.op_name = prim->name();
  pynative_executor->StoreAsyncStatus(op_run_info);
  forward_executor->InitOpRunInfo(op_run_info);
  return op_run_info;
}

void PyBoost::MakeOutputValue(const FrontendOpRunInfoPtr &op_run_info, const kernel::pyboost::OpPtr &op) {
  size_t size = op->outputs().size();
  // If op are Contiguous, Cast(precision, implicit cast), which are internal ops and not have stub output
  bool is_tuple_output = op_run_info->stub_output != nullptr ? op_run_info->stub_output->isa<stub::SequenceNode>()
                                                             : PredictOutTypeByName(op->primitive()->name()) == kTuple;
  if (op->output_value_simple_info() != nullptr) {
    op_run_info->op_grad_info->output_value_simple_info = op->output_value_simple_info();
    op_run_info->op_grad_info->output_value_simple_info->is_tuple_output_ = is_tuple_output;
  }
  if (!is_tuple_output) {
    MS_EXCEPTION_IF_CHECK_FAIL(size == kSizeOne, "The size is more than one!");
    if (op->output_abs() != nullptr || op->output_value_simple_info() != nullptr) {
      // Set auto grad meta data for op output
      if (op_run_info->requires_grad) {
        op->outputs()[0]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(
          Common::GetPyNativeExecutor()->grad_executor()->top_cell()->op_index(), InputType::kOpOutput));
      }
      op_run_info->real_out = op->outputs()[0];
      return;
    }
  }
  std::vector<ValuePtr> output_values(size);
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = op->outputs()[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    // Set auto grad meta data for op outputs
    if (op_run_info->requires_grad) {
      output_tensor->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(
        Common::GetPyNativeExecutor()->grad_executor()->top_cell()->op_index(), InputType::kOpOutput));
    }
    output_values[i] = output_tensor;
  }
  op_run_info->real_out = std::make_shared<ValueTuple>(output_values);
}

void PyBoost::UpdateStubOutput(const FrontendOpRunInfoPtr &op_run_info, const AbstractBasePtr &abstract,
                               const kernel::pyboost::OpPtr &op) {
  MS_EXCEPTION_IF_NULL(op);
  if (op_run_info->stub_output == nullptr) {
    return;
  }
  if (MS_UNLIKELY(op->output_value_simple_info() != nullptr)) {
    op_run_info->stub_output->SetValueSimpleInfo(op->output_value_simple_info());
  } else {
    MS_EXCEPTION_IF_NULL(abstract);
    auto success = op_run_info->stub_output->SetAbstract(abstract);
    if (!success) {
      const auto &op_name = op_run_info->base_op_run_info.op_name;
      MS_EXCEPTION(TypeError) << "The predict type and infer type is not match, predict type is "
                              << PredictOutType(op_run_info) << ", infer type is " << abstract->BuildType()
                              << ", the name of operator is [" << op_name
                              << "]. Please modify or add predict type of operator in predict_out_type_map.h.";
    }
    MS_LOG(DEBUG) << "Update StubNode abstract " << abstract->ToString();
  }
  op_run_info->stub_output->SetValue(op_run_info->real_out);
}

void PyBoost::UpdateOpRunInfo(const kernel::pyboost::OpPtr &op, const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(op_run_info);
  // Create output value
  MakeOutputValue(op_run_info, op);

  // Set output value to python
  UpdateStubOutput(op_run_info, op->output_abs(), op);
}

void PyBoost::DataSyncForGraph(const kernel::pyboost::OpPtr &op, ValuePtrList &&op_inputs) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      !runtime::OpExecutor::GetInstance().async_for_graph()) {
    // If execution mode is Graph Mode in MsContext, the tensor will be the input of graph which will execute in Graph
    // Mode, if the graph contain no CNode after optimization, the tensor need sync to host.
    for (const auto &output : op->outputs()) {
      auto device_address = std::static_pointer_cast<device::DeviceAddress>(output->device_address());
      runtime::DeviceAddressUtils::CreateKernelTensor(device_address, output);
      output->data_sync(true);
      output->set_abstract(std::weak_ptr<abstract::AbstractBase>());
    }
    for (const auto &input : op_inputs) {
      if (input->isa<tensor::BaseTensor>()) {
        auto tensor = input->cast<tensor::BaseTensorPtr>();
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor->device_address());
        runtime::DeviceAddressUtils::CreateKernelTensor(device_address, tensor);
      }
      UnsetValueAbstractCache(input);
    }
  }
}

PrimitivePtr PyBoost::ConvertPrimitive(const py::object &obj) {
  const auto &adapter = obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(adapter);

  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
#ifndef ENABLE_TEST
    return std::make_shared<Primitive>(adapter->name(), adapter->attrs());
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

void PyBoost::SetAnyValueForAbstract(const kernel::pyboost::OpPtr &op) {
  const auto &input_abs = op->input_abs();
  for (const auto &abs : input_abs) {
    Common::SetAbstractValueToAnyValue(abs);
  }
  Common::SetAbstractValueToAnyValue(op->output_abs());
}

void PyBoost::DoGrad(const kernel::pyboost::OpPtr &op, const FrontendOpRunInfoPtr &op_run_info,
                     ValuePtrList &&op_inputs) {
  static const std::string kDoGradName = "DoGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);
  MS_EXCEPTION_IF_NULL(op);
  // Update op grad info
  op_run_info->op_grad_info->input_value = std::move(op_inputs);
  op_run_info->op_grad_info->out_value = op_run_info->real_out;

  const auto &pynative_executor = Common::GetPyNativeExecutor();
  const auto &forward = pynative_executor->forward_executor();
  op_run_info->op_grad_info->output_size = op->outputs().size();
  if (op->output_value_simple_info() == nullptr) {
    op_run_info->op_grad_info->input_abs = op->input_abs();
    op_run_info->base_op_run_info.abstract = op->output_abs();
  }
  // Check and set input auto grad meta info and InputType
  if (MS_LIKELY(!forward->grad()->top_cell()->is_bprop_need_get_forward_graph())) {
    MarkPyBoostInputs(op_run_info->op_grad_info, forward->grad()->top_cell());
  }
  forward->ForwardOpGradImpl(op_run_info);
}

void PyBoost::MarkPyBoostInputs(const OpGradInfoPtr &op_grad_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  size_t input_size = op_grad_info->input_value.size();
  op_grad_info->input_value_grad_type.resize(input_size);
  for (size_t index = 0; index < input_size; ++index) {
    const auto &v = op_grad_info->input_value[index];
    if (v->isa<tensor::BaseTensor>()) {
      op_grad_info->input_value_grad_type[index] =
        Common::SetTensorGradInfo(v->cast<tensor::BaseTensorPtr>(), top_cell);
    } else if (v->isa<ValueSequence>()) {
      const auto &value_sequence = v->cast<ValueSequencePtr>();
      const auto &tuple_inputs = value_sequence->value();
      if (!tuple_inputs.empty() && tuple_inputs[0]->isa<tensor::BaseTensor>()) {
        op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
        for (const auto &elem : tuple_inputs) {
          auto grad_type = Common::SetTensorGradInfo(elem->cast<tensor::BaseTensorPtr>(), top_cell);
          if (Common::IsParam(grad_type)) {
            op_grad_info->input_value_grad_type[index] = InputType::kParameter;
          }
        }
      }
    } else if (v->isa<tensor::MapTensor>()) {
      op_grad_info->input_value_grad_type[index] = Common::SetTensorGradInfo(v->cast<tensor::MapTensorPtr>(), top_cell);
    } else if (v->isa<tensor::CSRTensor>()) {
      const auto &csr_tensor = v->cast<tensor::CSRTensorPtr>();
      auto fn = [&op_grad_info, index](const auto &csr_tensor_input) {
        auto grad_type = Common::SetTensorGradInfo(csr_tensor_input, nullptr);
        if (Common::IsParam(grad_type)) {
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

void DataConvert::PlantTensorTupleToVector(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                           size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
  }
  for (const auto &v : value_seq->value()) {
    if (!v->isa<tensor::BaseTensor>()) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    InputType input_type = InputType::kInput;
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      auto grad_type = Common::SetTensorGradInfo(tensor, top_cell);
      if (Common::IsParam(grad_type)) {
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
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kParameter);
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
  const auto it = op_run_info->base_op_run_info.input_types.end();
  (void)op_run_info->base_op_run_info.input_types.insert(it, input_num, InputType::kInput);
  op_run_info->op_grad_info->op_prim->set_attr("is_csr", MakeValue(true));
  op_run_info->op_grad_info->op_prim->set_attr("dense_shape", MakeValue(csr_tensor->shape()));
  if (op_run_info->requires_grad) {
    op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kOpOutput;
    for (int i = 0; i < input_num; ++i) {
      auto iter = op_run_info->base_op_run_info.expanded_input_values.rbegin() + i;
      auto grad_type = Common::SetTensorGradInfo((*iter)->cast<tensor::BaseTensorPtr>(), top_cell);
      if (Common::IsParam(grad_type)) {
        op_run_info->op_grad_info->input_value_grad_type[index] = InputType::kParameter;
      }
    }
  }
}

void DataConvert::GetTensorIdFromOutputValue(const ValuePtr &value, std::vector<std::string> *converted_tensor_id) {
  if (value->isa<tensor::BaseTensor>()) {
    (void)converted_tensor_id->emplace_back(value->cast<tensor::BaseTensorPtr>()->id());
    MS_LOG(DEBUG) << "Get top cell output tensor id " << converted_tensor_id->back();
  } else if (value->isa<ValueSequence>()) {
    const auto &seq = value->cast<ValueSequencePtr>();
    for (const auto &val : seq->value()) {
      GetTensorIdFromOutputValue(val, converted_tensor_id);
    }
  } else if (value->isa<ValueDictionary>()) {
    GetTensorIdFromOutputValue(ConvertValueDictToValueTuple(value), converted_tensor_id);
  }
}

void DataConvert::ConvertTupleValueToTensor(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &value_seq,
                                            size_t index, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &tuple_inputs = value_seq->value();
  if (tuple_inputs.empty()) {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
    return;
  }
  if (tuple_inputs[0]->isa<tensor::BaseTensor>()) {
    PlantTensorTupleToVector(op_run_info, value_seq, index, top_cell);
  } else {
    (void)op_run_info->base_op_run_info.expanded_input_values.emplace_back(value_seq);
    (void)op_run_info->base_op_run_info.input_types.emplace_back(InputType::kConstant);
  }
}

void DataConvert::MarkInputs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index,
                             const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  tensor::BaseTensorPtr tensor_ptr = nullptr;
  InputType input_type = InputType::kInput;
  if (v->isa<tensor::BaseTensor>()) {
    tensor_ptr = v->cast<tensor::BaseTensorPtr>();
    if (tensor_ptr->is_parameter()) {
      input_type = InputType::kParameter;
    }
    if (op_run_info->requires_grad) {
      op_run_info->op_grad_info->input_value_grad_type[index] = Common::SetTensorGradInfo(tensor_ptr, top_cell);
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
    ConvertTupleValueToTensor(op_run_info, v->cast<ValueSequencePtr>(), index, top_cell);
    return;
  } else if (v->isa<tensor::MapTensor>()) {
    ConvertMapTensor(op_run_info, v->cast<tensor::MapTensorPtr>(), top_cell, index);
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
    auto size = inputs[0]->cast<tensor::BaseTensorPtr>()->shape().size();
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

namespace {
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName};

mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_call_graph_compile_cache_;

AnfNodePtr CreateMakeTupleNode(const KernelGraphPtr &tape, const ValueSequencePtr &tuple,
                               const abstract::AbstractSequencePtr &abs_seq, const SpecialType &type) {
  AnfNodePtrList args{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < tuple->size(); ++i) {
    AnfNodePtr special_like_value = AutoGrad::BuildSpecialNode(tape, tuple->value()[i], abs_seq->elements()[i], type);
    (void)args.emplace_back(special_like_value);
  }
  auto special_like_value = tape->FuncGraph::NewCNode(args);
  special_like_value->set_abstract(abs_seq);
  return special_like_value;
}

AnfNodePtr CreateMakeDictNode(const KernelGraphPtr &tape, const ValueDictionaryPtr &v_dict,
                              const abstract::AbstractDictionaryPtr &abs_dict, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(v_dict);
  MS_EXCEPTION_IF_NULL(abs_dict);
  AnfNodePtrList key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList local_key_abs_inputs;
  abstract::AbstractBasePtrList local_value_abs_inputs;
  for (size_t i = 0; i < v_dict->size(); ++i) {
    (void)key_inputs.emplace_back(
      Common::CreateValueNodeByValue(v_dict->value()[i].first, abs_dict->elements()[i].first));
    (void)local_key_abs_inputs.emplace_back(abs_dict->elements()[i].first);
    AnfNodePtr special_like_value =
      AutoGrad::BuildSpecialNode(tape, v_dict->value()[i].second, abs_dict->elements()[i].second, type);
    (void)value_inputs.emplace_back(special_like_value);
    (void)local_value_abs_inputs.emplace_back(abs_dict->elements()[i].second);
  }
  auto local_key_node = tape->NewCNode(key_inputs);
  local_key_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
  auto local_value_node = tape->NewCNode(value_inputs);
  local_value_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
  auto dict_node = tape->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  dict_node->set_abstract(abs_dict);
  return dict_node;
}

ValueNodePtr GetSparseTensorShapeNode(const ShapeVector &shape) {
  auto value_shape = NewValueNode(shape);
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape.begin(), shape.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto abs_shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  value_shape->set_abstract(abs_shape);
  return value_shape;
}

ValuePtr WrapCOOTensor(const ValuePtr &coo_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(coo_out);
  auto coo_tensor = coo_out->cast<tensor::COOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indices_tensor = coo_tensor->GetIndices();
  auto shape_vector = coo_tensor->shape();
  return std::make_shared<tensor::COOTensor>(indices_tensor, value_tensor, shape_vector);
}

ValuePtr WrapCSRTensor(const ValuePtr &csr_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(csr_out);
  auto csr_tensor = csr_out->cast<tensor::CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indptr_tensor = csr_tensor->GetIndptr();
  auto indices_tensor = csr_tensor->GetIndices();
  auto shape_vector = csr_tensor->shape();
  return std::make_shared<tensor::CSRTensor>(indptr_tensor, indices_tensor, value_tensor, shape_vector);
}
}  // namespace

bool AutoGrad::IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
}

bool AutoGrad::NeedGrad(const std::vector<ValuePtr> &input_values) {
  for (const ValuePtr &input_arg : input_values) {
    MS_EXCEPTION_IF_NULL(input_arg);
    if (input_arg->isa<tensor::BaseTensor>()) {
      tensor::BaseTensorPtr input_tensor = nullptr;
      input_tensor = input_arg->cast<tensor::BaseTensorPtr>();
      auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
      if (auto_grad_meta_data != nullptr) {
        if (auto_grad_meta_data->input_type() == InputType::kParameter && Common::IsParamRequiresGrad(input_tensor)) {
          return true;
        }
        auto variable = auto_grad_meta_data->variable();
        if (variable != nullptr) {
          return true;
        }
      }
    } else if (input_arg->isa<ValueSequence>()) {
      auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
      if (NeedGrad(value_seq)) {
        return true;
      }
    } else if (input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
      return true;
    }
    MS_LOG(DEBUG) << "Get value " << input_arg->ToString();
  }
  return false;
}

bool AutoGrad::IsZerosLikeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimZerosLike)) {
    return true;
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    return std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                       [](const auto &node) { return IsZerosLikeNode(node) == true; });
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeDict)) {
    return IsZerosLikeNode(cnode->input(kIndex2));
  }
  return false;
}

ValuePtr AutoGrad::GetFakeZeroTensor() {
  static ValuePtr fake_v = std::make_shared<tensor::Tensor>(0);
  return fake_v;
}

ValuePtr AutoGrad::BuildSpecialValueGrad(const ValuePtr &value, const tensor::BaseTensorPtr &grad,
                                         autograd::FuncBuilder *func_builder, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (grad != nullptr) {
    return grad;
  }
  if (value->isa<tensor::BaseTensor>()) {
    return (type == SpecialType::kZerosLikeType ? func_builder->Zeros(value) : func_builder->Ones(value));
  }
  if (value->isa<ValueSequence>()) {
    ValuePtr zero_value = nullptr;
    auto v_seq = value->cast<ValueSequencePtr>();
    ValuePtrList v_list;
    for (const auto &item : v_seq->value()) {
      (void)v_list.emplace_back(BuildSpecialValueGrad(item, grad, func_builder, type));
    }
    return std::make_shared<ValueTuple>(v_list);
  }
  if (value->isa<Scalar>()) {
    auto fake_tensor = std::make_shared<tensor::Tensor>(0, value->type());
    return BuildSpecialValueGrad(fake_tensor, grad, func_builder, type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return WrapCSRTensor(csr_tensor, BuildSpecialValueGrad(csr_tensor->GetValues(), grad, func_builder, type));
  }
  if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return WrapCOOTensor(coo_tensor, BuildSpecialValueGrad(coo_tensor->GetValues(), grad, func_builder, type));
  }
  MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
  auto fake_tensor = std::make_shared<tensor::Tensor>(0, value->type());
  return BuildSpecialValueGrad(fake_tensor, grad, func_builder, type);
}

AnfNodePtr AutoGrad::BuildSpecialNode(const KernelGraphPtr &tape, const ValuePtr &value,
                                      const abstract::AbstractBasePtr &abs, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto prim_node =
      (type == SpecialType::kZerosLikeType ? NewValueNode(std::make_shared<Primitive>(*prim::kPrimZerosLike))
                                           : NewValueNode(std::make_shared<Primitive>(*prim::kPrimOnesLike)));
    auto value_node = Common::CreateValueNodeByValue(value, abs);
    auto special_like_value = tape->FuncGraph::NewCNode({prim_node, value_node});
    special_like_value->set_abstract(value_node->abstract());
    return special_like_value;
  }
  if (value->isa<ValueSequence>()) {
    auto tuple = value->cast<ValueSequencePtr>();
    abstract::AbstractSequencePtr abs_seq;
    if (abs == nullptr) {
      abs_seq = Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractSequencePtr>();
    } else {
      abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    }
    return CreateMakeTupleNode(tape, tuple, abs_seq, type);
  }
  if (value->isa<Scalar>()) {
    auto fake_tensor = GetFakeZeroTensor();
    return BuildSpecialNode(tape, fake_tensor, nullptr, type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto data = csr_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  }
  if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto data = coo_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  }
  if (value->isa<ValueDictionary>()) {
    auto v_dict = value->cast<ValueDictionaryPtr>();
    abstract::AbstractDictionaryPtr abs_dict;
    if (abs == nullptr) {
      abs_dict = Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractDictionaryPtr>();
    } else {
      abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    }
    return CreateMakeDictNode(tape, v_dict, abs_dict, type);
  }
  MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
  return BuildSpecialNode(tape, GetFakeZeroTensor(), nullptr, type);
}

AnfNodePtr AutoGrad::BuildSparseTensorNode(const KernelGraphPtr &tape, const ValuePtr &sparse_value,
                                           const AnfNodePtr &dout_value_node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sparse_value);
  if (sparse_value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = sparse_value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto indptr_node = Common::CreateValueNodeByValue(csr_tensor->GetIndptr());
    auto indices_node = Common::CreateValueNodeByValue(csr_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(csr_tensor->shape());
    auto special_like_csr_node = tape->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, dout_value_node, value_shape});
    special_like_csr_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_csr_node;
  }
  if (sparse_value->isa<tensor::COOTensor>()) {
    auto coo_tensor = sparse_value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto indices_node = Common::CreateValueNodeByValue(coo_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(coo_tensor->shape());
    auto special_like_coo_node =
      tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, dout_value_node, value_shape});
    special_like_coo_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_coo_node;
  }
  MS_LOG(EXCEPTION) << "Get invalid sparse tensor";
}

void AutoGrad::SetGradMetaData(const ValuePtr &value, const VariablePtr &variable, const ParameterPtr &param) {
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      MS_LOG(DEBUG) << "Tensor " << tensor->id() << " has no auto_grad_meta_data, create it";
      auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    }
    auto_grad_meta_data->set_variable(variable);
    if (param != nullptr) {
      auto_grad_meta_data->set_parameter(param);
      auto_grad_meta_data->set_input_type(InputType::kParameter);
    }
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &val : value_sequence->value()) {
      SetGradMetaData(val, variable);
    }
  } else if (value->isa<ValueDictionary>()) {
    auto value_dict = value->cast<ValueDictionaryPtr>();
    for (const auto &val : value_dict->value()) {
      SetGradMetaData(val.second, variable);
    }
  }
}

void AutoGrad::SetGradInfoForInputs(
  const ValuePtr &value, const VariablePtr &variable,
  std::vector<std::pair<tensor::BaseTensorPtr, AutoGradMetaDataPtr>> *param_meta_grad_info, const ParameterPtr &param) {
  if (value->isa<tensor::BaseTensor>()) {
    const auto &input_tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(variable);
    auto_grad_meta_data->set_parameter(param);
    param_meta_grad_info->emplace_back(input_tensor, auto_grad_meta_data);
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param_meta_grad_info, param);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param_meta_grad_info, param);
  }
}

// Create fake bprop
void AutoGrad::BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }
  size_t dout_index = cnode->size() - 1;
  const auto &dout = cnode->input(dout_index);
  const auto &dout_cnode = dout->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dout_cnode);
  // Size is same as op_arg size
  size_t input_size = cnode->size() - 2;
  for (size_t i = 1; i < input_size; ++i) {
    (void)outputs->emplace_back(dout_cnode);
  }
}

CallBackFn AutoGrad::CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                         const GraphCallCondition &graph_call_condition) {
  // kFlagJitCallGraph is set true to avoid compilig call_graph whe compiling the main graph
  call_graph->set_flag(kFlagJitCallGraph, true);
  // call graph not inline to grad top
  call_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  // Pynative bprop graph flag
  call_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  // Run graph by single op will use this kFlagPyNativeBpropGraphWithBpropCut flag
  if (graph_call_condition.is_dynamic_shape_process_) {
    call_graph->set_flag(kFlagPyNativeBpropGraphWithBpropCut, false);
    if (!graph_call_condition.is_jit_graph_) {
      call_graph->set_flag(kFlagEnableRunGraphBySingleOp, true);
    }
  }
  pipeline::ResourcePtr resource;
  constexpr auto kNeedCompile = "NeedCompile";
  const auto it = jit_call_graph_compile_cache_.find(cache_key);
  bool need_compile = (it == jit_call_graph_compile_cache_.end());
  if (need_compile) {
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(call_graph);
    if (graph_call_condition.is_func_grad_) {
      auto manager = resource->manager();
      manager->AddFuncGraph(call_graph, false);
      (void)opt::EnvironConversion(resource);
      if (graph_call_condition.jit_out_has_dict_) {
        MS_LOG(DEBUG) << "Jit out is dict, need convert make dict to pyexecute";
        (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
      }
    }
    if (graph_call_condition.is_jit_graph_ || !graph_call_condition.is_dynamic_shape_process_) {
      (void)jit_call_graph_compile_cache_.emplace(cache_key, resource);
    }
    resource->SetResult(kNeedCompile, true);
  } else {
    resource = it->second;
    // If resource func graph not compile(not call run grad graph), but hit cache
    need_compile = resource->GetResult(kNeedCompile).cast<bool>();
  }
  MS_EXCEPTION_IF_NULL(resource);
  bool is_control_flow = graph_call_condition.is_control_flow_;
  auto fn = [resource, need_compile, is_control_flow, kNeedCompile](const VectorRef &arg_list) -> VectorRef {
    if (need_compile) {
      MS_LOG(DEBUG) << "Start emit action for graph " << resource->func_graph()->ToString();
      auto manager = resource->manager();
      manager->AddFuncGraph(resource->func_graph(), true);
      resource->SetBackendAsync([]() { return compile::CreateBackend(); });
      // kFlagJitCallGraph is set false to compile sub graph in control flow
      if (is_control_flow) {
        for (const auto &g : manager->func_graphs()) {
          g->set_flag(kFlagJitCallGraph, false);
        }
      }
      (void)TaskEmitAction(resource);
      (void)ExecuteAction(resource);
      resource->SetResult(kNeedCompile, false);
    }
    MS_LOG(DEBUG) << "Start execute action for graph " << resource->func_graph()->ToString();
    compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
    return utils::cast<VectorRef>((*run)(arg_list));
  };
  return fn;
}

PrimitivePyPtr AutoGrad::BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (!cell_id.empty()) {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  // Only custom op need add this attr, hook function not need.
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  (void)bprop_cut->AddAttr("custom_op_name", MakeValue(prim->name()));
  if (is_need_recompute) {
    (void)bprop_cut->AddAttr("is_recompute", MakeValue(true));
  }
  return bprop_cut;
}

void AutoGrad::CheckRecomputeInputs(const GradParamPtr &grad_param) {
  if (!grad_param->op_grad_info->is_need_recompute) {
    return;
  }
  for (const auto &input : grad_param->op_grad_info->input_value) {
    if (!input->isa<ValueSequence>()) {
      continue;
    }
    const auto &seq = input->cast<ValueSequencePtr>();
    const auto val = seq->value();
    if (NeedGrad(val)) {
      MS_LOG(EXCEPTION) << "For recompute cell, now we do not support calculate tensor's gradient from tuple. "
                           "You need check your inputs of construct function from recompute cell, and not put "
                           "tensors in tuple which need grad!";
    }
  }
}

TopCellInfoPtr AutoGrad::FindPreTopcell(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                                        const std::string &op_info, const ValuePtr &value) {
  const auto &cur_top_cell = grad_executor->top_cell();
  // If top cell is ir grad, which must be first step and pre top cell can not be find
  if (cur_top_cell->is_ir_grad()) {
    // First run top cell, save op output info for replacement
    cur_top_cell->SaveTensorIdWithOpInfo(op_info, value);
    MS_LOG(DEBUG) << "Top cell " << cur_top_cell << " with " << cur_top_cell->already_run_cell_id()
                  << " run firstly, op info " << op_info;
    // First step or in dynamic process, no need forward output replaces
    op_grad_info->need_do_forward_output_replace = false;
    return nullptr;
  }
  if (cur_top_cell->use_dynamic_shape_process()) {
    MS_LOG(DEBUG) << "Current top cell " << cur_top_cell << " is in dynamic process";
    return nullptr;
  }
  // Not first step
  auto pre_top_cell = grad_executor->GetAlreadyRunTopCell(cur_top_cell->already_run_cell_id());
  if (pre_top_cell == nullptr) {
    pre_top_cell = grad_executor->GetPipelineRunTopCell(cur_top_cell->already_run_cell_id());
    MS_EXCEPTION_IF_NULL(pre_top_cell);
  }
  const auto &op_info_with_tensor_object = pre_top_cell->replace_info().op_info_with_tensor_object;
  op_grad_info->used_in_bprop_graph =
    op_info_with_tensor_object.find(op_grad_info->op_info) != op_info_with_tensor_object.end();
  return pre_top_cell;
}

void AutoGrad::UpdateGradOpInfo(const GradExecutor *grad_executor, const OpGradInfoPtr &op_grad_info,
                                const TopCellInfoPtr &pre_top_cell, bool is_jit_graph) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  const auto &top_cell = grad_executor->top_cell();
  if (!is_jit_graph) {
    grad_executor->dynamic_shape()->CheckNodeDynamic(top_cell, op_grad_info);
  }
  if (op_grad_info->need_do_forward_output_replace && op_grad_info->used_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(pre_top_cell);
    top_cell->UpdateTopCellForwardTensorInfoInBpropGraph(op_grad_info->op_info, op_grad_info->out_value,
                                                         pre_top_cell.get());
  }
}

void AutoGrad::ClearAutoGradStaticCache() { jit_call_graph_compile_cache_.clear(); }

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
            MS_LOG(DEBUG) << "Get used input node " << f_node->DebugString();
            (void)node_list->emplace_back(f_node);
          }
        }
      } else {
        if (auto f_node = common::AnfAlgo::VisitKernel(input_c, 0).first; f_node->isa<CNode>() && IsRealOp(f_node)) {
          MS_LOG(DEBUG) << "Get used input node " << f_node->DebugString();
          (void)node_list->emplace_back(f_node);
        }
      }
    }
  }
  // Check output used in single op bprop graph
  if (unused_inputs.find(cnode->size() - 1) == unused_inputs.end()) {
    MS_LOG(DEBUG) << "Get used output node " << cnode->DebugString();
    (void)node_list->emplace_back(cnode);
  }
}
}  // namespace PyNativeAlgo

void DispatchOp(const std::shared_ptr<runtime::AsyncTask> &task) {
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
