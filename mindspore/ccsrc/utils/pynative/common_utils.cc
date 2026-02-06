/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/utils/pynative/common_utils.h"

#include <string>
#include <vector>
#include <utility>
#include "abstract/abstract_value.h"
#include "base/float8_e4m3fn.h"
#include "ir/core_ops_primitive.h"
#include "ir/manager.h"
#include "ir/tensor_new.h"
#include "utils/ms_context.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/tensor_py.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/pynative/variable.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
std::function<void(void)> wait_bprop_callback = nullptr;
void RegisterWaitBpropFunc(const std::function<void(void)> &wait_func) { wait_bprop_callback = wait_func; }

std::function<void(void)> GetWaitBpropFunc() {
  MS_EXCEPTION_IF_NULL(wait_bprop_callback);
  return wait_bprop_callback;
}

namespace {
void PlantTupleParam(const FuncGraphPtr &bprop_graph, const abstract::AbstractSequencePtr &abs_seq,
                     AnfNodePtrList *make_tuple, AnfNodePtrList *new_param) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  MS_EXCEPTION_IF_NULL(make_tuple);
  MS_EXCEPTION_IF_NULL(new_param);
  MS_EXCEPTION_IF_NULL(abs_seq);
  for (size_t i = 0; i < abs_seq->size(); ++i) {
    const auto &cur_abs = abs_seq->elements()[i];
    if (cur_abs->isa<abstract::AbstractSequence>()) {
      auto is_tuple = cur_abs->isa<abstract::AbstractTuple>();
      AnfNodePtrList cur_make_tuple_inputs;
      auto prim = is_tuple ? prim::kPrimMakeTuple : prim::kPrimMakeList;
      (void)cur_make_tuple_inputs.emplace_back(NewValueNode(prim));
      PlantTupleParam(bprop_graph, cur_abs->cast<abstract::AbstractSequencePtr>(), &cur_make_tuple_inputs, new_param);
      auto cur_make_tuple_node = bprop_graph->NewCNode(cur_make_tuple_inputs);
      AbstractBasePtrList cur_abstract_elements;
      (void)std::transform(cur_make_tuple_inputs.begin() + 1, cur_make_tuple_inputs.end(),
                           std::back_inserter(cur_abstract_elements), [](const auto &e) { return e->abstract(); });
      AbstractBasePtr cur_abstract;
      if (is_tuple) {
        cur_abstract = std::make_shared<abstract::AbstractTuple>(cur_abstract_elements);
      } else {
        cur_abstract = std::make_shared<abstract::AbstractList>(cur_abstract_elements);
      }
      cur_make_tuple_node->set_abstract(cur_abstract);
      (void)make_tuple->emplace_back(cur_make_tuple_node);
    } else if (cur_abs->isa<abstract::AbstractTensor>()) {
      auto plant_param = bprop_graph->add_parameter();
      plant_param->set_abstract(cur_abs);
      (void)make_tuple->emplace_back(plant_param);
      (void)new_param->emplace_back(plant_param);
    } else if (cur_abs->isa<abstract::AbstractDictionary>()) {
      // Support output type of tuple contains dict
      const auto &abs_dict = cur_abs->cast<abstract::AbstractDictionaryPtr>();
      abstract::AbstractBasePtrList local_key_abs_inputs;
      abstract::AbstractBasePtrList local_value_abs_inputs;
      for (const auto &element : abs_dict->elements()) {
        (void)local_key_abs_inputs.emplace_back(element.first);
        (void)local_value_abs_inputs.emplace_back(element.second);
      }
      auto key_param = bprop_graph->add_parameter();
      key_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
      auto value_param = bprop_graph->add_parameter();
      value_param->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
      (void)new_param->emplace_back(key_param);
      (void)new_param->emplace_back(value_param);
      // Add Makedict node as tuple element
      auto dict_node = bprop_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), key_param, value_param});
      dict_node->set_abstract(abs_dict);
      (void)make_tuple->emplace_back(dict_node);
    } else {
      auto value = MakeValue(static_cast<int64_t>(0));
      auto value_node = NewValueNode(value);
      value_node->set_abstract(value->ToAbstract());
      (void)make_tuple->emplace_back(value_node);
    }
  }
}

template <typename S>
ValuePtr CastScalarToScalar(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return MakeValue(static_cast<int>(in));
    case kNumberTypeFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    case kNumberTypeFloat32:
      return MakeValue(static_cast<float>(in));
    case kNumberTypeBool:
      return MakeValue(static_cast<bool>(in));
    case kNumberTypeInt64:
      return MakeValue(static_cast<int64_t>(in));
    case kNumberTypeFloat64:
      return MakeValue(static_cast<double>(in));
    case kNumberTypeInt16:
      return MakeValue(static_cast<int16_t>(in));
    case kNumberTypeInt8:
      return MakeValue(static_cast<int8_t>(in));
    case kNumberTypeUInt64:
      return MakeValue(static_cast<uint64_t>(in));
    case kNumberTypeUInt32:
      return MakeValue(static_cast<uint32_t>(in));
    case kNumberTypeUInt16:
      return MakeValue(static_cast<uint16_t>(in));
    case kNumberTypeUInt8:
      return MakeValue(static_cast<uint8_t>(in));
    case kNumberTypeBFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    case kNumberTypeHiFloat8:
      return MakeValue(static_cast<hifloat8>(in).int_value());
    case kNumberTypeFloat8E5M2:
      return MakeValue(static_cast<float8_e5m2>(in).int_value());
    case kNumberTypeFloat8E4M3FN:
      return MakeValue(static_cast<float8_e4m3fn>(in).int_value());
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr CastScalarToTensor(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return tensor::from_scalar(static_cast<int>(in), kInt32);
    case kNumberTypeFloat16:
      return tensor::from_scalar(static_cast<float16>(in), kFloat16);
    case kNumberTypeFloat32:
      return tensor::from_scalar(static_cast<float>(in), kFloat32);
    case kNumberTypeBool:
      return tensor::from_scalar(static_cast<bool>(in), kBool);
    case kNumberTypeInt64:
      return tensor::from_scalar(static_cast<int64_t>(in), kInt64);
    case kNumberTypeFloat64:
      return tensor::from_scalar(static_cast<double>(in), kFloat64);
    case kNumberTypeInt16:
      return tensor::from_scalar(static_cast<int16_t>(in), kInt16);
    case kNumberTypeInt8:
      return tensor::from_scalar(static_cast<int8_t>(in), kInt8);
    case kNumberTypeUInt64:
      return tensor::from_scalar(static_cast<uint64_t>(in), kUInt64);
    case kNumberTypeUInt32:
      return tensor::from_scalar(static_cast<uint32_t>(in), kUInt32);
    case kNumberTypeUInt16:
      return tensor::from_scalar(static_cast<uint16_t>(in), kUInt16);
    case kNumberTypeUInt8:
      return tensor::from_scalar(static_cast<uint8_t>(in), kUInt8);
    case kNumberTypeBFloat16:
      return tensor::from_scalar(static_cast<bfloat16>(in), kBFloat16);
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr Cast(S in, const std::pair<TypeId, bool> &dst_type) {
  bool has_tensor_input = dst_type.second;
  if (has_tensor_input) {
    return CastScalarToTensor(in, dst_type.first);
  }
  return CastScalarToScalar(in, dst_type.first);
}
}  // namespace

void CommonUtils::ProcessTupleParam(const FuncGraphPtr &bprop_graph, size_t position) {
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
  AbstractBasePtrList cur_abstract_elements;
  (void)std::transform(make_tuple.begin() + 1, make_tuple.end(), std::back_inserter(cur_abstract_elements),
                       [](const auto &e) { return e->abstract(); });
  AbstractBasePtr cur_abstract = std::make_shared<abstract::AbstractTuple>(cur_abstract_elements);
  make_tuple_param->set_abstract(cur_abstract);
  auto manager = bprop_graph->manager();
  if (manager == nullptr) {
    manager = MakeManager({bprop_graph}, false);
  }
  MS_EXCEPTION_IF_NULL(manager);
  auto tr = manager->Transact();
  (void)tr.Replace(target_param, make_tuple_param);
  tr.Commit();
}

void CommonUtils::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR(filename, graph);
  }
#endif
}

void CommonUtils::ProcessDictParam(const FuncGraphPtr &bprop_graph, size_t position) {
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

AbstractBasePtr CommonUtils::SetAbstractValueToAnyValue(const AbstractBasePtr &abs) {
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

ValuePtrList CommonUtils::FlattenOnlyTensor(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList outputs;
  CommonUtils::FlattenValueSeqArg(v, false, true, &outputs);
  return outputs;
}

ValuePtrList CommonUtils::FlattenTensorSeqInValueSeq(const ValuePtrList &v, bool only_flatten_tensor) {
  ValuePtrList outputs;
  for (const auto &item : v) {
    FlattenValueSeqArg(item, only_flatten_tensor, false, &outputs);
  }
  return outputs;
}

ValuePtrList CommonUtils::FlattenTensorSeqInValue(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList outputs;
  FlattenValueSeqArg(v, true, false, &outputs);
  return outputs;
}

void CommonUtils::FlattenValueSeqArg(const ValuePtr &v, bool is_only_flatten_tensor_seq, bool is_filter_tensor,
                                     std::vector<ValuePtr> *flatten_v) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(flatten_v);
  MS_LOG(DEBUG) << "Get is only flatten tensor seq " << is_only_flatten_tensor_seq;
  if (v->isa<tensor::Tensor>()) {
    (void)flatten_v->emplace_back(v);
  } else if (v->isa<ValueSequence>()) {
    const auto &v_vec = v->cast<ValueSequencePtr>()->value();
    if (v_vec.empty() && !is_filter_tensor) {
      MS_LOG(DEBUG) << "Get empty tuple value";
      (void)flatten_v->emplace_back(v);
      MS_LOG(DEBUG) << "Get empty value sequence";
      return;
    }
    if (is_only_flatten_tensor_seq && !v_vec.front()->isa<tensor::Tensor>()) {
      (void)flatten_v->emplace_back(v);
    } else {
      for (const auto &elem : v_vec) {
        FlattenValueSeqArg(elem, is_only_flatten_tensor_seq, is_filter_tensor, flatten_v);
      }
    }
  } else if (v->isa<ValueDictionary>()) {
    auto dic_v = v->cast<ValueDictionaryPtr>();
    for (const auto &elem : dic_v->value()) {
      FlattenValueSeqArg(elem.second, is_only_flatten_tensor_seq, is_filter_tensor, flatten_v);
    }
  } else if (!is_filter_tensor) {
    MS_LOG(DEBUG) << "Get not tensor value: " << v->ToString();
    (void)flatten_v->emplace_back(v);
  }
}

tensor::TensorPtr CommonUtils::ShallowCopyAndDetachForTensor(const tensor::TensorPtr &tensor) {
  auto copy_tensor = std::make_shared<tensor::Tensor>(*tensor);
  copy_tensor->set_storage_info(tensor->storage_info());
  copy_tensor->set_auto_grad_meta_data(nullptr);
  return copy_tensor;
}

ValuePtr CommonUtils::ShallowCopyAndDetach(const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    return ShallowCopyAndDetachForTensor(tensor);
  } else if (value->isa<ValueSequence>()) {
    auto val_seq = value->cast<ValueSequencePtr>();
    std::vector<ValuePtr> res;
    res.reserve(val_seq->size());
    for (const auto &val : val_seq->value()) {
      (void)res.emplace_back(ShallowCopyAndDetach(val));
    }
    return std::make_shared<ValueTuple>(res);
  }
  return value;
}

tensor::TensorPtr CastUtils::TensorToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id) {
  MS_EXCEPTION_IF_NULL(src_value);
  auto src_tensor = src_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(src_tensor);
  (void)src_tensor->set_data_type(dst_type_id);
  return src_tensor;
}

// This function is used to convert scalar value to another scalar value with destination data type.
// The scope of scalar type includes common data types, such as `FP64`, `FP32`, `FP16, `Int64`, `Int32`, ...
// The following sort is based on the hot spots of the data type.
ValuePtr CastUtils::ScalarToDstDtypeValue(const ValuePtr &src_value, const std::pair<TypeId, bool> &dst_type) {
  MS_EXCEPTION_IF_NULL(src_value);
  // Tensor not do scalar cast
  if (src_value->isa<tensor::Tensor>()) {
    return nullptr;
  }
  if (src_value->isa<Int64Imm>()) {
    const auto &int64_v = src_value->cast<Int64ImmPtr>();
    return Cast<int64_t>(int64_v->value(), dst_type);
  }
  if (src_value->isa<FP32Imm>()) {
    const auto &fp32_v = src_value->cast<FP32ImmPtr>();
    return Cast<float>(fp32_v->value(), dst_type);
  }
  if (src_value->isa<Int32Imm>()) {
    const auto &int32_v = src_value->cast<Int32ImmPtr>();
    return Cast<int32_t>(int32_v->value(), dst_type);
  }
  if (src_value->isa<FP64Imm>()) {
    const auto &fp64_v = src_value->cast<FP64ImmPtr>();
    return Cast<double>(fp64_v->value(), dst_type);
  }
  if (src_value->isa<BoolImm>()) {
    const auto &bool_v = src_value->cast<BoolImmPtr>();
    return Cast<bool>(bool_v->value(), dst_type);
  }
  if (src_value->isa<Int16Imm>()) {
    const auto &int16_v = src_value->cast<Int16ImmPtr>();
    return Cast<int16_t>(int16_v->value(), dst_type);
  }
  MS_LOG(DEBUG) << "Now, the value [" << src_value->ToString() << "] is not supported to cast directly.";
  return nullptr;
}
}  // namespace pynative

bool HookUtils::HasRegisterHook(const py::object &obj) {
  if (!tensor::IsTensorPy(obj)) {
    return false;
  }
  auto tensor = tensor::ConvertToTensor(obj);
  const auto &grad_meta_data = tensor->auto_grad_meta_data();
  if (grad_meta_data == nullptr) {
    return false;
  }
  pynative::GetWaitBpropFunc()();
  const auto &grad_node = grad_meta_data->UnsafeGetGradNodeImpl();
  if (grad_node == nullptr) {
    return false;
  }
  return grad_node->py_tensor_pre_hooks() && !grad_node->py_tensor_pre_hooks()->empty();
}

py::list HookUtils::GetRegisterHookList(const py::object &obj) {
  if (!HasRegisterHook(obj)) {
    return py::list();
  }
  py::list hook_fn_list;
  auto tensor = tensor::ConvertToTensor(obj);
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(grad_meta_data);
  const auto &grad_node = grad_meta_data->UnsafeGetGradNodeImpl();
  if (const auto &backward_hooks = grad_node->py_tensor_pre_hooks()) {
    for (const auto &[id, hook] : *backward_hooks) {
      auto fn = hook->hook_fn_;
      if (py::isinstance<py::none>(fn)) {
        MS_LOG(DEBUG) << "Hook of Tensor[" << id << "] is None.";
        continue;
      }
      hook_fn_list.append(fn);
    }
  }
  return hook_fn_list;
}
}  // namespace mindspore
