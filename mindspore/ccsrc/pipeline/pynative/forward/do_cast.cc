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

#include "pipeline/pynative/forward/do_cast.h"
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/core/ops/array_ops.h"
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace pynative {
void CastOperation::DoCast(const FrontendOpRunInfoPtr &op_run_info) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeCast,
                                     op_run_info->base_op_run_info.op_name, true);
  // Mixed precision conversion tensors which has cast dtype
  SetTensorMixPrecisionCast(op_run_info);
  // Implicit transform
  SetImplicitCast(op_run_info);
}

void CastOperation::ClearRes() {
  implicit_cast_map_.clear();
  type_prim_cache_.clear();
}

bool CastOperation::IsValueTypeInvalid(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  return !v->isa<tensor::BaseTensor>() && !v->isa<tensor::CSRTensor>() && !v->isa<IntegerImm>() &&
         !v->isa<FloatImm>() && !v->isa<BoolImm>();
}

ValuePtr CastOperation::DoNormalCast(const FrontendOpRunInfoPtr &cast_run_info, const ValuePtr &v,
                                     const TypeId &type_id) const {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(cast_run_info);
  // Step 1: Cast scalar value to another scalar value with destination data type.
  // It is used to avoid to call `cast infer value function` or launch cast op to backend.
  ValuePtr dst_value = ScalarToDstDtypeValue(v, std::make_pair(type_id, true));
  if (dst_value != nullptr) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString() << " cast to value: " << dst_value->ToString();
    cast_run_info->real_out = dst_value;
    return dst_value;
  }

  if (v->isa<tensor::BaseTensor>()) {
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    if (type_id == tensor->data_type()) {
      cast_run_info->real_out = v;
      return cast_run_info->real_out;
    }
  }

  constexpr auto input_size = 2;
  cast_run_info->op_grad_info->op_prim = GetPrimByTypeId(type_id);
  auto type_id64 = std::make_shared<Int64Imm>(static_cast<int64_t>(type_id));
  PyNativeAlgo::Common::GetConstInputToAttr(
    cast_run_info->op_grad_info->op_prim, cast_run_info->base_op_run_info.op_name,
    cast_run_info->base_op_run_info.device_target, false, &cast_run_info->input_to_attr);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(v);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(type_id64);
  cast_run_info->input_size = input_size;
  cast_run_info->op_grad_info->input_value_grad_type.resize(input_size);
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->RunOpFrontend(cast_run_info);
  return cast_run_info->real_out;
}

ValuePtr CastOperation::DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                   const std::pair<TypeId, bool> &dst_type, const std::string &op_name,
                                   size_t index) const {
  MS_EXCEPTION_IF_NULL(v);
  // Step 1: Cast scalar value to another scalar value with destination data type.
  // It is used to avoid to call `cast infer value function` or launch cast op to backend.
  ValuePtr dst_value = ScalarToDstDtypeValue(v, dst_type);
  if (dst_value != nullptr) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString() << " cast to value: " << dst_value->ToString();
    return dst_value;
  }
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->source_type[index] != ops::OP_DTYPE::DT_BEGIN && v->isa<tensor::BaseTensor>()) {
    MS_LOG(DEBUG) << "Source value: " << v->ToString();
    dst_value = TensorToDstDtypeValue(v, dst_type.first);
    MS_LOG(DEBUG) << "Cast to value: " << dst_value->ToString() << " without dispatching cast op";
    return dst_value;
  }
  // When step 1 does not work, creating a cast op to get destination data type value.
  constexpr auto input_size = 2;
  const auto &cast_run_info = std::make_shared<FrontendOpRunInfo>();
  auto cast_prim = GetPrimByTypeId(dst_type.first);
  auto type_id64 = std::make_shared<Int64Imm>(static_cast<int64_t>(dst_type.first));
  cast_run_info->requires_grad = op_run_info->requires_grad;
  cast_run_info->base_op_run_info.op_name = prim::kPrimCast->name();
  cast_run_info->base_op_run_info.is_mixed_precision_cast = true;
  cast_run_info->base_op_run_info.next_op_name = op_name;
  cast_run_info->base_op_run_info.next_input_index = index;
  cast_run_info->base_op_run_info.use_dynamic_shape_process = op_run_info->base_op_run_info.use_dynamic_shape_process;
  cast_run_info->base_op_run_info.device_target =
    PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->GetCurrentDeviceTarget(cast_prim);
  bool is_dynamic_shape =
    cast_run_info->base_op_run_info.has_dynamic_output || cast_run_info->base_op_run_info.use_dynamic_shape_process;
  PyNativeAlgo::Common::GetConstInputToAttr(cast_prim, cast_run_info->base_op_run_info.op_name,
                                            cast_run_info->base_op_run_info.device_target, is_dynamic_shape,
                                            &cast_run_info->input_to_attr);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(v);
  (void)cast_run_info->op_grad_info->input_value.emplace_back(type_id64);
  cast_run_info->input_size = input_size;
  cast_run_info->op_grad_info->input_value_grad_type.resize(input_size);
  cast_run_info->op_grad_info->op_prim = cast_prim;
  PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->RunOpFrontend(cast_run_info);
  return cast_run_info->real_out;
}

ValuePtr CastOperation::DoParamMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast,
                                                const ValuePtr &v, const std::string &op_name, size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(is_cast);
  MS_EXCEPTION_IF_NULL(v);
  if (op_run_info->mix_type != kNotSet) {
    auto dst_dtype = kFloat16;
    if (op_run_info->mix_type == kFP32) {
      dst_dtype = kFloat32;
    } else if (op_run_info->mix_type == kBF16) {
      dst_dtype = kBFloat16;
    }
    const auto &tensor = v->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto source_dtype = tensor->Dtype();
    if (source_dtype != nullptr && (IsSubType(source_dtype, kFloat) || IsSubType(source_dtype, kBFloat)) &&
        *source_dtype != *dst_dtype) {
      MS_LOG(DEBUG) << "MixPrecision cast for " << op_run_info->base_op_run_info.op_name << " " << index
                    << "th input, and to type " << dst_dtype->ToString();
      *is_cast = true;
      return DoAutoCast(op_run_info, tensor, std::make_pair(dst_dtype->type_id(), true), op_name, index);
    }
  }
  return v;
}

ValuePtr CastOperation::DoParamMixPrecisionCastTuple(const FrontendOpRunInfoPtr &op_run_info, bool *is_cast,
                                                     const ValueSequencePtr &value_seq, const std::string &op_name,
                                                     size_t index) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(is_cast);
  MS_EXCEPTION_IF_NULL(value_seq);
  size_t tuple_size = value_seq->size();
  const auto &value_tuple = value_seq->value();
  ValuePtrList result(tuple_size, nullptr);
  for (size_t i = 0; i < tuple_size; i++) {
    if (value_tuple[i]->isa<tensor::MetaTensor>()) {
      MS_LOG(DEBUG) << "Call cast for item " << i;
      result[i] = DoParamMixPrecisionCast(op_run_info, is_cast, value_tuple[i], op_name, index);
    } else if (value_tuple[i]->isa<ValueSequence>()) {
      result[i] =
        DoParamMixPrecisionCastTuple(op_run_info, is_cast, value_tuple[i]->cast<ValueSequencePtr>(), op_name, index);
    } else {
      result[i] = value_tuple[i];
    }
  }
  if (value_seq->isa<ValueList>()) {
    return std::make_shared<ValueList>(result);
  } else {
    return std::make_shared<ValueTuple>(result);
  }
}

void CastOperation::DoSignatureCast(const FrontendOpRunInfoPtr &op_run_info,
                                    const std::map<SignatureEnumDType, std::pair<TypeId, bool>> &dst_type,
                                    const std::vector<SignatureEnumDType> &dtypes) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  const auto &signature = op_run_info->signatures;
  auto &input_args = op_run_info->op_grad_info->input_value;
  size_t input_args_size = input_args.size();
  if (dtypes.size() > input_args_size) {
    MS_LOG(EXCEPTION) << "Signature dtypes size[" << dtypes << "] is greater than input_args_size[" << input_args_size
                      << "].";
  }
  for (size_t i = 0; i < dtypes.size(); ++i) {
    // No need to implicit cast if no dtype.
    if (dtypes[i] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      MS_LOG(DEBUG) << "Get kDTypeEmptyDefaultValue";
      continue;
    }
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second.first == kTypeUnknown) {
      MS_LOG(DEBUG) << "Can not find dtype " << (it == dst_type.end()) << ", or type is unknown "
                    << (it->second.first == kTypeUnknown);
      continue;
    }
    const auto &v = input_args[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    TypeId arg_type_id = kTypeUnknown;
    if (v->isa<tensor::MetaTensor>()) {
      const auto &arg = v->cast<tensor::MetaTensorPtr>();
      arg_type_id = arg->data_type();
    }
    // Implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (arg_type_id == it->second.first);
    }
    if (sig == SignatureEnumRW::kRWWrite && arg_type_id != kTypeUnknown && !is_same_type) {
      prim::RaiseExceptionForConvertRefDtype(op_run_info->op_grad_info->op_prim, TypeIdToString(arg_type_id),
                                             TypeIdToString(it->second.first), i);
    }
    if (is_same_type) {
      MS_LOG(DEBUG) << "Get same dtype";
      continue;
    }

    if (IsValueTypeInvalid(v)) {
      std::string type_str = v->type() == nullptr ? "None, value is \"" + v->ToString() + "\"" : v->type()->ToString();
      MS_EXCEPTION(TypeError) << "For '" << op_run_info->op_grad_info->op_prim->name() << "', the " << (i + 1)
                              << "th input " << signature[i].name << " can not be implicitly converted. "
                              << "Its type is " << type_str << ". Only support Tensor or Scalar.";
    }
    MS_LOG(DEBUG) << "Implicit cast for " << op_run_info->base_op_run_info.op_name << " " << i << "th input, from type "
                  << (v->type() == nullptr ? v->ToString() : v->type()->ToString()) << " to type "
                  << TypeIdToType(it->second.first)->ToString();
    input_args[i] = DoAutoCast(op_run_info, v, it->second, op_run_info->base_op_run_info.op_name, i);
  }
}

void CastOperation::SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->async_status.disable_mix_precision) {
    // Pure function running, mix precision cast is disable, or cell not set mix precision
    MS_LOG(DEBUG) << "No mix precision for " << op_run_info->base_op_run_info.op_name;
    return;
  }
  MS_EXCEPTION_IF_NULL(op_run_info->op_grad_info->op_prim);
  const auto &signature = op_run_info->signatures;
  for (size_t i = 0; i < op_run_info->none_init_inputs_num; i++) {
    const auto &v = op_run_info->op_grad_info->input_value[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    // mix precision for non param
    bool is_cast = false;
    ValuePtr cast_output = nullptr;
    if (v->isa<tensor::MetaTensor>()) {
      auto meta_tensor = v->cast<tensor::MetaTensorPtr>();
      if (meta_tensor && meta_tensor->is_parameter()) {
        // If parameter write(not kRWRead), no need cast
        if (sig != SignatureEnumRW::kRWRead) {
          continue;
        }
      }
      cast_output = DoParamMixPrecisionCast(op_run_info, &is_cast, v, op_run_info->op_grad_info->op_prim->name(), i);
    } else if (v->isa<ValueSequence>()) {
      // mix precision for tuple inputs
      cast_output = DoParamMixPrecisionCastTuple(op_run_info, &is_cast, v->cast<ValueSequencePtr>(),
                                                 op_run_info->op_grad_info->op_prim->name(), i);
    }
    if (is_cast) {
      MS_EXCEPTION_IF_NULL(cast_output);
      op_run_info->op_grad_info->input_value[i] = cast_output;
    }
  }
}

namespace {
std::pair<std::vector<TypeId>, std::vector<bool>> GetTypeInfo(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<TypeId> args_type_id;
  std::vector<bool> args_has_tensor;
  args_type_id.resize(op_run_info->input_size);
  args_has_tensor.resize(op_run_info->input_size, false);

  const auto &input_value = op_run_info->op_grad_info->input_value;
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    if (input_value[i]->isa<tensor::BaseTensor>()) {
      args_type_id[i] = input_value[i]->cast<tensor::BaseTensorPtr>()->data_type();
      if (op_run_info->source_type[i] == ops::OP_DTYPE::DT_BEGIN) {
        args_has_tensor[i] = true;
      }
    } else if (input_value[i]->isa<Scalar>()) {
      const auto type = input_value[i]->cast<ScalarPtr>()->type();
      MS_EXCEPTION_IF_NULL(type);
      args_type_id[i] = type->type_id();
    } else {
      MS_LOG(DEBUG) << "Get input value " << input_value[i]->ToString();
      args_type_id[i] = kTypeUnknown;
    }
  }
  return {args_type_id, args_has_tensor};
}
}  // namespace

void CastOperation::SetImplicitCast(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &it = implicit_cast_map_.find(prim->name());
  if (it == implicit_cast_map_.end()) {
    std::vector<SignatureEnumDType> dtypes;
    bool has_dtype_sig = GetSignatureType(op_run_info->signatures, &dtypes);
    if (!has_dtype_sig) {
      PrimSignature sig_value{has_dtype_sig, {}};
      implicit_cast_map_[prim->name()] = sig_value;
      MS_LOG(DEBUG) << "Op " << prim->name() << " has no signature";
      return;
    }
    const auto &signature = op_run_info->signatures;
    auto sig_size = signature.size();
    // Ignore monad signature
    for (const auto &sig : signature) {
      if (sig.default_value != nullptr && sig.default_value->isa<Monad>()) {
        --sig_size;
      }
    }
    if (sig_size > 0 && sig_size != op_run_info->none_init_inputs_num) {
      MS_EXCEPTION(ValueError) << op_run_info->base_op_run_info.op_name << " inputs number "
                               << op_run_info->none_init_inputs_num << " does not match the requires "
                               << "signature size " << sig_size;
    }

    auto [args_type_id, args_has_tensor] = GetTypeInfo(op_run_info);
    auto dst_type = GetSignatureTypeMap(dtypes, args_type_id, args_has_tensor);
    DoSignatureCast(op_run_info, dst_type, dtypes);
    PrimSignature sig_value{has_dtype_sig, dtypes};
    implicit_cast_map_[prim->name()] = sig_value;
  } else {
    if (!it->second.has_dtype_sig) {
      MS_LOG(DEBUG) << op_run_info->base_op_run_info.op_name << " have no dtype sig";
      return;
    }
    MS_LOG(DEBUG) << "Do signature for " << op_run_info->base_op_run_info.op_name << " with cache";
    auto [args_type_id, args_has_tensor] = GetTypeInfo(op_run_info);
    auto dst_type = GetSignatureTypeMap(it->second.dtypes, args_type_id, args_has_tensor);
    DoSignatureCast(op_run_info, dst_type, it->second.dtypes);
  }
}
}  // namespace pynative
}  // namespace mindspore
