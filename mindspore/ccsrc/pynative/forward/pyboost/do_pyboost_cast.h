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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_PYBOOST_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_PYBOOST_H_

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <map>
#include <utility>
#include "pynative/forward/cast_base.h"
#include "utils/convert_utils.h"
#include "include/utils/operator/primitive_utils.h"
#include "ir/cell.h"

namespace mindspore {
namespace pynative {
static constexpr auto kCast = "Cast";

class PyBoostCastOperation : public CastBaseOperation {
 public:
  PyBoostCastOperation() = default;
  ~PyBoostCastOperation() = default;

  template <typename... InputArgs, std::size_t... Index>
  auto SetTensorMixPrecisionCastHelper(const PyboostOpRunInfoPtr &op_run_info, const size_t &input_size,
                                       std::index_sequence<Index...>, const InputArgs &...input_args) {
    if (op_run_info->mix_type == kAutoPromote) {
      auto [args_type_id, args_has_tensor] = GetTypeInfo(op_run_info, input_size, std::make_tuple(input_args...),
                                                         std::make_index_sequence<sizeof...(InputArgs)>{});
      auto promote_type_id = GetMixPrecisionPromoteType(args_type_id, args_has_tensor);
      if (promote_type_id != kTypeUnknown) {
        op_run_info->mix_precision_type = TypeIdToType(promote_type_id);
      }
      MS_LOG(DEBUG) << "Set op " << op_run_info->op_prim->name() << " promote type " << TypeIdToString(promote_type_id);
    }
    return std::make_tuple(SetTensorMixPrecisionCast(op_run_info, input_args, Index)...);
  }

  template <typename... InputArgs>
  auto DoMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const size_t &input_size,
                          const InputArgs &...input_args) {
    // Mixed precision conversion tensors which has cast dtype
    if (op_run_info->async_status.disable_mix_precision) {
      return std::make_tuple(input_args...);
    }
    return SetTensorMixPrecisionCastHelper(op_run_info, input_size, std::make_index_sequence<sizeof...(InputArgs)>(),
                                           input_args...);
  }

  template <typename T>
  T SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const T &t, size_t index) const {
    MS_EXCEPTION_IF_NULL(t);
    MS_LOG(DEBUG) << "Get input type " << typeid(t).name();
    return t;
  }

  template <typename Item>
  void GetTypeIdInfo(const PyboostOpRunInfoPtr &op_run_info, const size_t &input_size,
                     std::vector<TypeId> *args_type_id, std::vector<bool> *args_has_tensor, size_t i, const Item &v) {
    MS_EXCEPTION_IF_NULL(v);
    MS_LOG(DEBUG) << "Get type info of " << v->ToString();
    if (v->template isa<tensor::Tensor>()) {
      args_type_id->push_back(v->template cast<tensor::TensorPtr>()->data_type());
      // Indicate have do type cast
      if (op_run_info->source_type[i] == ops::OP_DTYPE::DT_BEGIN) {
        args_has_tensor->push_back(true);
      } else {
        args_has_tensor->push_back(false);
      }
    } else if (v->template isa<Scalar>()) {
      const auto type = v->template cast<ScalarPtr>()->type();
      MS_EXCEPTION_IF_NULL(type);
      args_type_id->push_back(type->type_id());
      args_has_tensor->push_back(false);
    } else if (v->template isa<ValueSequence>()) {
      const auto &value_seq = v->template cast<ValueSequencePtr>();
      MS_EXCEPTION_IF_NULL(value_seq);
      const auto &elements = value_seq->value();
      bool has_tensor = false;
      // Not support tuple(tuple<Tensor>) yet.
      for (const auto &element : elements) {
        MS_EXCEPTION_IF_NULL(element);
        MS_LOG(DEBUG) << "Get tuple element " << element->ToString();
        if (element->template isa<tensor::Tensor>()) {
          args_type_id->push_back(element->template cast<tensor::TensorPtr>()->data_type());
          // No tuple[int] to tuple[Tensor] type_cast yet.
          args_has_tensor->push_back(true);
          has_tensor = true;
        }
      }
      // For tuple<scalar>
      if (!has_tensor) {
        args_type_id->push_back(kTypeUnknown);
        args_has_tensor->push_back(false);
      }
    } else {
      MS_LOG(DEBUG) << "Get value " << v->ToString();
      args_type_id->push_back(kTypeUnknown);
      args_has_tensor->push_back(false);
    }
  }

  template <typename Item>
  void GetTypeIdInfo(const PyboostOpRunInfoPtr &op_run_info, const size_t &input_size,
                     std::vector<TypeId> *args_type_id, std::vector<bool> *args_has_tensor, size_t i,
                     const std::optional<Item> &t) {
    args_type_id->push_back(kTypeUnknown);
    args_has_tensor->push_back(false);
  }

  template <typename TupleInput, size_t... Index>
  std::pair<std::vector<TypeId>, std::vector<bool>> GetTypeInfo(const PyboostOpRunInfoPtr &op_run_info,
                                                                const size_t &input_size, const TupleInput &tuple_input,
                                                                std::index_sequence<Index...>) {
    std::vector<TypeId> args_type_id;
    std::vector<bool> args_has_tensor;

    args_type_id.reserve(input_size);
    args_has_tensor.reserve(input_size);
    (GetTypeIdInfo(op_run_info, input_size, &args_type_id, &args_has_tensor, Index, std::get<Index>(tuple_input)), ...);
    return {args_type_id, args_has_tensor};
  }

  // Implicit transform
  template <size_t N, typename... InputArgs>
  auto DoImplicitCast(const PyboostOpRunInfoPtr &op_run_info, const size_t &input_size,
                      const std::vector<std::vector<size_t>> &same_type_table,
                      const std::tuple<InputArgs...> &input_args) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    MS_LOG(DEBUG) << "Get signature " << same_type_table;
    const auto &it = implicit_cast_map_.find(op_run_info->op_prim->name());
    const auto &op_def = ops::GetOpDef(op_run_info->op_prim->name());
    MS_EXCEPTION_IF_NULL(op_def);
    if (it == implicit_cast_map_.end()) {
      std::vector<SignatureEnumDType> dtypes;
      // Get current inputs signatures
      bool has_dtype_sig = GetSignatureType(op_def->signatures_, &dtypes);
      if (dtypes.size() > input_size) {
        MS_LOG(EXCEPTION) << "Signature dtypes size[" << dtypes << "] is greater than input_args_size[" << input_size
                          << "].";
      }
      if (!has_dtype_sig) {
        PrimSignature sig_value{has_dtype_sig, {}};
        implicit_cast_map_[op_run_info->op_prim->name()] = sig_value;
        MS_LOG(DEBUG) << "Op " << op_run_info->op_prim->name() << " has no signature";
        return input_args;
      }
      PrimSignature sig_value{has_dtype_sig, dtypes};
      implicit_cast_map_[op_run_info->op_prim->name()] = sig_value;

      auto [args_type_id, args_has_tensor] =
        GetTypeInfo(op_run_info, input_size, input_args, std::make_index_sequence<sizeof...(InputArgs)>{});
      auto dst_type = GetSignatureTypeMap(dtypes, args_type_id, args_has_tensor);
      return SetImplicitCast(op_run_info, op_def->signatures_, dst_type, dtypes, input_args,
                             std::make_index_sequence<sizeof...(InputArgs)>{});
    } else {
      if (!it->second.has_dtype_sig) {
        MS_LOG(DEBUG) << op_run_info->op_prim->name() << " have no dtype sig";
        return input_args;
      }
      MS_LOG(DEBUG) << "Do signature for " << op_run_info->op_prim->name() << " with cache";
      auto [args_type_id, args_has_tensor] =
        GetTypeInfo(op_run_info, input_size, input_args, std::make_index_sequence<sizeof...(InputArgs)>{});
      auto dst_type = GetSignatureTypeMap(it->second.dtypes, args_type_id, args_has_tensor);
      return SetImplicitCast(op_run_info, op_def->signatures_, dst_type, it->second.dtypes, input_args,
                             std::make_index_sequence<sizeof...(InputArgs)>{});
    }
  }

 private:
  template <typename TupleInput, size_t... N>
  auto SetImplicitCast(const PyboostOpRunInfoPtr &op_run_info, const std::vector<Signature> &signatures,
                       const std::map<SignatureEnumDType, std::pair<TypeId, bool>> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, const TupleInput &input_args,
                       std::index_sequence<N...>) const {
    MS_EXCEPTION_IF_NULL(op_run_info);
    return std::make_tuple(
      DoSignatureCast(op_run_info, signatures[N], dst_type, dtypes, N, std::get<N>(input_args))...);
  }

  template <typename Item>
  Item DoSignatureCast(const PyboostOpRunInfoPtr &op_run_info, const Signature &signature,
                       const std::map<SignatureEnumDType, std::pair<TypeId, bool>> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, size_t index, const Item &t) const {
    // No need to implicit cast if no dtype.
    if (dtypes.empty() || index >= dtypes.size() || dtypes[index] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      MS_LOG(DEBUG) << "Get kDTypeEmptyDefaultValue, or index " << index << " larger than dtype size " << dtypes.size();
      return t;
    }
    auto it = dst_type.find(dtypes[index]);
    if (it == dst_type.end() || it->second.first == kTypeUnknown) {
      MS_LOG(DEBUG) << "Can not find dtype " << (it == dst_type.end()) << ", or type is unknown "
                    << (it->second.first == kTypeUnknown);
      return t;
    }

    TypeId arg_type_id = kTypeUnknown;
    if (t->template isa<tensor::Tensor>()) {
      const auto &arg = t->template cast<tensor::TensorPtr>();
      arg_type_id = arg->data_type();
    }
    // Implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (arg_type_id == it->second.first);
    }
    if (signature.rw == SignatureEnumRW::kRWWrite && arg_type_id != kTypeUnknown && !is_same_type) {
      MS_EXCEPTION(TypeError) << prim::ErrorMessageForConvertRefDtype(op_run_info->op_prim, TypeIdToString(arg_type_id),
                                                                      TypeIdToString(it->second.first), index);
    }
    if (is_same_type) {
      MS_LOG(DEBUG) << "Get same dtype";
      return t;
    }

    if (IsValueTypeInvalid(t)) {
      std::string type_str = t->type() == nullptr ? "None, value is \"" + t->ToString() + "\"" : t->type()->ToString();
      MS_EXCEPTION(TypeError) << "For '" << op_run_info->op_prim->name() << "', the " << (index + 1) << "th input "
                              << signature.name << " can not be implicitly converted. "
                              << "Its type is " << type_str << ". Only support Tensor or Scalar.";
    }
    MS_LOG(DEBUG) << "Implicit cast for " << op_run_info->op_prim->name() << " " << index << "th input, from type "
                  << (t->type() == nullptr ? t->ToString() : t->type()->ToString()) << " to type "
                  << TypeIdToType(it->second.first)->ToString();
    // Has tensor input
    return DoAutoCast(op_run_info, it->second, index, t);
  }

  template <typename Item>
  std::optional<Item> DoSignatureCast(const PyboostOpRunInfoPtr &op_run_info, const Signature &signature,
                                      const std::map<SignatureEnumDType, std::pair<TypeId, bool>> &dst_type,
                                      const std::vector<SignatureEnumDType> &dtypes, size_t index,
                                      const std::optional<Item> &t) const {
    if (!t.has_value()) {
      return std::nullopt;
    }
    return std::make_optional(DoSignatureCast(op_run_info, signature, dst_type, dtypes, index, t.value()));
  }

  template <class Item>
  bool IsValueTypeInvalid(const Item &v) const {
    MS_EXCEPTION_IF_NULL(v);
    return !v->template isa<tensor::Tensor>() && !v->template isa<tensor::CSRTensor>() &&
           !v->template isa<IntegerImm>() && !v->template isa<FloatImm>() && !v->template isa<BoolImm>();
  }

  //  template <class Item, class = typename std::enable_if<std::is_same<Item, tensor::Tensor>::value, Item>::type>
  template <class Item>
  Item DoAutoCast(const PyboostOpRunInfoPtr &op_run_info, const std::pair<TypeId, bool> &dst_type, size_t index,
                  const Item &t) const {
    MS_EXCEPTION_IF_NULL(t);
    MS_LOG(DEBUG) << "Get input type " << typeid(t).name();
    ValuePtr v = t->template cast<ValuePtr>();
    auto ret = DoAutoCast(op_run_info, dst_type, index, v)->template cast<Item>();
    MS_EXCEPTION_IF_NULL(ret);
    return ret;
  }

  ValuePtr DoAutoCast(const PyboostOpRunInfoPtr &op_run_info, const std::pair<TypeId, bool> &dst_type, size_t index,
                      const ValuePtr &v) const;
  tensor::TensorPtr DoAutoCast(const PyboostOpRunInfoPtr &op_run_info, const std::pair<TypeId, bool> &dst_type,
                               size_t index, const tensor::TensorPtr &t) const;
  ValuePtr SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index) const;
  tensor::TensorPtr SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const tensor::TensorPtr &t,
                                              size_t index) const;
  std::optional<tensor::TensorPtr> SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info,
                                                             const std::optional<tensor::TensorPtr> &t,
                                                             size_t index) const;
  ValueTuplePtr SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const ValueTuplePtr &v_tuple,
                                          size_t index) const;
  ValueListPtr SetTensorMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const ValueListPtr &v_list,
                                         size_t index) const;
  ValuePtrList SetSeqMixPrecisionCast(const PyboostOpRunInfoPtr &op_run_info, const ValueSequencePtr &v_seq,
                                      size_t index) const;
};
using PyBoostCastOperationPtr = std::shared_ptr<PyBoostCastOperation>;

}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_PYBOOST_H_
