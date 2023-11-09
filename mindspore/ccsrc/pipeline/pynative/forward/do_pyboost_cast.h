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
#include <utility>
#include "pipeline/pynative/forward/cast_base.h"
#include "frontend/operator/composite/do_signature.h"
#include "ir/cell.h"

namespace mindspore {
namespace pynative {
static constexpr auto kCast = "Cast";

class PyBoostCastOperation : public CastBaseOperation {
 public:
  PyBoostCastOperation() = default;
  ~PyBoostCastOperation() = default;

  template <typename... InputArgs>
  auto DoMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, const InputArgs &...input_args) {
    // Mixed precision conversion tensors which has cast dtype
    if (op_run_info->async_status.disable_mix_precision) {
      return std::make_tuple(input_args...);
    }
    size_t index = 0;
    auto increase_index_fn = [&index]() { return index++; };
    auto ret = std::make_tuple(SetTensorMixPrecisionCast(op_run_info, input_args, increase_index_fn())...);
    return ret;
  }

  template <typename T>
  T SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, const T &t, size_t index) {
    MS_EXCEPTION_IF_NULL(t);
    return t;
  }

  // Implicit transform
  template <typename... InputArgs>
  auto DoImplicitCast(const FrontendOpRunInfoPtr &op_run_info, const std::tuple<InputArgs...> &input_args) const {
    MS_EXCEPTION_IF_NULL(op_run_info);
    mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> type_indexes;
    std::vector<SignatureEnumDType> dtypes;
    const auto &it = implicit_cast_map_.find(op_run_info->base_op_run_info.op_name);
    if (it == implicit_cast_map_.end()) {
      bool has_dtype_sig = GetSignatureType(op_run_info->signatures, &dtypes);
      if (!has_dtype_sig) {
        PrimSignature sig_value{has_dtype_sig, {}, {}};
        implicit_cast_map_[op_run_info->base_op_run_info.op_name] = sig_value;
        return input_args;
      }
      auto sig_size = op_run_info->signatures.size();
      if (sig_size > 0 && sig_size != sizeof...(InputArgs)) {
        MS_EXCEPTION(ValueError) << op_run_info->base_op_run_info.op_name << " inputs size " << sizeof...(InputArgs)
                                 << " does not match the requires "
                                 << "signature size " << sig_size;
      }
      GetTypeIndex(dtypes, &type_indexes);
      PrimSignature sig_value{has_dtype_sig, dtypes, type_indexes};
      implicit_cast_map_[op_run_info->base_op_run_info.op_name] = sig_value;
    } else {
      type_indexes = it->second.type_indexes;
      dtypes = it->second.dtypes;
    }

    return SetImplicitCast(op_run_info, type_indexes, dtypes, input_args,
                           std::make_index_sequence<sizeof...(InputArgs)>{});
  }

 private:
  template <typename Tuple, size_t... N>
  auto SetImplicitCast(const FrontendOpRunInfoPtr &op_run_info,
                       const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                       const std::vector<SignatureEnumDType> &dtypes, const Tuple &input_args,
                       std::index_sequence<N...>) const {
    MS_EXCEPTION_IF_NULL(op_run_info);
    mindspore::HashMap<SignatureEnumDType, TypeId> dst_type;
    (GetDstType(op_run_info, type_indexes, &dst_type, std::get<N>(input_args)), ...);
    return std::make_tuple(DoSignatureCast(op_run_info, dst_type, dtypes, N, std::get<N>(input_args))...);
  }

  template <typename Item>
  void GetDstType(const FrontendOpRunInfoPtr &op_run_info,
                  const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                  mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type, const Item &v) const {
    constexpr size_t index_size = 2;
    for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
      const auto &type = it->first;
      const auto &indexes = it->second;
      if (type == SignatureEnumDType::kDTypeEmptyDefaultValue || indexes.size() < index_size) {
        continue;
      }
      size_t priority = 0;
      TypeId max_type = TypeId::kTypeUnknown;
      bool has_scalar_float32 = false;
      bool has_scalar_int64 = false;
      bool has_tensor_int8 = false;
      // Find the maximum priority of the same dtype
      for (size_t index : indexes) {
        if (index >= op_run_info->input_size) {
          MS_LOG(EXCEPTION) << "The index " << index << " exceeds the size of py_args " << op_run_info->input_size;
        }
        if (v->template isa<FloatImm>()) {
          has_scalar_float32 = true;
        }
        if (!v->template isa<BoolImm>() && v->template isa<IntegerImm>()) {
          has_scalar_int64 = true;
        }
        if (v->template isa<tensor::Tensor>()) {
          auto arg = v->template cast<tensor::TensorPtr>();
          TypeId arg_type_id = arg->data_type();
          auto type_priority = prim::type_map.find(arg_type_id);
          if (type_priority == prim::type_map.end()) {
            continue;
          }
          if (arg_type_id == kNumberTypeInt8) {
            has_tensor_int8 = true;
          }
          if (type_priority->second > priority) {
            max_type = type_priority->first;
            priority = type_priority->second;
          }
        }
      }
      max_type = JudgeMaxType(max_type, has_scalar_float32, has_scalar_int64, has_tensor_int8);
      MS_EXCEPTION_IF_NULL(dst_type);
      (void)dst_type->emplace(std::make_pair(type, max_type));
    }
  }

  template <typename Item>
  void GetDstType(const FrontendOpRunInfoPtr &op_run_info,
                  const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                  mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type, const std::optional<Item> &v) const {
    if (!v.has_value()) {
      return;
    }
    GetDstType(op_run_info, type_indexes, dst_type, v.value());
  }

  template <typename Item>
  Item DoSignatureCast(const FrontendOpRunInfoPtr &op_run_info,
                       const mindspore::HashMap<SignatureEnumDType, TypeId> &dst_type,
                       const std::vector<SignatureEnumDType> &dtypes, size_t index, const Item &t) const {
    // No need to implicit cast if no dtype.
    const auto &signature = op_run_info->signatures;
    if (dtypes.empty() || dtypes[index] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      return t;
    }
    auto it = dst_type.find(dtypes[index]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      return t;
    }

    TypeId arg_type_id = kTypeUnknown;
    if (t->template isa<tensor::MetaTensor>()) {
      const auto &arg = t->template cast<tensor::MetaTensorPtr>();
      arg_type_id = arg->data_type();
    }
    // Implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (prim::type_map.find(arg_type_id) == prim::type_map.end() || arg_type_id == it->second);
    }
    if (signature[index].rw == SignatureEnumRW::kRWWrite && arg_type_id != kTypeUnknown && !is_same_type) {
      prim::RaiseExceptionForConvertRefDtype(op_run_info->op_grad_info->op_prim, TypeIdToMsTypeStr(arg_type_id),
                                             TypeIdToMsTypeStr(it->second), index);
    }
    if (is_same_type) {
      return t;
    }

    if (IsValueTypeInvalid(t)) {
      std::string type_str = t->type() == nullptr ? "None, value is \"" + t->ToString() + "\"" : t->type()->ToString();
      MS_EXCEPTION(TypeError) << "For '" << op_run_info->op_grad_info->op_prim->name() << "', the " << (index + 1)
                              << "th input " << signature[index].name << " can not be implicitly converted. "
                              << "Its type is " << type_str << ". Only support Tensor or Scalar.";
    }
    MS_LOG(DEBUG) << "Implicit cast for " << op_run_info->base_op_run_info.op_name << " " << index
                  << "th input, and to type " << TypeIdToType(it->second)->ToString();
    return DoAutoCast(op_run_info, it->second, t);
  }

  template <typename Item>
  std::optional<Item> DoSignatureCast(const FrontendOpRunInfoPtr &op_run_info,
                                      const mindspore::HashMap<SignatureEnumDType, TypeId> &dst_type,
                                      const std::vector<SignatureEnumDType> &dtypes, size_t index,
                                      const std::optional<Item> &t) const {
    if (!t.has_value()) {
      return std::nullopt;
    }
    return std::make_optional(DoSignatureCast(op_run_info, dst_type, dtypes, index, t.value()));
  }

  template <class Item>
  bool IsValueTypeInvalid(const Item &v) const {
    MS_EXCEPTION_IF_NULL(v);
    return !v->template isa<tensor::Tensor>() && !v->template isa<tensor::CSRTensor>() &&
           !v->template isa<IntegerImm>() && !v->template isa<FloatImm>() && !v->template isa<BoolImm>();
  }

  //  template <class Item, class = typename std::enable_if<std::is_same<Item, tensor::Tensor>::value, Item>::type>
  template <class Item>
  Item DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const TypeId &type_id, const Item &t) const {
    return t;
  }

  tensor::TensorPtr DoAutoCast(const FrontendOpRunInfoPtr &op_run_info, const TypeId &type_id,
                               const tensor::TensorPtr &t) const;
  tensor::TensorPtr SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, const tensor::TensorPtr &t,
                                              size_t index);
  std::optional<tensor::TensorPtr> SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info,
                                                             const std::optional<tensor::TensorPtr> &t, size_t index);
  ValuePtr SetTensorMixPrecisionCast(const FrontendOpRunInfoPtr &op_run_info, const ValueSequencePtr &v_seq,
                                     size_t index);
};
using PyBoostCastOperationPtr = std::shared_ptr<PyBoostCastOperation>;

}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_DO_CAST_PYBOOST_H_
