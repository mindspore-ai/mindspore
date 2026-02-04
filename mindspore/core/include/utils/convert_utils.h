/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_
#define MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <stack>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <optional>
#include <algorithm>

#include "utils/hash_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/kernel_tensor_value.h"
#include "ir/signature.h"
#include "mindapi/base/macros.h"
#include "utils/simple_info.h"

namespace mindspore {
namespace tensor {
class Tensor;
}  // namespace tensor

MS_CORE_API bool BaseRefToBool(const BaseRef &in, bool *out);
MS_CORE_API bool BaseRefToInt(const ValuePtr &v, int64_t *value);
MS_CORE_API bool ValueToBool(const ValuePtr &in, bool *out);

// Isomorphism
struct PairHasher {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

enum EquivState { kNotEquiv = 0, kEquiv = 1, kPending = 2 };

using FuncGraphPairMapEquiv = mindspore::HashMap<std::pair<FuncGraphPtr, FuncGraphPtr>, EquivState, PairHasher>;
using NodeMapEquiv = mindspore::HashMap<AnfNodePtr, AnfNodePtr>;

MS_CORE_API bool Isomorphic(const FuncGraphPtr &g1, const FuncGraphPtr &g2, FuncGraphPairMapEquiv *equiv_func_graph,
                            NodeMapEquiv *equiv_node);

MS_CORE_API tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar,
                                             const std::optional<TypePtr> &dtype = std::nullopt);

MS_CORE_API tensor::TensorPtr SequenceToTensor(const ValueSequencePtr &sequence);

MS_CORE_API ValuePtr CreateValueFromTensor(const tensor::TensorPtr &tensor);

template <typename T>
std::vector<T> TensorValueToVector(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  std::vector<T> value;
  size_t element_size = tensor->DataSize();
  auto cpu_tensor = tensor->cpu();
  auto *data = static_cast<T *>(cpu_tensor->data_c());
  for (size_t i = 0; i < element_size; i++) {
    value.push_back(data[i]);
  }
  return value;
}

MS_CORE_API void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors);

MS_CORE_API size_t CountValueNum(const ValueSequencePtr &value_sequence);

MS_CORE_API KernelTensorValuePtr ConvertValueToKernelTensorValue(const ValuePtr &value);

MS_CORE_API tensor::MetaSparseTensorPtr TensorListToSparseTensor(const abstract::AbstractBasePtr &abs_sparse,
                                                                 const tensor::TensorPtrList &tensor_list);
// Convert base shape to shape vector, support the tuple shape.
MS_CORE_API std::vector<ShapeVector> BaseShapeToShapeVector(const abstract::BaseShapePtr &base_shape);
// Convert base shape to shape, not support the tuple shape.
MS_CORE_API ShapeVector BaseShapeToShape(const abstract::BaseShapePtr &base_shape);

MS_CORE_API ValuePtr UpdateValueByAttrDataType(const ValuePtr &value, const std::string &attr_data_type);

MS_CORE_API std::map<SignatureEnumDType, std::pair<TypeId, bool>> GetSignatureTypeMap(
  const std::vector<SignatureEnumDType> &dtypes, const std::vector<TypeId> &args_type_id,
  const std::vector<bool> &args_is_tensor, const std::set<size_t> &write_indices = {});

MS_CORE_API TypeId ConvertTypeForTensorsOrScalars(const TypeId &type1, const TypeId &type2);

MS_CORE_API bool IsFloatTensor(const TypeId &type_id, bool is_tensor);

MS_CORE_API TypeId GetMixPrecisionPromoteType(const std::vector<TypeId> &args_type_id,
                                              const std::vector<bool> &args_is_tensor);

MS_CORE_API std::string ValueSimpleInfoToString(const ValueSimpleInfo &value_simple_info);

MS_CORE_API abstract::AbstractBasePtr TransformValueSimpleInfoToAbstract(const ValueSimpleInfo &value_simple_info);

template <typename T>
ValuePtr OptionalToValue(const std::optional<T> &val) {
  if (!val.has_value()) {
    return kNone;
  }
  return val.value();
}

template <typename T>
using imm_element_t = typename ImmTraits<T>::type::element_type;

template <typename T, typename = void>
struct has_imm_element : std::false_type {};

template <typename T>
struct has_imm_element<T, std::void_t<imm_element_t<T>>> : std::true_type {};

template <typename T>
inline constexpr bool has_imm_element_v = has_imm_element<T>::value;

template <typename T>
std::enable_if_t<has_imm_element_v<T>, std::shared_ptr<imm_element_t<T>>> PackToValue(const T &val) {
  return std::make_shared<imm_element_t<T>>(val);
}

template <typename T>
ValueTuplePtr PackToValue(const std::vector<T> &vals) {
  std::vector<ValuePtr> values;
  values.reserve(vals.size());
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(values),
                       [](const T &val) { return PackToValue(val); });
  return std::make_shared<ValueTuple>(values);
}

template <typename T>
using PackedType = decltype(PackToValue(std::declval<const T &>()));

template <typename T>
std::optional<PackedType<T>> PackToValue(const std::optional<T> &val) {
  if (!val) {
    return std::nullopt;
  }
  return PackToValue(*val);
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_
