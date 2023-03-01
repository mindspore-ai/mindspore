/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_

#include <limits>
#include <memory>
#include <utility>
#include <stack>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/convert_utils_base.h"
#include "utils/any.h"
#include "base/base_ref.h"
#include "mindspore/core/ops/core_ops.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/visible.h"

namespace mindspore {
namespace tensor {
class Tensor;
}  // namespace tensor

COMMON_EXPORT bool BaseRefToBool(const BaseRef &in, bool *out);
COMMON_EXPORT bool BaseRefToInt(const ValuePtr &v, int64_t *value);
COMMON_EXPORT bool ValueToBool(const ValuePtr &in, bool *out);

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

COMMON_EXPORT bool Isomorphic(const FuncGraphPtr &g1, const FuncGraphPtr &g2, FuncGraphPairMapEquiv *equiv_func_graph,
                              NodeMapEquiv *equiv_node);

COMMON_EXPORT tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar);

COMMON_EXPORT ValuePtr CreateValueFromTensor(const tensor::TensorPtr &tensor);

template <typename T>
std::vector<T> TensorValueToVector(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  std::vector<T> value;
  auto element_size = tensor->data().size();
  auto *data = static_cast<T *>(tensor->data_c());
  for (auto i = 0; i < element_size; i++) {
    value.push_back(data[i]);
  }
  return value;
}

COMMON_EXPORT void TensorValueToTensor(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors);

COMMON_EXPORT ValuePtr ShallowCopyTensorValue(const ValuePtr &value);

COMMON_EXPORT size_t CountValueNum(const ValueSequencePtr &value_sequence);

COMMON_EXPORT bool IsAKGSparseOP(const AnfNodePtr &cnode);

COMMON_EXPORT tensor::MetaSparseTensorPtr TensorListToSparseTensor(const abstract::AbstractBasePtr &abs_sparse,
                                                                   const tensor::TensorPtrList &tensor_list);
// Convert base shape to shape vector, support the tuple shape.
COMMON_EXPORT std::vector<ShapeVector> BaseShapeToShapeVector(const abstract::BaseShapePtr &base_shape);
// Convert base shape to shape, not support the tuple shape.
COMMON_EXPORT ShapeVector BaseShapeToShape(const abstract::BaseShapePtr &base_shape);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_
