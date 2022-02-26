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
#include "base/core_ops.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/visible.h"

namespace mindspore {
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
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

COMMON_EXPORT size_t CountValueNum(const ValueTuplePtr &value_tuple);

// sparse_attr_map converts CNode{kPrimSparseGetAttr, SparseTensor}
// to CNode{kPrimTupleGetItem, SparseTensor, int64_t(index)}, used
// in backend common optimization pass: sparse_process.cc
const mindspore::HashMap<std::string, int64_t> sparse_attr_map = {
  {prim::kCSRTensorGetIndptr, 0},     {prim::kCSRTensorGetIndices, 1}, {prim::kCSRTensorGetValues, 2},
  {prim::kCSRTensorGetDenseShape, 3}, {prim::kCOOTensorGetIndices, 0}, {prim::kCOOTensorGetValues, 1},
  {prim::kCOOTensorGetDenseShapes, 2}};
// make_sparse_set records all make_sparse primitives, and tries to replace
// make_sparse to make_tuple, used in backend common optimization pass:
// sparse_process.cc
const mindspore::HashSet<std::string> make_sparse_set = {{prim::kMakeCSRTensor}, {prim::kMakeCOOTensor}};
// sparse_op_set records all sparse_compute operators, which takes sparsetensor
// and (possibly) dense tensors, used in backend common optimization pass:
// sparse_process.cc
const mindspore::HashSet<std::string> sparse_op_set = {{prim::kSparseTensorDenseMatmul},
                                                       {prim::kCSRDenseMul},
                                                       {prim::kCSRReduceSum},
                                                       {prim::kCSRMV},
                                                       {prim::kCSRMul},
                                                       {prim::kCSRGather},
                                                       {prim::kCSR2COO}};

COMMON_EXPORT bool IsCustomCSROP(const AnfNodePtr &cnode);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONVERT_UTILS_H_
