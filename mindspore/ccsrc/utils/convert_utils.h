/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_H_

#include <limits>
#include <memory>
#include <utility>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "pybind11/pybind11.h"
#include "utils/convert_utils_base.h"
#include "utils/any.h"
#include "utils/base_ref.h"
#include "ir/base.h"
#include "ir/anf.h"

namespace py = pybind11;

namespace mindspore {
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace tensor

py::object AnyToPyData(const Any &value);
py::object BaseRefToPyData(const BaseRef &value);
bool BaseRefToBool(const BaseRef &in, bool *out);
bool ValueToBool(const ValuePtr &in, bool *out);
py::object ValuePtrToPyData(const ValuePtr &value);

AbstractBasePtr PyListDtype2AbstractTensor(const py::object &shape_obj, const py::object &type_obj);

bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &output, const py::tuple &args,
                                       const std::shared_ptr<py::object> &ret_val);

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

using FuncGraphPairMapEquiv = std::unordered_map<std::pair<FuncGraphPtr, FuncGraphPtr>, EquivState, PairHasher>;
using NodeMapEquiv = std::unordered_map<AnfNodePtr, AnfNodePtr>;

bool Isomorphic(FuncGraphPtr g1, FuncGraphPtr g2, FuncGraphPairMapEquiv *equiv_func_graph, NodeMapEquiv *equiv_node);

tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_CONVERT_UTILS_H_
