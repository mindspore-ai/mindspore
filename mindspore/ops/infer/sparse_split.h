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

#ifndef MINDSPORE_CORE_OPS_SPARSE_SPLIT_H_
#define MINDSPORE_CORE_OPS_SPARSE_SPLIT_H_
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseSplit = "SparseSplit";
class OPS_API SparseSplit : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseSplit);
  SparseSplit() : BaseOperator(kNameSparseSplit) {
    InitIOName({"split_dim", "indices", "values", "shape"}, {"y_indices", "y_values", "y_shape"});
  }
  auto get_num_split() const {
    auto num_splits = GetValue<int64_t>(GetAttr("num_split"));
    return num_splits;
  }
};
OPS_API abstract::AbstractBasePtr SparseSplitInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_SPLIT_H_
