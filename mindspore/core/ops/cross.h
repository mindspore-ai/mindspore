/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_CROSS_H_
#define MINDSPORE_CORE_OPS_CROSS_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCross = "Cross";

class MIND_API Cross : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Cross);
  Cross() : BaseOperator(kNameCross) { InitIOName({"x1", "x2"}, {"y"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define a dim which is the first dimension to slice.
  /// \param[in] offsets Define a vector to indicate the start index to slice on the corresponding axis.
  void Init(const int64_t dim);

  /// \brief Method to set dim attribute.
  ///
  /// \param[in] dim Define a dim.
  void set_dim(const int64_t dim);

  /// \brief Method to get dim attribute.
  ///
  /// \return dim.
  int64_t get_dim() const;
};

abstract::AbstractBasePtr CrossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimCrossPtr = std::shared_ptr<Cross>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CROSS_H_
