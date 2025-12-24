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

#ifndef MINDSPORE_CORE_OPS_PDIST_H_
#define MINDSPORE_CORE_OPS_PDIST_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePdist = "Pdist";
/// \brief Computes batched the p norm distance between each pair of row vectors in one collection.
/// Refer to Python API @ref mindspore.ops.Pdist for more details.
class OPS_API Pdist : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Pdist);
  /// \brief Constructor.
  Pdist() : BaseOperator(kNamePdist) { InitIOName({"x"}, {"y"}); }

  void set_p(const float p);
  /// \brief Get p.
  ///
  /// \return p.
  float get_p() const;
};

OPS_API abstract::AbstractBasePtr PdistInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PDIST_H_
