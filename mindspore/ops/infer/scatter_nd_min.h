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

#ifndef MINDSPORE_CORE_OPS_SCATTER_ND_MIN_H_
#define MINDSPORE_CORE_OPS_SCATTER_ND_MIN_H_

#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScatterNdMin = "ScatterNdMin";
class OPS_API ScatterNdMin : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScatterNdMin);
  ScatterNdMin() : BaseOperator(kNameScatterNdMin) { InitIOName({"input_x", "indices", "updates"}, {"y"}); }

  void Init(const bool use_locking = false);

  void set_use_locking(const bool use_locking);

  bool get_use_locking() const;
};

OPS_API abstract::AbstractBasePtr ScatterNdMinInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimScatterNdMinPtr = std::shared_ptr<ScatterNdMin>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCATTER_ND_ADD_H_
