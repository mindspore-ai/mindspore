/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LP_NORM_H_
#define MINDSPORE_CORE_OPS_LP_NORM_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLpNorm = "LpNorm";
class OPS_API LpNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LpNorm);
  LpNorm() : BaseOperator(kNameLpNorm) { InitIOName({"input"}, {"output"}); }

  void Init(const std::vector<int64_t> &axis, const int64_t p = 2, const bool keep_dims = false,
            const float epsilon = 1e-12);

  void set_axis(const std::vector<int64_t> &axis);

  void set_keep_dims(const bool keep_dims);

  void set_p(const int64_t p);

  void set_epsilon(const float epsilon);

  std::vector<int64_t> get_axis() const;

  int64_t get_p() const;

  bool get_keep_dims() const;

  float get_epsilon() const;
};

OPS_API abstract::AbstractBasePtr LpNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args);

using PrimLpNormPtr = std::shared_ptr<LpNorm>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LP_NORM_H_
