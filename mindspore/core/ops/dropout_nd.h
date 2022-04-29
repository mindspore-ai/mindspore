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

#ifndef MINDSPORE_CORE_OPS_DROPOUTND_H_
#define MINDSPORE_CORE_OPS_DROPOUTND_H_
#include <vector>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace ops {
class MIND_API Dropout2D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Dropout2D);
  /// \brief Constructor.
  Dropout2D() : BaseOperator(prim::kDropout2D) { InitIOName({"input_x"}, {"output"}); }

  void Init(float keep_prob = 0.0);

  void set_keep_prob(float keep_prob);

  float get_keep_prob() const;
};

class MIND_API Dropout3D : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Dropout3D);
  /// \brief Constructor.
  Dropout3D() : BaseOperator(prim::kDropout3D) { InitIOName({"input_x"}, {"output"}); }

  void Init(float keep_prob = 0.0);

  void set_keep_prob(float keep_prob);

  float get_keep_prob() const;
};

abstract::AbstractBasePtr Dropout2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);

abstract::AbstractBasePtr Dropout3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_DROPOUTND_H_
