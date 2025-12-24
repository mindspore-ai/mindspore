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

#ifndef MINDSPORE_CORE_OPS_QUANTILE_H_
#define MINDSPORE_CORE_OPS_QUANTILE_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameQuantile = "Quantile";
class OPS_API Quantile : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Quantile);
  Quantile() : BaseOperator(kNameQuantile) { InitIOName({"input", "q"}, {"out"}); }
  void Init() const {}
  void set_dim(int64_t dim);
  void set_keepdim(bool keepdim);
  void set_ignorenan(bool ignorenan);
  int64_t get_dim() const;
  bool get_keepdim() const;
  bool get_ignorenan() const;
};
abstract::AbstractBasePtr QuantileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimQuantilePtr = std::shared_ptr<Quantile>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANTILE_H_
