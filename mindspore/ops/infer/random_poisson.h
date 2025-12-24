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

#ifndef MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
#define MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kRandomPoisson = "RandomPoisson";
class OPS_API RandomPoisson : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomPoisson);
  RandomPoisson() : BaseOperator(kRandomPoisson) { InitIOName({"shape", "rate"}, {"output"}); }
  void Init() const {}
  void set_seed(const int64_t seed);
  int64_t get_seed() const;
  void set_seed2(const int64_t seed2);
  int64_t get_seed2() const;
};
OPS_API abstract::AbstractBasePtr RandomPoissonInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimPrimRandomPoissonPtr = std::shared_ptr<RandomPoisson>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_POISSON_H_
