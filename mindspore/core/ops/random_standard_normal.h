/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
#define MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRandomStandardNormal = "RandomStandardNormal";
class RandomStandardNormal : public PrimitiveC {
 public:
  RandomStandardNormal() : PrimitiveC(kNameRandomStandardNormal) {}
  ~RandomStandardNormal() = default;
  MS_DECLARE_PARENT(RandomStandardNormal, PrimitiveC);
  void Init(const int64_t seed, const int64_t seed2);

  void set_seed(const int64_t seed);
  void set_seed2(const int64_t seed2);

  bool get_seed() const;
  bool get_seed2() const;
};

AbstractBasePtr RandomStandardNormalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args);
using PrimRandomStandardNormalPtr = std::shared_ptr<RandomStandardNormal>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_STANDARD_NORMAL_H_
