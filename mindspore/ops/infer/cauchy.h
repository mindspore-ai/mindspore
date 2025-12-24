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

#ifndef MINDSPORE_CORE_OPS_CAUCHY_H_
#define MINDSPORE_CORE_OPS_CAUCHY_H_
#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCauchy = "Cauchy";
class OPS_API Cauchy : public BaseOperator {
 public:
  Cauchy() : BaseOperator(kNameCauchy) {}
  MIND_API_BASE_MEMBER(Cauchy);
  void Init() const {}

  void set_sigma(const float);
  float get_sigma() const;
  void set_median(const float);
  float get_median() const;
  void set_size(const std::vector<int64_t>);
  std::vector<int64_t> get_size() const;
};
using PrimCauchy = std::shared_ptr<Cauchy>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_Cauchy_H_
