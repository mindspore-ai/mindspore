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

#ifndef MINDSPORE_CORE_OPS_NAN_TO_NUM_H_
#define MINDSPORE_CORE_OPS_NAN_TO_NUM_H_
#include <vector>
#include <limits>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNanToNum = "NanToNum";

class MIND_API NanToNum : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(NanToNum);
  NanToNum() : BaseOperator(kNameNanToNum) { InitIOName({"x"}, {"y"}); }
  void Init(float nan, float posinf, float neginf);
  void set_nan_value(float nan_value);
  float get_nan_value() const;
  void set_posinf_value(float nan_value);
  float get_posinf_value() const;
  void set_neginf_value(float nan_value);
  float get_neginf_value() const;
};
abstract::AbstractBasePtr NanToNumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimNanToNumPtr = std::shared_ptr<NanToNum>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NAN_TO_NUM_H_
