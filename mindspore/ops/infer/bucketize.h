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

#ifndef MINDSPORE_CORE_OPS_BUCKETIZE_H_
#define MINDSPORE_CORE_OPS_BUCKETIZE_H_
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
constexpr auto kNameBucketize = "Bucketize";
/// \brief Bucketizes 'input' based on 'boundaries'.
/// Refer to Python API @ref mindspore.ops.Bucketize for more details.
class OPS_API Bucketize : public BaseOperator {
 public:
  Bucketize() : BaseOperator(kNameBucketize) { InitIOName({"input"}, {"output"}); }
  MIND_API_BASE_MEMBER(Bucketize);
  void Init(const std::vector<float> &boundaries);
  void set_boundaries(const std::vector<float> &boundaries);
  std::vector<float> get_boundaries() const;
};

OPS_API abstract::AbstractBasePtr BucketizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_BUCKETIZE_H_
