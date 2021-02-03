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

#ifndef MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
#define MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
#include <vector>
#include <memory>
#include <string>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameL2Normalize = "L2Normalize";
class L2Normalize : public PrimitiveC {
 public:
  explicit L2Normalize(const std::string &name = kNameL2Normalize) : PrimitiveC(name) {}
  ~L2Normalize() = default;
  MS_DECLARE_PARENT(L2Normalize, PrimitiveC);
  void Init(const std::vector<int64_t> &axis, const float epsilon = 1e-4);
  void set_axis(const std::vector<int64_t> &axis);
  void set_epsilon(const float epsilon);
  std::vector<int64_t> get_axis() const;
  float get_epsilon() const;
};
AbstractBasePtr L2NormalizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args);
using PrimL2NormalizePtr = std::shared_ptr<L2Normalize>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_L2_NORMALIZE_H_
