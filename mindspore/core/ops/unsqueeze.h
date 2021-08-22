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

#ifndef MINDSPORE_CORE_OPS_UNSQUEEZE_H_
#define MINDSPORE_CORE_OPS_UNSQUEEZE_H_

#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUnsqueeze = "Unsqueeze";
class Unsqueeze : public PrimitiveC {
 public:
  Unsqueeze() : PrimitiveC(kNameUnsqueeze) {}
  ~Unsqueeze() = default;
  MS_DECLARE_PARENT(Unsqueeze, PrimitiveC);
  void Init(const std::vector<int64_t> axis);
  void set_axis(const std::vector<int64_t> axis);
  std::vector<int64_t> get_axis() const;
};
AbstractBasePtr UnsqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
using PrimUnsqueezePtr = std::shared_ptr<Unsqueeze>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_UNSQUEEZE_H_
