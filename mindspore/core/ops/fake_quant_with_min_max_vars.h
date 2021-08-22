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
#ifndef MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
#define MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFakeQuantWithMinMaxVars = "FakeQuantWithMinMaxVars";
class FakeQuantWithMinMaxVars : public PrimitiveC {
 public:
  FakeQuantWithMinMaxVars() : PrimitiveC(kNameFakeQuantWithMinMaxVars) {}
  ~FakeQuantWithMinMaxVars() = default;
  MS_DECLARE_PARENT(FakeQuantWithMinMaxVars, PrimitiveC);
  void Init(const bool narrow_range = false, const int64_t num_bits = 8);
  void set_narrow_range(const bool narrow_range);
  void set_num_bits(const int64_t num_bits);
  bool get_narrow_range() const;
  int64_t get_num_bits() const;
};
AbstractBasePtr FakeQuantWithMinMaxVarsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args);
using PrimFakeQuantWithMinMaxVarsPtr = std::shared_ptr<FakeQuantWithMinMaxVars>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FAKE_QUANT_WITH_MIN_MAX_VARS_H_
