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

#ifndef MINDSPORE_CORE_OPS_FILL_V2_H_
#define MINDSPORE_CORE_OPS_FILL_V2_H_

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFillV2 = "FillV2";
class MS_CORE_API FillV2 : public PrimitiveC {
 public:
  FillV2() : PrimitiveC(kNameFillV2) { InitIOName({"shape", "value"}, {"y"}); }
  ~FillV2() = default;
  MS_DECLARE_PARENT(FillV2, PrimitiveC);
};

AbstractBasePtr FillV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);

using PrimFillV2Ptr = std::shared_ptr<FillV2>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FILL_V2_H_
