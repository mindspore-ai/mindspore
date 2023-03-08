/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_GENERATE_EOD_MASK_H
#define MINDSPORE_GENERATE_EOD_MASK_H
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGenerateEodMask = "GenerateEodMask";
/// \brief
class MIND_API GenerateEodMask : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(GenerateEodMask);
  /// \brief Constructor.
  GenerateEodMask() : BaseOperator(kNameGenerateEodMask) {
    InitIOName({"inputs_ids"}, {"position_ids", "attention_mask"});
  }
  /// \brief Init.
  void Init() const {}
  /// \brief Set axis.
  void set_eod_token_id(const int64_t eod_token_id);
  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_eod_token_id() const;
};

MIND_API abstract::AbstractBasePtr GenerateEodMaskInfer(const abstract::AnalysisEnginePtr &,
                                                        const PrimitivePtr &primitive,
                                                        const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_GENERATE_EOD_MASK_H
