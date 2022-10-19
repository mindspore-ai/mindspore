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

#ifndef MINDSPORE_CORE_OPS_EXTRACT_GLIMPSE_H_
#define MINDSPORE_CORE_OPS_EXTRACT_GLIMPSE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameExtractGlimpse = "ExtractGlimpse";
class MIND_API ExtractGlimpse : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ExtractGlimpse);
  ExtractGlimpse() : BaseOperator(kNameExtractGlimpse) { InitIOName({"input", "size", "offsets"}, {"output"}); }

  void Init(const bool &centered, const bool &normalized, const bool &uniform_noise, const std::string &noise);

  /// \brief Set centered.
  void set_centered(const bool &centered);

  /// \brief Set normalized.
  void set_normalized(const bool &normalized);

  /// \brief Set uniform_noise.
  void set_uniform_noise(const bool &uniform_noise);

  /// \brief Set noise.
  void set_noise(const std::string &noise);

  /// \brief Get centered.
  ///
  /// \return centered.
  bool get_centered() const;

  /// \brief Get normalized.
  ///
  /// \return normalized.
  bool get_normalized() const;

  /// \brief Get uniform_noise.
  ///
  /// \return uniform_noise.
  bool get_uniform_noise() const;

  /// \brief Get noise.
  ///
  /// \return noise.
  std::string get_noise() const;
};
abstract::AbstractBasePtr ExtractGlimpseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args);
using PrimExtractGlimpsePtr = std::shared_ptr<ExtractGlimpse>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_EXTRACT_GLIMPSE_H_
