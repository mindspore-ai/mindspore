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

#ifndef MINDSPORE_CORE_OPS_RANGE_H_
#define MINDSPORE_CORE_OPS_RANGE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRange = "Range";
/// \brief Creates a sequence of numbers in range [start, limit) with step size delta.
/// Refer to Python API @ref mindspore.nn.Range for more details.
class MS_CORE_API Range : public PrimitiveC {
 public:
  /// \brief Constructor.
  Range() : PrimitiveC(kNameRange) {}
  /// \brief Destructor.
  ~Range() = default;
  MS_DECLARE_PARENT(Range, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.nn.Range for the inputs.
  void Init(const int64_t d_type, const int64_t start, const int64_t limit, const int64_t delta);
  /// \brief Set d_type.
  void set_d_type(const int64_t d_type);
  /// \brief Set start.
  void set_start(const int64_t start);
  /// \brief Set limit.
  void set_limit(const int64_t limit);
  /// \brief Set delta.
  void set_delta(const int64_t delta);
  /// \brief Get d_type.
  ///
  /// \return d_type.
  int64_t get_d_type() const;
  /// \brief Get start.
  ///
  /// \return start.
  int64_t get_start() const;
  /// \brief Get limit.
  ///
  /// \return limit.
  int64_t get_limit() const;
  /// \brief Get delta.
  ///
  /// \return delta.
  int64_t get_delta() const;
};

AbstractBasePtr RangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
using PrimRangePtr = std::shared_ptr<Range>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANGE_H_
