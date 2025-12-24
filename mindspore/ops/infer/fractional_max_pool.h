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

#ifndef MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_H_
#define MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_H_
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFractionalMaxPool = "FractionalMaxPool";
class OPS_API FractionalMaxPool : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FractionalMaxPool);
  FractionalMaxPool() : BaseOperator(kNameFractionalMaxPool) {
    InitIOName({"x"}, {"y", "row_pooling_sequence", "col_pooling_sequence"});
  }
  void Init(const std::vector<float> pooling_ratio, const bool pseudo_random = false, const bool overlapping = false,
            const bool deterministic = false, const int64_t seed = 0, const int64_t seed2 = 0);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.FractionalMaxPool for the inputs.
  void set_pooling_ratio(const std::vector<float> pooling_ratio);
  /// \brief Set pooling ratio.
  void set_pseudo_random(const bool pseudo_random);
  /// \brief Set pseudo random.
  void set_overlapping(const bool overlapping);
  /// \brief Set overlapping.
  void set_deterministic(const bool deterministic);
  /// \brief Set deterministic.
  void set_seed(const int64_t seed);
  /// \brief Set seed.
  void set_seed2(const int64_t seed2);
  /// \brief Set seed2.
  std::vector<float> get_pooling_ratio() const;
  /// \brief Method to get pooling ratio attributes.
  ///
  /// \return pooling ratio attributes.
  bool get_pseudo_random() const;
  /// \brief Method to get pseudo random attributes.
  ///
  /// \return pseudo random attributes.
  bool get_overlapping() const;
  /// \brief Method to get overlapping attributes.
  ///
  /// \return overlapping attributes.
  bool get_deterministic() const;
  /// \brief Method to get deterministic attributes.
  ///
  /// \return deterministic attributes.
  int64_t get_seed() const;
  /// \brief Method to get seed attributes.
  ///
  /// \return seed attributes.
  int64_t get_seed2() const;
  /// \brief Method to get seed2 attributes.
  ///
  /// \return pooling seed2 attributes.
};
OPS_API abstract::AbstractBasePtr FractionalMaxPoolInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimFractionalMaxPool = std::shared_ptr<FractionalMaxPool>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FRACTIONAL_MAX_POOL_H_
