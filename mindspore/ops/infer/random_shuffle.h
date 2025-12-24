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

#ifndef MINDSPORE_CORE_OPS_RANDOM_SHUFFLE_H_
#define MINDSPORE_CORE_OPS_RANDOM_SHUFFLE_H_

#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
namespace mindspore {
namespace ops {
constexpr auto kRandomShuffle = "RandomShuffle";
/// \brief Randomly shuffles a Tensor along its first dimension.
/// Refer to Python API @ref mindspore.ops.RandomShuffle for more details.
class OPS_API RandomShuffle : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(RandomShuffle);
  /// \brief Constructor.
  RandomShuffle() : BaseOperator(kRandomShuffle) { InitIOName({"input_x"}, {"output"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.RandomShuffle for the inputs.
  void Init(const int64_t seed = 0, const int64_t seed2 = 0);

  /// \brief Set seed. Defaults to 0.
  void set_seed(const int64_t seed);
  /// \brief Get seed.
  ///
  /// \return seed.
  int64_t get_seed() const;

  /// \brief Set seed2. Defaults to 0.
  void set_seed2(const int64_t seed2);
  /// \brief Get seed2.
  ///
  /// \return seed2.
  int64_t get_seed2() const;
};

OPS_API abstract::AbstractBasePtr RandomShuffleInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimRandomShufflePtr = std::shared_ptr<RandomShuffle>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_SHUFFLE_H_
