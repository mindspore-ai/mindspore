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

#ifndef MINDSPORE_CORE_OPS_ANGLE_ATOM_ENERGY_H_
#define MINDSPORE_CORE_OPS_ANGLE_ATOM_ENERGY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAngleAtomEnergy = "AngleAtomEnergy";
/// \brief AngleAtomEnergy operation. Refer to Python API @ref mindspore.ops.AngleAtomEnergy for more details.
class OPS_API AngleAtomEnergy : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AngleAtomEnergy);
  /// \brief Constructor.
  AngleAtomEnergy() : BaseOperator(kNameAngleAtomEnergy) {
    InitIOName({"uint_crd_f", "scaler_f", "atom_a", "atom_b", "atom_c", "angle_k", "angle_theta0"}, {"ene"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.AngleAtomEnergy for the inputs.
  void Init(const int64_t angle_numbers);
  /// \brief Set angle_numbers.
  void set_angle_numbers(const int64_t angle_numbers);
  /// \brief Get angle_numbers.
  ///
  /// \return angle_numbers.
  int64_t get_angle_numbers() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ANGLE_ATOM_ENERGY_H_
