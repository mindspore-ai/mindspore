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
 * limitations under the License
 */

#ifndef MINDSPORE_CORE_OPS_SOLVE_TRIANGULAR_H_
#define MINDSPORE_CORE_OPS_SOLVE_TRIANGULAR_H_

#include <map>
#include <set>
#include <string>
#include <memory>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSolveTriangular = "SolveTriangular";
/// \brief Assert defined MatrixSolve operator prototype.
class MIND_API SolveTriangular : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SolveTriangular);
  /// \brief Constructor.
  SolveTriangular() : BaseOperator(kNameSolveTriangular) { InitIOName({"a", "b"}, {"output"}); }
  /// \brief Init.
  void Init(bool lower, bool unit_diagonal, std::string trans);

  /// \brief Method to set unit_diagonal attributes.
  void set_unit_diagonal(bool unit_diagonal);
  /// \brief Method to get unit_diagonal attributes.
  bool get_unit_diagonal() const;

  /// \brief Method to set lower attributes.
  void set_lower(bool lower);
  /// \brief Method to get lower attributes.
  bool get_lower() const;

  /// \brief Method to set trans attributes.
  void set_trans(std::string trans);
  /// \brief Method to get trans attributes.
  std::string get_trans() const;
};
abstract::AbstractBasePtr SolveTriangularInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimSolveTriangularPtr = std::shared_ptr<SolveTriangular>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SOLVE_TRIANGULAR_H_
