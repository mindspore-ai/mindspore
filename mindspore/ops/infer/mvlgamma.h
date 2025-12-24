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

#ifndef MINDSPORE_CORE_OPS_MVLGAMMA_H_
#define MINDSPORE_CORE_OPS_MVLGAMMA_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMvlgamma = "Mvlgamma";
/// \brief Computes the multivariate log-gamma function with dimension p element-wise.
/// Refer to Python API @ref mindspore.ops.Mvlgamma for more details.
class OPS_API Mvlgamma : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Mvlgamma);
  /// \brief Constructor.
  Mvlgamma() : BaseOperator(kNameMvlgamma) { InitIOName({"x"}, {"y"}); }
  /// \brief Init.
  void Init(const int64_t p = 0);
  /// \brief Set p.
  void set_p(const int64_t p);
  int64_t get_p() const;
};

OPS_API abstract::AbstractBasePtr MvlgammaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args);
using PrimMvlgammaPtr = std::shared_ptr<Mvlgamma>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_MVLGAMMA_H_
