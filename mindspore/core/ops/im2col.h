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

#ifndef MINDSPORE_CORE_OPS_IM2COL_H_
#define MINDSPORE_CORE_OPS_IM2COL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/base_operator.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameIm2Col = "Im2Col";
/// \brief Im2Col operation. Refer to Python API @ref mindspore.ops.Im2Col for more details.
class MIND_API Im2Col : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Im2Col);

  /// \brief Constructor.
  Im2Col() : BaseOperator(kNameIm2Col) { InitIOName({"x"}, {"y"}); }

  void set_ksizes(const std::vector<int64_t> &ksizes);

  std::vector<int64_t> get_ksizes() const;

  void set_strides(const std::vector<int64_t> &strides);

  std::vector<int64_t> get_strides() const;

  void set_dilations(const std::vector<int64_t> &dilations);

  std::vector<int64_t> get_dilations() const;

  void set_pads(const std::vector<int64_t> &pads);

  std::vector<int64_t> get_pads() const;
};

MIND_API abstract::AbstractBasePtr Im2ColInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args);
using PrimIm2ColPtr = std::shared_ptr<Im2Col>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_IM2COL_H_
