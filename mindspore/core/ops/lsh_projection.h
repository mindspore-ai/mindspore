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

#ifndef MINDSPORE_CORE_OPS_LSH_PROJECTION_H_
#define MINDSPORE_CORE_OPS_LSH_PROJECTION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLshProjection = "LshProjection";
/// \brief LshProjection defined LshProjection operator prototype of lite, which is to project an input to a bit vector.
class MS_CORE_API LshProjection : public PrimitiveC {
 public:
  /// \brief Constructor.
  LshProjection() : PrimitiveC(kNameLshProjection) {}

  /// \brief Destructor.
  ~LshProjection() = default;

  MS_DECLARE_PARENT(LshProjection, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] type Define the type of the output.
  void Init(const LshProjectionType &type);

  /// \brief Method to set type attribute.
  ///
  /// \param[in] type Define the type of the output.
  void set_type(const LshProjectionType &type);

  /// \brief Method to get type attribute.
  ///
  /// \return the type of the output.
  LshProjectionType get_type() const;
};

AbstractBasePtr LshProjectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args);
using PrimLshProjectionPtr = std::shared_ptr<LshProjection>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LSH_PROJECTION_H_
