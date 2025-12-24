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

#ifndef MINDSPORE_CORE_OPS_CONSTANT_OF_SHAPE_H_
#define MINDSPORE_CORE_OPS_CONSTANT_OF_SHAPE_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConstantOfShape = "ConstantOfShape";
/// \brief ConstantOfShape defined ConstantOfShape operator prototype of lite.
class OPS_API ConstantOfShape : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ConstantOfShape);
  /// \brief Constructor.
  ConstantOfShape() : BaseOperator(kNameConstantOfShape) { InitIOName({"shape"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] data_type Define data type of output data.
  /// \param[in] value Define value of output data.
  void Init(int64_t data_type, const std::vector<float> &value);

  /// \brief Method to set data type attribute.
  ///
  /// \param[in] data_type Define data type of output data.
  void set_data_type(int64_t data_type);

  /// \brief Method to set value attribute.
  ///
  /// \param[in] value Define value of output data.
  void set_value(const std::vector<float> &value);

  /// \brief Method to get data type attribute.
  ///
  /// \return data type attribute.
  int64_t get_data_type() const;

  /// \brief Method to get value attribute.
  ///
  /// \return value attribute.
  std::vector<float> get_value() const;
};

OPS_API abstract::AbstractBasePtr ConstantOfShapeInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONSTANT_OF_SHAPE_H_
