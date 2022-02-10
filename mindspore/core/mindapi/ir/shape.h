/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_IR_SHAPE_H_
#define MINDSPORE_CORE_MINDAPI_IR_SHAPE_H_

#include "mindapi/base/base.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/ir/common.h"

namespace mindspore::api {
/// \brief Shape defines dimensions of a tensor.
class MIND_API Shape : public Base {
 public:
  MIND_API_BASE_MEMBER(Shape);

  /// \brief Create Shape with the given shape dimensions.
  ///
  /// \param[in] shape The shape dimensions.
  explicit Shape(const ShapeVector &shape);

  /// \brief Get the shape dimensions.
  ///
  /// \return The shape dimensions.
  const ShapeVector &shape() const;
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_SHAPE_H_
