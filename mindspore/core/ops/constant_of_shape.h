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
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameConstantOfShape = "ConstantOfShape";
class ConstantOfShape : public PrimitiveC {
 public:
  ConstantOfShape() : PrimitiveC(kNameConstantOfShape) {}
  ~ConstantOfShape() = default;
  MS_DECLARE_PARENT(ConstantOfShape, PrimitiveC);
  void Init(int64_t data_type, const std::vector<float> &value);
  void set_data_type(int64_t data_type);
  void set_value(const std::vector<float> &value);
  int64_t get_data_type() const;
  std::vector<float> get_value() const;
};

AbstractBasePtr ConstantOfShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args);
using PrimConstantOfShapePtr = std::shared_ptr<ConstantOfShape>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CONSTANT_OF_SHAPE_H_
