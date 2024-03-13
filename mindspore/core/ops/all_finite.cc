/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/all_finite.h"

#include <memory>
#include <set>
#include <vector>
#include <string>
#include <map>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr AllFiniteInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) {
  MS_EXCEPTION_IF_NULL(primitive);
  ShapeVector ret_shape = {};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr AllFiniteInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) {
  MS_EXCEPTION_IF_NULL(primitive);
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

AbstractBasePtr AllFiniteInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  auto types = AllFiniteInferType(primitive, input_args);
  auto shapes = AllFiniteInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

class MIND_API AGAllFiniteInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AllFiniteInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AllFiniteInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AllFiniteInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AllFinite, prim::kPrimAllFinite, AGAllFiniteInfer, false);
}  // namespace ops
}  // namespace mindspore
