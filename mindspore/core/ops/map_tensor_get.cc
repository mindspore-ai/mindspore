/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "ops/map_tensor_get.h"
#include <memory>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sparse_tensor_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MapTensorGet, BaseOperator);

abstract::ShapePtr MapTensorGetInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto key_tensor_shape = CheckAndConvertUtils::GetTensorInputShape(kNameMapTensorGet, input_args, kInputIndex1);
  if (key_tensor_shape->shape().size() != 1) {
    MS_EXCEPTION(TypeError) << kNameMapTensorGet << " - key_tensor shape should be 1 rank"
                            << " but got " << key_tensor_shape->ToString() << ".";
  }
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGet, input_args, kInputIndex0);
  auto value_shape = abs_map_tensor->value_shape();
  // Concate key shape and value shape as the result shape.
  ShapeVector shape_vec = key_tensor_shape->shape();
  const auto &value_shape_vec = value_shape->shape();
  (void)shape_vec.insert(shape_vec.end(), value_shape_vec.begin(), value_shape_vec.end());
  auto infer_shape = std::make_shared<abstract::Shape>(shape_vec);
  return infer_shape;
}

TypePtr MapTensorGetInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  // Check number of arguments.
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorGet);
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGet, input_args, kInputIndex0);

  // Get key dtype, value dtype and value shape of the map tensor.
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  auto value_dtype = map_tensor_type->value_dtype();
  return value_dtype;
}

AbstractBasePtr MapTensorGetInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  auto value_dtype = MapTensorGetInferType(primitive, input_args);
  auto infer_shape = MapTensorGetInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(value_dtype, infer_shape);
}

// AG means auto generated
class MIND_API AGMapTensorGetInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MapTensorGetInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MapTensorGet, prim::kPrimMapTensorGet, AGMapTensorGetInfer, false);
}  // namespace ops
}  // namespace mindspore
