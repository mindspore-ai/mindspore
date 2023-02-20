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
#include "ops/map_tensor_get.h"

#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "utils/ms_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MapTensorGet, BaseOperator);
AbstractBasePtr MapTensorGetInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorGet);
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGet, input_args, kInputIndex0);

  // Get key dtype, value dtype and value shape of the map tensor.
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  auto key_dtype = map_tensor_type->key_dtype();
  auto value_dtype = map_tensor_type->value_dtype();
  auto value_shape = abs_map_tensor->value_shape();

  // Check 'key_tensor' dtype and shape.
  auto key_tensor_dtype = CheckAndConvertUtils::GetTensorInputType(kNameMapTensorGet, input_args, kInputIndex1);
  if (!common::IsEqual(key_dtype, key_tensor_dtype)) {
    MS_EXCEPTION(TypeError) << kNameMapTensorGet << " - required key_tensor dtype " << key_dtype->ToString()
                            << " but got " << key_tensor_dtype->ToString() << ".";
  }
  auto key_tensor_shape = CheckAndConvertUtils::GetTensorInputShape(kNameMapTensorGet, input_args, kInputIndex1);
  if (key_tensor_shape->shape().size() != 1) {
    MS_EXCEPTION(TypeError) << kNameMapTensorGet << " - key_tensor shape should be 1 rank"
                            << " but got " << key_tensor_shape->ToString() << ".";
  }

  // Concate key shape and value shape as the result shape.
  ShapeVector shape_vec = key_tensor_shape->shape();
  const auto &value_shape_vec = value_shape->shape();
  (void)shape_vec.insert(shape_vec.end(), value_shape_vec.begin(), value_shape_vec.end());
  auto infer_shape = std::make_shared<abstract::Shape>(shape_vec);
  return std::make_shared<abstract::AbstractTensor>(value_dtype, infer_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MapTensorGet, prim::kPrimMapTensorGet, MapTensorGetInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
