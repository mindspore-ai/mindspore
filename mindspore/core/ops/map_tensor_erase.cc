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
#include "ops/map_tensor_erase.h"

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
MIND_API_OPERATOR_IMPL(MapTensorErase, BaseOperator);
AbstractBasePtr MapTensorEraseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorErase);
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorErase, input_args, kInputIndex0);

  // Get key dtype of the map tensor.
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);
  auto key_dtype = map_tensor_type->key_dtype();

  // Check 'key_tensor' dtype and shape.
  auto key_tensor_dtype = CheckAndConvertUtils::GetTensorInputType(kNameMapTensorErase, input_args, kInputIndex1);
  if (!common::IsEqual(key_dtype, key_tensor_dtype)) {
    MS_EXCEPTION(TypeError) << kNameMapTensorErase << " - required key_tensor dtype " << key_dtype->ToString()
                            << " but got " << key_tensor_dtype->ToString() << ".";
  }
  auto key_tensor_shape = CheckAndConvertUtils::GetTensorInputShape(kNameMapTensorErase, input_args, kInputIndex1);
  if (key_tensor_shape->shape().size() != 1) {
    MS_EXCEPTION(TypeError) << kNameMapTensorErase << " - key_tensor shape should be 1 rank"
                            << " but got " << key_tensor_shape->ToString() << ".";
  }
  // Return the input AbstractMapTensor.
  return abs_map_tensor;
}
REGISTER_PRIMITIVE_EVAL_IMPL(MapTensorErase, prim::kPrimMapTensorErase, MapTensorEraseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
