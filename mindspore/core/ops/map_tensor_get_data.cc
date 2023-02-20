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
#include "ops/map_tensor_get_data.h"

#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MapTensorGetData, BaseOperator);
AbstractBasePtr MapTensorGetDataInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMapTensorGetData);
  // Check argument abstracts.
  auto abs_map_tensor =
    CheckAndConvertUtils::CheckArgs<abstract::AbstractMapTensor>(kNameMapTensorGetData, input_args, kInputIndex0);
  auto map_tensor_type = abs_map_tensor->map_tensor_type();
  MS_EXCEPTION_IF_NULL(map_tensor_type);

  // key abstract
  const auto &key_dtype = map_tensor_type->key_dtype();
  ShapeVector shape = abs_map_tensor->shape()->shape();
  if (shape.empty()) {
    MS_LOG(EXCEPTION) << "Invalid shape:" << input_args[0]->ToString();
  }
  ShapeVector key_shape = {shape[0]};
  auto key_abs = std::make_shared<abstract::AbstractTensor>(key_dtype, key_shape);

  // value abstract
  auto value_dtype = map_tensor_type->value_dtype();
  auto value_abs = std::make_shared<abstract::AbstractTensor>(value_dtype, abs_map_tensor->shape());

  AbstractBasePtrList abstract_list{key_abs, value_abs};
  return std::make_shared<abstract::AbstractTuple>(abstract_list);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MapTensorGetData, prim::kPrimMapTensorGetData, MapTensorGetDataInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
