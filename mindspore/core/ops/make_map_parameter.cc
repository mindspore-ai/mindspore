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
#include "ops/make_map_parameter.h"

#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/map_tensor.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(MakeMapParameter, BaseOperator);
AbstractBasePtr MakeMapParameterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // Check number of arguments.
  constexpr int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, kNameMakeMapParameter);

  // MakeMapParameter(key_tensor, value_tensor, default_value)
  constexpr int64_t key_arg_index = 0;
  constexpr int64_t value_arg_index = 1;
  constexpr int64_t default_value_arg_index = 2;
  if (!input_args[key_arg_index]->isa<abstract::AbstractTensor>() ||
      !input_args[value_arg_index]->isa<abstract::AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "The args of MakeMapParameter is invalid, they must be tensor. Please check:"
                      << input_args[key_arg_index]->ToString() << ", " << input_args[value_arg_index]->ToString();
  }
  // key_arg

  auto key_arg = input_args[key_arg_index]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(key_arg);
  auto key_arg_tensor = key_arg->cast<tensor::TensorPtr>();
  TypeId key_dtype_id =
    ((key_arg_tensor != nullptr) ? static_cast<TypeId>(key_arg_tensor->data_type_c()) : TypeId::kNumberTypeInt32);
  // value_arg
  auto value_arg = input_args[value_arg_index]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_arg);
  auto value_arg_tensor = value_arg->cast<tensor::TensorPtr>();
  TypeId value_dtype_id =
    ((value_arg_tensor != nullptr) ? static_cast<TypeId>(value_arg_tensor->data_type_c()) : TypeId::kNumberTypeFloat32);

  // value_shape
  auto shape = input_args[value_arg_index]->GetShapeTrack();
  MS_EXCEPTION_IF_NULL(shape);
  ShapeVector input_value_shape = shape->cast<abstract::ShapePtr>()->shape();
  if (input_value_shape.empty()) {
    MS_LOG(EXCEPTION) << "The input value shape is empty";
  }
  ShapeVector value_shape(input_value_shape.begin() + 1, input_value_shape.end());

  // shape
  ValuePtr default_value = input_args[default_value_arg_index]->GetValueTrack();
  auto map_tensor = std::make_shared<tensor::MapTensor>(key_dtype_id, value_dtype_id, value_shape, default_value);
  return std::make_shared<abstract::AbstractMapTensor>(map_tensor);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MakeMapParameter, prim::kPrimMakeMapParameter, MakeMapParameterInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
