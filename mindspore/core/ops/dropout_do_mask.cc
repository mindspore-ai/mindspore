/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/dropout_do_mask.h"

#include <set>
#include <string>
#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
T GetAndCheckKeepProp(const tensor::TensorPtr &keep_prop) {
  auto value = reinterpret_cast<T *>(keep_prop->data_c());
  MS_EXCEPTION_IF_NULL(value);
  T min = (T)0.0;
  T max = (T)1.0;
  if (*value < min || *value > max) {
    MS_EXCEPTION(ValueError) << "DropoutDoMask's keep_prop input value must in the range [0, 1], but got " << *value;
  }
  return *value;
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
  auto mask_shape = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 1);

  auto x_shape_vector = x_shape->shape();
  auto mask_shape_vector = mask_shape->shape();

  int64_t x_size = 1;
  for (size_t i = 0; i < x_shape_vector.size(); i++) {
    x_size *= x_shape_vector[i];
  }
  if (mask_shape_vector.size() != 1) {
    MS_EXCEPTION(ValueError) << "DropoutDoMask input mask must be 1-dimension.";
  }
  auto mask_size = mask_shape_vector[0] * 8;
  if (x_size > mask_size) {
    MS_EXCEPTION(ValueError) << "DropoutDoMask input mask do not match input,  input_x shape: " << x_shape->ToString()
                             << ", mask shape: " << mask_shape->ToString();
  }
  auto keep_prop = input_args[kInputIndex2];
  if (keep_prop->isa<abstract::AbstractTensor>()) {
    auto keep_prop_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(keep_prop->BuildShape())[kShape];
    if (!keep_prop_shape.empty()) {
      MS_EXCEPTION(ValueError) << "DropoutDoMask keep_prop's dim must be 0(scalar), but got " << keep_prop_shape.size();
    }
  }
  return x_shape;
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto keep_prop = input_args[kInputIndex2];
  MS_EXCEPTION_IF_NULL(keep_prop);
  auto keep_prop_value = keep_prop->BuildValue();
  MS_EXCEPTION_IF_NULL(keep_prop_value);

  if (keep_prop->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> keep_prop_valid_types = {kFloat16, kFloat32, kFloat64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("keep prop", keep_prop->BuildType(), keep_prop_valid_types,
                                                     op_name);
    if (keep_prop_value->isa<tensor::Tensor>()) {
      auto keep_prop_tensor = keep_prop_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(keep_prop_tensor);
      TypeId tensor_type = keep_prop_tensor->data_type();
      if (tensor_type == TypeId::kNumberTypeFloat16) {
        (void)GetAndCheckKeepProp<float16>(keep_prop_tensor);
      } else if (tensor_type == TypeId::kNumberTypeFloat32) {
        (void)GetAndCheckKeepProp<float>(keep_prop_tensor);
      } else {
        (void)GetAndCheckKeepProp<double>(keep_prop_tensor);
      }
    }
  } else if (keep_prop->isa<abstract::AbstractScalar>()) {
    if (keep_prop_value != nullptr) {
      if (!keep_prop_value->isa<FloatImm>()) {
        MS_EXCEPTION(TypeError) << "DropoutDoMask's keep_prop input type must be float.";
      }
      auto value = GetValue<float>(keep_prop_value);
      if (value < 0 || value > 1) {
        MS_EXCEPTION(ValueError) << "DropoutDoMask's keep_prop input value must in the range [0, 1].";
      }
    }
  } else {
    MS_EXCEPTION(TypeError) << "The DropoutDoMask's keep_prop input must be a float number or tensor.";
  }

  (void)CheckAndConvertUtils::CheckTensorTypeValid("inputs", input_args[1]->BuildType(), {kUInt8}, op_name);
  const std::set<TypePtr> input_valid_types = {kFloat16, kFloat32, kInt32};
  return CheckAndConvertUtils::CheckTensorTypeValid("inputs", input_args[0]->BuildType(), input_valid_types, op_name);
}
}  // namespace

AbstractBasePtr DropoutDoMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("infer shape", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           primitive->name());
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(DropoutDoMask, prim::kPrimDropoutDoMask, DropoutDoMaskInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
