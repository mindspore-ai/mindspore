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

#include <vector>
#include <cmath>

#include "ops/grad/upsample_trilinear_3d_grad.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T>
void check_dims_grad(string check_dim_name, string op_name, std::vector<T> check_vector) {
  for (size_t i = 0; i < check_vector.size(); i++) {
    if (check_vector[i] <= static_cast<T>(0.0)) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', arg '" << check_dim_name << "' dimension " << i
                        << " value is <= 0.";
    }
  }
}

abstract::ShapePtr UpsampleTrilinear3DGradInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  string op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto grad_shape_ptr = input_args[0]->BuildShape();
  auto input_size_ptr = primitive->GetAttr("input_size");
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  auto input_size = GetValue<std::vector<int64_t>>(input_size_ptr);
  (void)CheckAndConvertUtils::CheckPositiveVector("input_size", input_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("the elements number of input_size", SizeToLong(input_size.size()), kEqual,
                                           SizeToLong(kInputIndex5), op_name);
  const size_t kDimSize5 = 5;
  if (grad_shape.size() != kDimSize5) {
    MS_EXCEPTION(TypeError) << "grad_shape of UpsampleTrilinear3D must be 5, but got" << grad_shape.size();
  }

  const size_t kOutputSizeDims = 3;
  const size_t kScalesDims = 3;
  auto output_size = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_size"));
  auto scales = GetValue<std::vector<float>>(primitive->GetAttr("scales"));
  if (output_size.empty() && scales.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', either output_size or scales should be defined.";
  } else if (!output_size.empty() && !scales.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', only one of output_size or scales should be defined.";
  }
  if (!output_size.empty() && output_size.size() != kOutputSizeDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', output_size must be size of 3, but got "
                             << std::to_string(output_size.size()) << ".";
  }
  if (!scales.empty() && scales.size() != kScalesDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', scales must be size of 3, but got "
                             << std::to_string(scales.size()) << ".";
  }
  // Check that the entering gradient, "grad", has the same dimensions as output
  string name_ = "scales";
  std::vector<int64_t> output_shape(input_size.size());
  output_shape[0] = input_size[0];
  output_shape[1] = input_size[1];
  if (output_size.empty()) {
    check_dims_grad(name_, op_name, scales);
    output_shape[kInputIndex2] = static_cast<int64_t>(std::floor(input_size[kInputIndex2] * scales[kInputIndex0]));
    output_shape[kInputIndex3] = static_cast<int64_t>(std::floor(input_size[kInputIndex3] * scales[kInputIndex1]));
    output_shape[kInputIndex4] = static_cast<int64_t>(std::floor(input_size[kInputIndex4] * scales[kInputIndex2]));
  } else {
    name_ = "output_size";
    check_dims_grad(name_, op_name, output_size);
    output_shape[kInputIndex2] = output_size[kInputIndex0];
    output_shape[kInputIndex3] = output_size[kInputIndex1];
    output_shape[kInputIndex4] = output_size[kInputIndex2];
  }

  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(input_size);
  }

  bool shape_error = false;
  const int shape_dims = 4;
  for (size_t i = 0; i < shape_dims; i++) {
    if (output_shape[i] != grad_shape[i]) {
      shape_error = true;
      break;
    }
  }
  if (shape_error) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', The shape of grad, which is the same as that of output, is "
                             << input_args[kInputIndex0]->BuildShape()->ToString() << ", but the shape of output is ("
                             << std::to_string(output_shape[kInputIndex0]) << ", "
                             << std::to_string(output_shape[kInputIndex1]) << ", "
                             << std::to_string(output_shape[kInputIndex2]) << ", "
                             << std::to_string(output_shape[kInputIndex3]) << ", "
                             << std::to_string(output_shape[kInputIndex4]) << ").";
  }

  // Return the dinput shape
  return std::make_shared<abstract::Shape>(input_size);
}

TypePtr UpsampleTrilinear3DGradInferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  TypePtr grad_type = input_args[kInputIndex0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, common_float_types, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(UpsampleTrilinear3DGrad, BaseOperator);
AbstractBasePtr UpsampleTrilinear3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_types = UpsampleTrilinear3DGradInferType(primitive, input_args);
  auto infer_shapes = UpsampleTrilinear3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shapes, infer_types);
}

std::vector<int64_t> UpsampleTrilinear3DGrad::get_out_spatial_size() const {
  auto value_ptr = this->GetAttr("output_size");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<float> UpsampleTrilinear3DGrad::get_scale_factors() const {
  auto value_ptr = this->GetAttr("scales");
  return GetValue<std::vector<float>>(value_ptr);
}
bool UpsampleTrilinear3DGrad::get_align_corners() const {
  auto value_ptr = this->GetAttr("align_corners");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGUpsampleTrilinear3DGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleTrilinear3DGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleTrilinear3DGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UpsampleTrilinear3DGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UpsampleTrilinear3DGrad, prim::kPrimUpsampleTrilinear3DGrad,
                                 AGUpsampleTrilinear3DGradInfer, false);
}  // namespace ops
}  // namespace mindspore
