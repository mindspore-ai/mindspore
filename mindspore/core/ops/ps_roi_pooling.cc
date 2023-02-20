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
#include "ops/ps_roi_pooling.h"

#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PSROIPoolingInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto group_size_ptr = primitive->GetAttr("group_size");
  MS_EXCEPTION_IF_NULL(group_size_ptr);
  auto group_size = GetValue<int64_t>(group_size_ptr);
  constexpr int64_t max_group_size = 128;
  // The value of group_size must be less than 128
  if (group_size <= 0 || group_size >= max_group_size) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', 'group_size' should be in the range (0, 128), but got: " << group_size;
  }

  auto output_dim_ptr = primitive->GetAttr("output_dim");
  MS_EXCEPTION_IF_NULL(output_dim_ptr);
  auto output_dim = GetValue<int64_t>(output_dim_ptr);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  constexpr size_t x_out_shape_dim = 4;
  if (!IsDynamicRank(x_shape)) {
    if (x_shape.size() != x_out_shape_dim) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', input x shape must be 4d(NCHW), but got: " << x_shape.size();
    }
    if (x_shape[1] != abstract::Shape::kShapeDimAny) {
      // the first dimension of the input data should be equal group_size * group_size * output_dim
      if (x_shape[1] / (group_size * group_size) != output_dim) {
        MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', the second dimension(" << x_shape[1]
                          << ") of the input x is illegal, it is not equal to group_size(" << group_size
                          << ") * group_size(" << group_size << ") * output_dim(" << output_dim << ").";
      }
    }
  }
  auto rois_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  std::vector<int64_t> ret_shape(x_out_shape_dim);
  if (IsDynamicRank(rois_shape)) {
    ret_shape = {-1, output_dim, group_size, group_size};
  } else {
    constexpr size_t rois_shape_dim = 3;
    constexpr size_t dim2 = 2;
    if (rois_shape.size() < rois_shape_dim) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', the dimension of 'rois' should be equal 3, but got: " << rois_shape.size();
    }
    if (rois_shape[0] == abstract::Shape::kShapeDimAny || rois_shape[dim2] == abstract::Shape::kShapeDimAny) {
      ret_shape = {-1, output_dim, group_size, group_size};
    } else {
      ret_shape = {rois_shape[0] * rois_shape[dim2], output_dim, group_size, group_size};
    }
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr PSROIPoolingInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), {kFloat64, kFloat32, kFloat16},
                                                   prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("rois", input_args[1]->BuildType(), {kFloat64, kFloat32, kFloat16},
                                                   prim->name());

  auto input_type = input_args[0]->BuildType();
  auto rois_type = input_args[1]->BuildType();
  if (input_type->ToString() != rois_type->ToString()) {
    MS_EXCEPTION(TypeError) << "For '" << prim->name()
                            << "', input[features] is expected to have the same type with input[rois], but got type ("
                            << input_type << ", " << rois_type << ").";
  }

  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(PSROIPooling, BaseOperator);
AbstractBasePtr PSROIPoolingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infertype = PSROIPoolingInferType(primitive, input_args);
  auto infershape = PSROIPoolingInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGPSROIPoolingInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PSROIPoolingInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PSROIPooling, prim::kPrimPSROIPooling, AGPSROIPoolingInfer, false);
}  // namespace ops
}  // namespace mindspore
