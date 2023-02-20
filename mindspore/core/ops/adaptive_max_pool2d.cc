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

#include "ops/adaptive_max_pool2d.h"

#include <memory>
#include <set>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
std::vector<int64_t> AdaptiveMaxPool2D::output_size() const {
  auto value_ptr = GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
abstract::BaseShapePtr AdaptiveMaxPool2DInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 1) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the num of input args should be 1, but got "
                             << input_args.size();
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  if (shape_map.empty()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>());
  }

  auto in_shape_vector = shape_map[kShape];
  const auto &output_size_ptr = primitive->GetAttr("output_size");
  MS_EXCEPTION_IF_NULL(output_size_ptr);
  const auto &output_size = GetValue<std::vector<int64_t>>(output_size_ptr);
  if (in_shape_vector.size() == 1) {
    if (in_shape_vector[0] != kDynamicRankValue) {
      MS_EXCEPTION(ValueError)
        << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must be 3 "
           "or 4, but got shape size is 1.";
    }
  } else if ((in_shape_vector.size() != kFormatCHWShapeSize && in_shape_vector.size() != kFormatNCHWShapeSize) ||
             output_size.size() != kOutputSizeAttrSize) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must be 3 "
                                "or 4 and the size of attr[output_size] must be 2, but got shape size:"
                             << in_shape_vector.size() << " and output_size size:" << output_size.size();
  }

  // Update the output shape by output size and input shape.
  if (in_shape_vector.size() != 1) {
    auto input_size_iter = in_shape_vector.rbegin();
    auto output_size_iter = output_size.rbegin();
    for (; output_size_iter != output_size.rend(); ++output_size_iter, ++input_size_iter) {
      // If output size is none, the input shape should be used.
      if (*output_size_iter != kPyValueNone) {
        *input_size_iter = *output_size_iter;
      }
    }
  }
  auto in_shape = std::make_shared<abstract::Shape>(in_shape_vector);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{in_shape, in_shape});
}

TypePtr AdaptiveMaxPool2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  if (input_args.size() != 1) {
    MS_EXCEPTION(ValueError) << "For primitive[AdaptiveMaxPool2D], the num of input args should be 1, but got "
                             << input_args.size();
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input_type =
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim->name());

  auto indices_type = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, indices_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool2D, BaseOperator);
AbstractBasePtr AdaptiveMaxPool2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(AdaptiveMaxPool2DInferShape(primitive, input_args),
                                AdaptiveMaxPool2DInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGAdaptiveMaxPool2DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool2DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveMaxPool2D, prim::kPrimAdaptiveMaxPool2D, AGAdaptiveMaxPool2DInfer, false);
}  // namespace ops
}  // namespace mindspore
