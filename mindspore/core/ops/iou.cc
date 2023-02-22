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

#include "ops/iou.h"

#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
constexpr int64_t kIOUInputNums = 2;
constexpr size_t kIOUInputDims = 2;
constexpr size_t kCoordinatesIndex = 1;
constexpr int64_t kCoordinatesSize = 4;
constexpr float kDefaultValue = 1.0;
const char kEpsName[] = "eps";
}  // namespace
MIND_API_OPERATOR_IMPL(IOU, BaseOperator);
class IOUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kIOUInputNums,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);

    if (primitive->GetAttr(kEpsName) == nullptr) {
      primitive->set_attr(kEpsName, MakeValue(kDefaultValue));
    }
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(x_shape_ptr);
    auto y_shape_ptr = input_args[kInputIndex1]->BuildShape();
    MS_EXCEPTION_IF_NULL(y_shape_ptr);
    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr);
    auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(y_shape_ptr);
    auto x_shp = x_shape_map[kShape];
    auto y_shp = y_shape_map[kShape];
    if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    }

    if (x_shp.size() != kIOUInputDims || y_shp.size() != kIOUInputDims) {
      MS_EXCEPTION(ValueError) << "For '" << kNameIOU
                               << "', input x, y must have the same dimension size and must be 2. But got x size = "
                               << x_shp.size() << ", y size = " << y_shp.size() << ".";
    }
    if (x_shp[kCoordinatesIndex] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("anchor_boxes.shape[1]", x_shp[kCoordinatesIndex], kEqual,
                                               kCoordinatesSize, prim_name);
    }
    if (y_shp[kCoordinatesIndex] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("gt_boxes.shape[1]", y_shp[kCoordinatesIndex], kEqual, kCoordinatesSize,
                                               prim_name);
    }

    ShapeVector ret_shape;
    ret_shape.push_back(y_shp[0]);
    ret_shape.push_back(x_shp[0]);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kIOUInputNums,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);

    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("y", input_args[kInputIndex1]->BuildType());
    return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  }
};

abstract::AbstractBasePtr IouInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  IOUInfer io_infer;
  auto shape = io_infer.InferShape(primitive, input_args);
  auto type = io_infer.InferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(IOU, prim::kPrimIOU, IOUInfer, false);
}  // namespace ops
}  // namespace mindspore
