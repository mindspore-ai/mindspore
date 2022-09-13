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

#include <algorithm>
#include <set>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kIouInputNums = 2;
constexpr size_t kIouInputDims = 2;
constexpr size_t kCoordinatesIndex = 1;
constexpr int64_t kCoordinatesSize = 4;
}  // namespace
MIND_API_OPERATOR_IMPL(IOU, BaseOperator);
class IOUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                             kIouInputNums, prim_name);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(x_shape_ptr);
    auto y_shape_ptr = input_args[kInputIndex1]->BuildShape();
    MS_EXCEPTION_IF_NULL(y_shape_ptr);
    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr);
    auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(y_shape_ptr);
    auto x_shp = x_shape_map[kShape];
    auto y_shp = y_shape_map[kShape];
    if (x_shp.size() != kIouInputDims || y_shp.size() != kIouInputDims) {
      MS_EXCEPTION(ValueError)
        << "For 'BatchMatMul', input x, y must have the same dimension size and must be 2. But got x size = "
        << x_shp.size() << ", y size = " << y_shp.size() << ".";
    }
    (void)CheckAndConvertUtils::CheckInteger("input numbers", x_shp[kCoordinatesIndex], kEqual, kCoordinatesSize,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("input numbers", y_shp[kCoordinatesIndex], kEqual, kCoordinatesSize,
                                             prim_name);
    ShapeVector ret_shape;
    ret_shape.push_back(y_shp[0]);
    ret_shape.push_back(x_shp[0]);
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
    (void)types.emplace("y", input_args[kInputIndex1]->BuildType());
    return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(IOU, prim::kPrimIOU, IOUInfer, false);
}  // namespace ops
}  // namespace mindspore
