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
#include "ops/resize_linear_1d.h"

#include <set>
#include <string>
#include <algorithm>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kInputShape0Dim = 3;
const int64_t kInputShape1Dim = 1;
abstract::ShapePtr ResizeLinear1DInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  const int64_t shape0_dim = 3;
  std::vector<int64_t> output_shape(shape0_dim, abstract::Shape::kShapeDimAny);

  auto shape0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (!IsDynamicRank(shape0)) {
    (void)CheckAndConvertUtils::CheckInteger("images' rank", SizeToLong(shape0.size()), kEqual, shape0_dim, prim_name);
    output_shape[kInputIndex0] = shape0[kInputIndex0];
    output_shape[kInputIndex1] = shape0[kInputIndex1];
  }

  auto value_ptr = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(value_ptr);

  if (!IsValueKnown(value_ptr)) {
    return std::make_shared<abstract::Shape>(output_shape);
  }

  auto size_type = input_args[kInputIndex1]->BuildType();
  std::vector<int64_t> size_value{};
  if (size_type->isa<TensorType>()) {
    const int64_t kDimOne = 1;
    auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("rank of size's shape", SizeToLong(size_shape.size()), kEqual, kDimOne,
                                             prim_name);
    size_value = CheckAndConvertUtils::CheckTensorIntValue("size", value_ptr, prim_name);
  } else if (IsIdentidityOrSubclass(size_type, kTuple) || IsIdentidityOrSubclass(size_type, kList)) {
    size_value = CheckAndConvertUtils::CheckIntOrTupleInt("size", value_ptr, prim_name);
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the `size` "
                            << " must be a tuple、list or tensor with all Int elements, but got "
                            << value_ptr->type_name() << ".";
  }

  const int64_t size_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("size", SizeToLong(size_value.size()), kEqual, size_num, prim_name);
  const int64_t kNumZero = 0;
  for (size_t i = 0; i < size_value.size(); ++i) {
    CheckAndConvertUtils::CheckInteger("size", size_value[i], kGreaterThan, kNumZero, prim_name);
  }

  output_shape[kInputIndex2] = size_value[kInputIndex0];

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeLinear1DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr arg) { return arg == nullptr; })) {
    MS_LOG(EXCEPTION) << "For 'ResizeLinear1D', input args contain nullptr.";
  }
  auto prim_name = primitive->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto size_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid0_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", x_type, valid0_types, prim_name);
  if (size_type->isa<TensorType>()) {
    const std::set<TypePtr> valid1_types = {kInt32, kInt64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, valid1_types, prim_name);
  }
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ResizeLinear1D, BaseOperator);

void ResizeLinear1D::set_coordinate_transformation_mode(const std::string coordinate_transformation_mode) {
  (void)this->AddAttr("coordinate_transformation_mode", api::MakeValue(coordinate_transformation_mode));
}
std::string ResizeLinear1D::get_coordinate_transformation_mode() const {
  auto value_ptr = GetAttr("coordinate_transformation_mode");
  return GetValue<std::string>(value_ptr);
}

void ResizeLinear1D::Init(const std::string coordinate_transformation_mode) {
  this->set_coordinate_transformation_mode(coordinate_transformation_mode);
}

abstract::AbstractBasePtr ResizeLinear1DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ResizeLinear1DInferType(primitive, input_args);
  auto infer_shape = ResizeLinear1DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGResizeLinear1DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeLinear1DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeLinear1D, prim::kPrimResizeLinear1D, AGResizeLinear1DInfer, false);
}  // namespace ops
}  // namespace mindspore
