/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ops/truncated_normal.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/random_ops.h"
#include "mindspore/core/ops/op_utils.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
const uint32_t kInputDims = 1;
const uint32_t kInputSizes = 2;
abstract::ShapePtr TruncatedNormalInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  if (!CheckAndConvertUtils::IsTensor(input_args[0])) {
    MS_EXCEPTION(TypeError) << "Input[0] only support tensor!";
  }
  auto shape_input_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape());
  auto shape_input = shape_input_map[kShape];
  if (IsDynamicRank(shape_input)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInpuDims = 1;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto input_value = input_args[0]->GetValue();
  MS_EXCEPTION_IF_NULL(input_value);
  auto input_type = input_args[0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  if (shape_v.size() != kInpuDims) {
    MS_EXCEPTION(ValueError) << "The input tensor must be a 1-D tensor.";
  }
  if (ops::IsValueKnown(input_value)) {
    auto out_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", input_value, "", input_type);
    size_t shape_m =
      std::accumulate(out_shape.begin(), out_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    if (shape_m > LongToSize(max_length)) {
      MS_EXCEPTION(ValueError) << "The number of elements of output must be less than max length: " << max_length
                               << ", but got " << shape_m
                               << "! The shape of  output must be reduced or max_length must be increased";
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> output_shape;
    for (int i = 0; i < shape_v[0]; i++) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr TruncatedNormalInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const uint32_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  const std::set<TypePtr> valid_input_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_args[0]->GetType(), valid_input_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_value);
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "The dtype of " + prim_name + " is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_output_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_output_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(TruncatedNormal, BaseOperator);

void TruncatedNormal::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

int64_t TruncatedNormal::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void TruncatedNormal::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

int64_t TruncatedNormal::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void TruncatedNormal::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

AbstractBasePtr TruncatedNormalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = TruncatedNormalInferType(primitive, input_args);
  auto infer_shape = TruncatedNormalInferShape(primitive, input_args);
  return abstract::MakeAbstractTensor(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGTruncatedNormalInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TruncatedNormalInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TruncatedNormalInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return TruncatedNormalInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TruncatedNormal, prim::kPrimTruncatedNormal, AGTruncatedNormalInfer, false);
}  // namespace ops
}  // namespace mindspore
