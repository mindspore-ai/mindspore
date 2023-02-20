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
#include "ops/parameterized_truncated_normal.h"

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <functional>
#include <map>
#include <numeric>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
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
void ParameterizedTruncatedNormalCheckdims(const std::vector<AbstractBasePtr> &input_args, const int64_t batch_size) {
  std::vector<string> parameters = {"mean", "stdevs", "min", "max"};
  if (batch_size < 0) {
    MS_EXCEPTION(ValueError) << "For ParameterizedTruncatedNormal, the batch size must be >= 0.";
  }
  for (size_t i = 1; i < input_args.size(); i++) {
    auto para_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[i]->BuildShape())[kShape];
    if (para_shape.size() > 1) {
      MS_EXCEPTION(ValueError) << "For ParameterizedTruncatedNormal, " << parameters.at(i - 1)
                               << "should be at most rank 1, but got rank " << para_shape.size() << ".";
    }
    if (para_shape.size() == 1) {
      int64_t para_num = std::accumulate(para_shape.begin(), para_shape.end(), int64_t(1), std::multiplies{});
      if (!(para_num == 1 || para_num == batch_size)) {
        MS_EXCEPTION(ValueError) << "For ParameterizedTruncatedNormal, " << parameters.at(i - 1) << "must be 0d, or "
                                 << parameters.at(i - 1) << ".shape = (" << batch_size << ", ), but got (" << para_num
                                 << ", ).";
      }
    }
  }
}

abstract::ShapePtr ParameterizedTruncatedNormalInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t kShapeSize = 2;
  auto shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamic(shape_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  (void)CheckAndConvertUtils::CheckInteger("rank of argument[shape]", SizeToLong(shape_shape.size()), kEqual, 1,
                                           op_name);
  if (shape_shape[0] > 0) {
    (void)CheckAndConvertUtils::CheckInteger("size of argument[shape]", shape_shape[0], kGreaterEqual, kShapeSize,
                                             op_name);
  }

  auto shape_value = input_args[kInputIndex0]->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);
  if (!shape_value->isa<AnyValue>() && !shape_value->isa<None>()) {
    auto out_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", shape_value, op_name);
    (void)CheckAndConvertUtils::CheckPositiveVector("shape", out_shape, op_name);
    ParameterizedTruncatedNormalCheckdims(input_args, out_shape[0]);

    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> output_shape = {-2};
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr ParameterizedTruncatedNormalInferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_type_shape = input_args[kInputIndex0]->BuildType();
  auto input_type_mean = input_args[kInputIndex1]->BuildType();
  auto input_type_stdevs = input_args[kInputIndex2]->BuildType();
  auto input_type_min = input_args[kInputIndex3]->BuildType();
  auto input_type_max = input_args[kInputIndex4]->BuildType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_type_shape, valid_types, prim_name);
  const std::set<TypePtr> valid_types_mean = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("mean", input_type_mean);
  (void)types.emplace("stdevs", input_type_stdevs);
  (void)types.emplace("min", input_type_min);
  (void)types.emplace("max", input_type_max);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types_mean, prim_name);
  return input_type_mean;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ParameterizedTruncatedNormal, BaseOperator);

void ParameterizedTruncatedNormal::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}
int64_t ParameterizedTruncatedNormal::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void ParameterizedTruncatedNormal::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

int64_t ParameterizedTruncatedNormal::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void ParameterizedTruncatedNormal::set_seed2(const int64_t seed2) {
  (void)this->AddAttr(kSeed2, api::MakeValue(seed2));
}

AbstractBasePtr ParameterizedTruncatedNormalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ParameterizedTruncatedNormalInferType(primitive, input_args);
  auto infer_shape = ParameterizedTruncatedNormalInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGParameterizedTruncatedNormalInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ParameterizedTruncatedNormalInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ParameterizedTruncatedNormalInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ParameterizedTruncatedNormalInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ParameterizedTruncatedNormal, prim::kPrimParameterizedTruncatedNormal,
                                 AGParameterizedTruncatedNormalInfer, false);
}  // namespace ops
}  // namespace mindspore
