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
#include "ops/randperm_v2.h"

#include <memory>
#include <set>
#include <vector>
#include <climits>
#include <string>

#include "utils/check_convert_utils.h"
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
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t GetDtypeMaxForCheckOverFlow(TypePtr tid) {
  int64_t max = 0;
  int64_t max_float16 = 65504;
  switch (tid->type_id()) {
    case kNumberTypeUInt8:
      max = UCHAR_MAX;
      break;
    case kNumberTypeInt8:
      max = SCHAR_MAX;
      break;
    case kNumberTypeInt16:
      max = SHRT_MAX;
      break;
    case kNumberTypeInt32:
      max = INT_MAX;
      break;
    case kNumberTypeFloat16:
      max = max_float16;
      break;
    default:
      max = LONG_MAX - 1;
      break;
  }
  return max;
}

abstract::ShapePtr RandpermV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(primitive);
  auto dtype_value = primitive->GetAttr("dtype");
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For RandpermV2, the dtype of " << prim_name << "is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  int64_t max_data = GetDtypeMaxForCheckOverFlow(output_type);
  auto n_value = input_args[kInputIndex0]->BuildValue();
  auto seed_value = input_args[kInputIndex1]->BuildValue();
  auto offset_value = input_args[kInputIndex2]->BuildValue();
  if (!n_value->isa<AnyValue>() && !n_value->isa<None>()) {
    auto n = CheckAndConvertUtils::CheckTensorIntValue("n", n_value, prim_name)[0];
    auto seed = CheckAndConvertUtils::CheckTensorIntValue("seed", seed_value, prim_name)[0];
    auto offset = CheckAndConvertUtils::CheckTensorIntValue("offset", offset_value, prim_name)[0];
    if (seed < 0 && seed != -1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the input seed must be greater than 0 or equal to 0 or -1, but got data: " << seed
                               << ".";
    }
    if (offset < 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the input offset must be greater than or equal to 0, but got data: " << offset
                               << ".";
    }
    if (n <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input n must be greater than 0, but got data: " << n
                               << ".";
    }
    if (n > max_data + 1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', n_value must be less than or equal to max_data "
                               << "of its type +1, but got data: " << n << ".";
    }
    ShapeVector out_shape = {n};
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    ShapeVector output_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr RandpermV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto input_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  std::set<TypePtr> input_valid_types{kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("input_n", input_type, input_valid_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For RandpermV2, the dtype of " << prim_name << "is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_output_types = {kInt32, kInt64, kInt16, kInt8, kUInt8, kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_output_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(RandpermV2, BaseOperator);

AbstractBasePtr RandpermV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = RandpermV2InferType(primitive, input_args);
  auto infer_shape = RandpermV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGRandpermV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return RandpermV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return RandpermV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return RandpermV2Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0, 1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RandpermV2, prim::kPrimRandpermV2, AGRandpermV2Infer, false);
}  // namespace ops
}  // namespace mindspore
