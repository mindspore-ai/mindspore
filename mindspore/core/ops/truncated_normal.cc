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

#include <string>
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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const uint32_t kInputDims = 1;
const uint32_t kInputSizes = 2;
abstract::ShapePtr TruncatedNormalInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "Input[0] only support tensor!";
  }
  auto shape_input_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto shape_input = shape_input_map[kShape];
  if (IsDynamicRank(shape_input)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInpuDims = 1;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto input_shape = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);
  auto input_shape_tensor = input_shape_value_ptr->cast<tensor::TensorPtr>();
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  if (shape_v.size() != kInpuDims) {
    MS_EXCEPTION(ValueError) << "The input tensor must be a 1-D tensor.";
  }
  if (!input_args[0]->BuildValue()->isa<AnyValue>() && !input_args[0]->BuildValue()->isa<None>()) {
    std::vector<int64_t> out_shape;
    int64_t shape_m = 1;
    if (input_type_element->type_id() == kNumberTypeInt32) {
      auto input_shape_ptr = reinterpret_cast<int32_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
        } else {
          MS_EXCEPTION(ValueError) << "Each dimension must be greater than 0.";
        }
      }
    } else if (input_type_element->type_id() == kNumberTypeInt64) {
      auto input_shape_ptr = reinterpret_cast<int64_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
        } else {
          MS_EXCEPTION(ValueError) << "Each dimension must be greater than 0.";
        }
      }
    }
    if (shape_m > max_length) {
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
  (void)CheckAndConvertUtils::CheckTensorTypeValid("shape", input_args[0]->BuildType(), valid_input_types, prim_name);
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
  return abstract::MakeAbstract(infer_shape, infer_type);
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
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TruncatedNormal, prim::kPrimTruncatedNormal, AGTruncatedNormalInfer, false);
}  // namespace ops
}  // namespace mindspore
