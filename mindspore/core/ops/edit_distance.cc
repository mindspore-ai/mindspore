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

#include "ops/edit_distance.h"

#include <set>
#include <vector>
#include <algorithm>
#include <memory>

#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
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
void EditDistance::Init(const bool normalize) { this->set_normalize(normalize); }
void EditDistance::set_normalize(const bool normalize) { (void)this->AddAttr(kNormalize, api::MakeValue(normalize)); }
bool EditDistance::normalize() const {
  auto value_ptr = GetAttr(kNormalize);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::ShapePtr EditDistanceInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t kInputNum = 6;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kInputNum, prim_name);

  auto GetShape = [&input_args](size_t index) {
    auto &abs = input_args[index];
    MS_EXCEPTION_IF_NULL(abs);
    return CheckAndConvertUtils::ConvertShapePtrToShapeMap(abs->BuildShape())[kShape];
  };

  auto hypothesis_indices_shape = GetShape(kIndex0);
  auto hypothesis_values_shape = GetShape(kIndex1);
  auto hypothesis_shape_shape = GetShape(kIndex2);
  auto truth_indices_shape = GetShape(kIndex3);
  auto truth_values_shape = GetShape(kIndex4);
  auto truth_shape_shape = GetShape(kIndex5);

  const int64_t indices_rank = 2;
  const int64_t values_shape_rank = 1;
  (void)CheckAndConvertUtils::CheckInteger("hypothesis_indices rank", SizeToLong(hypothesis_indices_shape.size()),
                                           kEqual, indices_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("truth_indices rank", SizeToLong(truth_indices_shape.size()), kEqual,
                                           indices_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("hypothesis_values rank", SizeToLong(hypothesis_values_shape.size()), kEqual,
                                           values_shape_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("hypothesis_shape rank", SizeToLong(hypothesis_shape_shape.size()), kEqual,
                                           values_shape_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("truth_values rank", SizeToLong(truth_values_shape.size()), kEqual,
                                           values_shape_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("truth_shape rank", SizeToLong(truth_shape_shape.size()), kEqual,
                                           values_shape_rank, prim_name);

  if (hypothesis_values_shape[kIndex0] != hypothesis_indices_shape[kIndex0]) {
    MS_EXCEPTION(ValueError) << "hypothesis_values shape should be equal to hypothesis_indices shape[0] but got "
                             << "hypothesis_values shape: " << hypothesis_values_shape[kIndex0]
                             << " and hypothesis_indices shape[0]: " << hypothesis_indices_shape[kIndex0] << ".";
  }

  auto hypothesis_shape_shape_val = hypothesis_shape_shape[kIndex0];

  if (hypothesis_shape_shape_val != hypothesis_indices_shape[kIndex1]) {
    MS_EXCEPTION(ValueError) << "hypothesis_shape should be equal to hypothesis_indices shape[1] but got "
                             << "hypothesis_shape: " << hypothesis_shape_shape_val
                             << " and hypothesis_indices shape[1]: " << hypothesis_indices_shape[kIndex1] << ".";
  }
  if (truth_values_shape[kIndex0] != truth_indices_shape[kIndex0]) {
    MS_EXCEPTION(ValueError) << "truth_values shape should be equal to truth_indices shape[0] but got "
                             << "truth_values shape: " << truth_values_shape[kIndex0]
                             << " and truth_indices shape[0]: " << truth_indices_shape[kIndex0] << ".";
  }
  if (hypothesis_shape_shape_val != truth_shape_shape[kIndex0]) {
    MS_EXCEPTION(ValueError) << "hypothesis_shape should be equal to truth_shape but got hypothesis_shape: "
                             << hypothesis_shape_shape_val << " and truth_shape: " << truth_shape_shape[kIndex0] << ".";
  }

  const std::set<TypePtr> valid_types = {kInt64};
  auto hypothesis_shape_abs = input_args[kIndex2];
  auto truth_shape_abs = input_args[kIndex5];
  auto hypothesis_shape_type = hypothesis_shape_abs->BuildType();
  MS_EXCEPTION_IF_NULL(hypothesis_shape_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("hypothesis_shape", hypothesis_shape_type, valid_types, prim_name);
  auto truth_shape_type = truth_shape_abs->BuildType();
  MS_EXCEPTION_IF_NULL(truth_shape_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("truth_shape", truth_shape_type, valid_types, prim_name);

  auto hypothesis_shape_value = hypothesis_shape_abs->BuildValue();
  MS_EXCEPTION_IF_NULL(hypothesis_shape_value);
  if (!hypothesis_shape_value->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "hypothesis_shape_value should be a Tensor but got: " << hypothesis_shape_value->ToString()
                      << ".";
  }
  auto hypothesis_shape_ptr = static_cast<int64_t *>(hypothesis_shape_value->cast<tensor::TensorPtr>()->data_c());
  auto truth_shape_value = truth_shape_abs->BuildValue();
  MS_EXCEPTION_IF_NULL(truth_shape_value);
  if (!truth_shape_value->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "truth_shape_value should be a Tensor but got: " << truth_shape_value->ToString() << ".";
  }
  auto truth_shape_ptr = static_cast<int64_t *>(truth_shape_value->cast<tensor::TensorPtr>()->data_c());
  ShapeVector infer_shape;
  for (auto i = 0; i < hypothesis_shape_shape[kIndex0] - 1; ++i) {
    infer_shape.push_back(std::max(*hypothesis_shape_ptr, *truth_shape_ptr));
    hypothesis_shape_ptr++;
    truth_shape_ptr++;
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr EditDistanceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const std::set<TypePtr> indices_valid_types = {kInt64};
  const std::set<TypePtr> values_valid_types = {kNumber};

  auto hypothesis_indices_type = input_args[kIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(hypothesis_indices_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("hypothesis_indices", hypothesis_indices_type, indices_valid_types,
                                                   prim_name);
  auto hypothesis_values_type = input_args[kIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(hypothesis_values_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("hypothesis_values", hypothesis_values_type, values_valid_types,
                                                   prim_name);
  auto truth_indices_type = input_args[kIndex3]->BuildType();
  MS_EXCEPTION_IF_NULL(truth_indices_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("truth_indices", truth_indices_type, indices_valid_types, prim_name);
  auto truth_values_type = input_args[kIndex4]->BuildType();
  MS_EXCEPTION_IF_NULL(truth_values_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("truth_values", truth_values_type, values_valid_types, prim_name);
  return kFloat32;
}
}  // namespace

MIND_API_OPERATOR_IMPL(EditDistance, BaseOperator);
AbstractBasePtr EditDistanceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_shape = EditDistanceInferShape(primitive, input_args);
  auto infer_type = EditDistanceInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGEditDistanceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EditDistanceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EditDistanceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EditDistanceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EditDistance, prim::kPrimEditDistance, AGEditDistanceInfer, false);
}  // namespace ops
}  // namespace mindspore
