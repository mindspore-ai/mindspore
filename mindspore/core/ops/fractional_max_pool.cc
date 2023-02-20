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

#include "ops/fractional_max_pool.h"

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <cmath>

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
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kPoolingRatioDims = 4;
abstract::TupleShapePtr FractionalMaxPoolInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("input_rank", SizeToLong(in_shape.size()), kEqual, x_rank, op_name);
  auto pooling_ratio = GetValue<std::vector<float>>(primitive->GetAttr(kPoolingRatio));
  if (pooling_ratio.size() != kPoolingRatioDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the size of parameter 'pooling_ratio' must be 4, but got "
                             << std::to_string(pooling_ratio.size()) << ".";
  }
  if (!common::IsFloatEqual(pooling_ratio[kInputIndex0], 1.0)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the first element of parameter 'pooling_ratio' must be 1.0, but got "
                             << std::to_string(pooling_ratio[0]) << ".";
  }
  if (pooling_ratio[kInputIndex1] < 1.0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the second element of pooling ratio must be greater than or equal to 1.0, but got "
                             << std::to_string(pooling_ratio[kInputIndex1]) << ".";
  }
  if (pooling_ratio[kInputIndex2] < 1.0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the third element of pooling ratio must be greater than or equal to 1.0, but got "
                             << std::to_string(pooling_ratio[kInputIndex2]) << ".";
  }
  if (!common::IsFloatEqual(pooling_ratio[kInputIndex3], 1.0)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the forth element of parameter 'pooling_ratio' must be 1.0, but got "
                             << std::to_string(pooling_ratio[kInputIndex3]) << ".";
  }
  std::vector<int64_t> out_shape(x_rank);
  for (int i = 0; i < x_rank; ++i) {
    out_shape[IntToSize(i)] = static_cast<int64_t>(std::floor(in_shape[IntToSize(i)] / pooling_ratio[IntToSize(i)]));
  }

  int64_t row = out_shape[kInputIndex1] + 1;
  int64_t col = out_shape[kInputIndex2] + 1;
  std::vector<int64_t> row_dim = {row};
  std::vector<int64_t> col_dim = {col};
  abstract::ShapePtr output0_shape = std::make_shared<abstract::Shape>(out_shape);
  abstract::ShapePtr output1_shape = std::make_shared<abstract::Shape>(row_dim);
  abstract::ShapePtr output2_shape = std::make_shared<abstract::Shape>(col_dim);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output0_shape, output1_shape, output2_shape});
}

TuplePtr FractionalMaxPoolInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input", x_dtype, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, kInt64, kInt64});
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPool, BaseOperator);
AbstractBasePtr FractionalMaxPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = FractionalMaxPoolInferType(primitive, input_args);
  auto infer_shape = FractionalMaxPoolInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void FractionalMaxPool::Init(const std::vector<float> pooling_ratio, const bool pseudo_random, const bool overlapping,
                             const bool deterministic, const int64_t seed, const int64_t seed2) {
  set_pooling_ratio(pooling_ratio);
  set_pseudo_random(pseudo_random);
  set_overlapping(overlapping);
  set_deterministic(deterministic);
  set_seed(seed);
  set_seed2(seed2);
}

void FractionalMaxPool::set_pooling_ratio(const std::vector<float> pooling_ratio) {
  (void)this->AddAttr("pooling_ratio", api::MakeValue(pooling_ratio));
}

void FractionalMaxPool::set_pseudo_random(const bool pseudo_random) {
  (void)this->AddAttr("pseudo_random", api::MakeValue(pseudo_random));
}

void FractionalMaxPool::set_overlapping(const bool overlapping) {
  (void)this->AddAttr("overlapping", api::MakeValue(overlapping));
}

void FractionalMaxPool::set_deterministic(const bool deterministic) {
  (void)this->AddAttr("deterministic", api::MakeValue(deterministic));
}

void FractionalMaxPool::set_seed(const int64_t seed) { (void)this->AddAttr("seed", api::MakeValue(seed)); }

void FractionalMaxPool::set_seed2(const int64_t seed2) { (void)this->AddAttr("seed2", api::MakeValue(seed2)); }

std::vector<float> FractionalMaxPool::get_pooling_ratio() const {
  auto value_ptr = GetAttr("pooling_ratio");
  return GetValue<std::vector<float>>(value_ptr);
}

bool FractionalMaxPool::get_pseudo_random() const {
  auto value_ptr = GetAttr("pseudo_random");
  return GetValue<bool>(value_ptr);
}

bool FractionalMaxPool::get_overlapping() const {
  auto value_ptr = GetAttr("overlapping");
  return GetValue<bool>(value_ptr);
}

bool FractionalMaxPool::get_deterministic() const {
  auto value_ptr = GetAttr("deterministic");
  return GetValue<bool>(value_ptr);
}

int64_t FractionalMaxPool::get_seed() const {
  auto value_ptr = GetAttr("seed");
  return GetValue<int64_t>(value_ptr);
}

int64_t FractionalMaxPool::get_seed2() const {
  auto value_ptr = GetAttr("seed2");
  return GetValue<int64_t>(value_ptr);
}

// AG means auto generated
class MIND_API AGFractionalMaxPoolInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalMaxPool, prim::kPrimFractionalMaxPool, AGFractionalMaxPoolInfer, false);
}  // namespace ops
}  // namespace mindspore
