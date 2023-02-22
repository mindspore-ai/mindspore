/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/histogram_fixed_width.h"

#include <iostream>
#include <memory>
#include <set>

#include "mindapi/ir/type.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr HistogramFixedWidthInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  int32_t nbins = static_cast<int32_t>(GetValue<int64_t>(primitive->GetAttr(kNbins)));
  ShapeVector out_shape = std::vector<int64_t>(1, nbins);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr HistogramFixedWidthInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  const std::set<TypePtr> valid_types = {kInt32, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("range", input_args[1]->BuildType(), valid_types, prim_name);
  TypePtr y_dtype = kInt32;
  return y_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(HistogramFixedWidth, BaseOperator);

void HistogramFixedWidth::set_nbins(const int32_t nbins) {
  (void)CheckAndConvertUtils::CheckInteger(kNbins, nbins, kGreaterEqual, 1, this->name());
  (void)this->AddAttr(kNbins, api::MakeValue(nbins));
}

void HistogramFixedWidth::set_dtype(const TypeId dtype) { (void)this->AddAttr("dtype", api::Type::GetType(dtype)); }

int32_t HistogramFixedWidth::get_nbins() const { return static_cast<int32_t>(GetValue<int64_t>(GetAttr(kNbins))); }

TypeId HistogramFixedWidth::get_dtype() const {
  return GetAttr("dtype")->cast<api::TensorTypePtr>()->element()->type_id();
}

void HistogramFixedWidth::Init(const int32_t nbins, const TypeId dtype) {
  std::cout << nbins;
  this->set_nbins(nbins);
  this->set_dtype(dtype);
}

AbstractBasePtr HistogramFixedWidthInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputsNum, primitive->name());
  auto infer_type = HistogramFixedWidthInferType(primitive, input_args);
  auto infer_shape = HistogramFixedWidthInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGHistogramFixedWidthInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return HistogramFixedWidthInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return HistogramFixedWidthInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return HistogramFixedWidthInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(HistogramFixedWidth, prim::kPrimHistogramFixedWidth, AGHistogramFixedWidthInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
