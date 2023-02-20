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

#include "ops/grad/sync_batch_norm_grad.h"

#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/primitive.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSyncBatchNormGradInputSize = 5;
TuplePtr SyncBatchNormGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSyncBatchNormGradInputSize, prim_name);
  auto x_dtype = input_args[1]->BuildType();
  auto scale_dtype = input_args[2]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, scale_dtype, scale_dtype});
}

abstract::TupleShapePtr SyncBatchNormGradInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual,
                                           kSyncBatchNormGradInputSize, prim_name);
  auto y_backprop_shape_ptr = input_args[0]->BuildShape();
  auto x_shape_ptr = input_args[1]->BuildShape();
  auto scale_shape_ptr = input_args[2]->BuildShape();
  auto y_backprop_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(y_backprop_shape_ptr)[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  // y_backprop and x must have same shape
  CheckAndConvertUtils::Check("shape of y_backprop ", y_backprop_shape, kEqual, x_shape, prim_name);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{x_shape_ptr, scale_shape_ptr, scale_shape_ptr});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SyncBatchNormGrad, BaseOperator);
AbstractBasePtr SyncBatchNormGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(SyncBatchNormGradInferShape(primitive, input_args),
                                SyncBatchNormGradInferType(primitive, input_args));
}

void SyncBatchNormGrad::Init(const float epsilon, const std::string group, const int64_t device_num) {
  set_epsilon(epsilon);
  set_group(group);
  set_device_num(device_num);
}

void SyncBatchNormGrad::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon));
}

void SyncBatchNormGrad::set_group(const std::string group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

void SyncBatchNormGrad::set_device_num(const int64_t device_num) {
  (void)this->AddAttr(kDeviceNum, api::MakeValue(device_num));
}

// AG means auto generated
class MIND_API AGSyncBatchNormGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SyncBatchNormGrad, prim::kPrimSyncBatchNormGrad, AGSyncBatchNormGradInfer, false);
}  // namespace ops
}  // namespace mindspore
