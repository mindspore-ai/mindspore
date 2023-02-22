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

#include "ops/grad/max_pool_3d_grad_grad.h"

#include <algorithm>
#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int kHalfNum = 2;
void CalculatePad(const int64_t &shape, const int64_t &kernel, const int64_t &stride, const std::string &name,
                  int64_t *pad1, int64_t *pad2) {
  (void)CheckAndConvertUtils::CheckInteger("strides size", stride, kGreaterThan, 0, name);
  int64_t tail = shape % stride;
  int64_t pad = tail > 0 ? (kernel - tail) : (kernel - stride);
  pad = IntToLong(std::max(LongToInt(pad), 0));
  *pad1 = pad / kHalfNum;
  *pad2 = pad / kHalfNum + pad % kHalfNum;
}

abstract::ShapePtr MaxPool3DGradGradInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_dim = 5;
  auto origin_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("origin input shape size", SizeToLong(origin_input_shape.size()), kEqual,
                                           input_dim, primitive->name());

  auto origin_output_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("origin output shape size", SizeToLong(origin_output_shape.size()), kEqual,
                                           input_dim, primitive->name());

  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("grad shape size", SizeToLong(grad_shape.size()), kEqual, input_dim,
                                           primitive->name());

  CheckAndConvertUtils::Check("grad_shape", origin_input_shape, kEqual, grad_shape, primitive->name(), ValueError);

  std::string pad_mode = GetValue<std::string>(primitive->GetAttr(kPadMode));
  std::vector<int64_t> kernels = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  (void)CheckAndConvertUtils::CheckInteger("kernels size", SizeToLong(kernels.size()), kEqual, input_dim,
                                           primitive->name());
  std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  (void)CheckAndConvertUtils::CheckInteger("strides size", SizeToLong(strides.size()), kEqual, input_dim,
                                           primitive->name());
  std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
  if (pad_mode == "SAME") {
    CalculatePad(origin_input_shape[kDim2], kernels[kDim2], strides[kDim2], primitive->name(), &pads[kDim0],
                 &pads[kDim1]);
    CalculatePad(origin_input_shape[kDim3], kernels[kDim3], strides[kDim3], primitive->name(), &pads[kDim2],
                 &pads[kDim3]);
    CalculatePad(origin_input_shape[kDim4], kernels[kDim4], strides[kDim4], primitive->name(), &pads[kDim4],
                 &pads[kDim5]);
  }
  for (size_t i = 0; i < pads.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger("element of pad_list ", pads[i], kGreaterEqual, 0, primitive->name());
  }
  (void)primitive->SetAttrs({{kPadList, MakeValue(pads)}});
  return std::make_shared<abstract::Shape>(origin_output_shape);
}

TypePtr MaxPool3DGradGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)types.emplace("origin_input", input_args[0]->BuildType());
  (void)types.emplace("origin_output", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("grad", input_args[kInputIndex2]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

AbstractBasePtr MaxPool3DGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPool3DGradGradInferType(primitive, input_args);
  auto infer_shape = MaxPool3DGradGradInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}

MIND_API_OPERATOR_IMPL(MaxPool3DGradGrad, MaxPoolGradGrad);

// AG means auto generated
class MIND_API AGMaxPool3DGradGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DGradGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPool3DGradGrad, prim::kPrimMaxPool3DGradGrad, AGMaxPool3DGradGradInfer, false);
}  // namespace ops
}  // namespace mindspore
