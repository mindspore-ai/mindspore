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

#include "ops/avg_pool_v1.h"

#include <memory>
#include <set>
#include <vector>
#include <cmath>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/format.h"
#include "mindapi/base/types.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kFormatNCHWIndN = 0;
constexpr int64_t kFormatNCHWIndC = 1;
constexpr int64_t kFormatNCHWIndH = 2;
constexpr int64_t kFormatNCHWIndW = 3;
constexpr int64_t kFormatNHWCIndN = 0;
constexpr int64_t kFormatNHWCIndH = 1;
constexpr int64_t kFormatNHWCIndW = 2;
constexpr int64_t kFormatNHWCIndC = 3;
abstract::ShapePtr AvgPoolV1InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  int64_t format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  const int64_t x_size = 4;
  const int64_t attr_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, x_size, op_name);

  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  auto pad_mode = PadMode(GetValue<int64_t>(primitive->GetAttr(kPadMode)));
  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  (void)CheckAndConvertUtils::CheckInteger("kernel size", SizeToLong(kernel_size.size()), kEqual, attr_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("strides size", SizeToLong(strides.size()), kEqual, attr_size, op_name);

  int64_t batch = 0, in_h = 0, in_w = 0, channel = 0;
  int64_t kernel_h = 0, kernel_w = 0;
  int64_t stride_h = 0, stride_w = 0;

  if (format == NHWC) {
    batch = in_shape[kFormatNHWCIndN];
    channel = in_shape[kFormatNHWCIndC];
    in_h = in_shape[kFormatNHWCIndH];
    in_w = in_shape[kFormatNHWCIndW];
    kernel_h = kernel_size[kFormatNHWCIndH];
    kernel_w = kernel_size[kFormatNHWCIndW];
    stride_h = strides[kFormatNHWCIndH];
    stride_w = strides[kFormatNHWCIndW];
  } else if (format == NCHW) {
    batch = in_shape[kFormatNCHWIndN];
    channel = in_shape[kFormatNCHWIndC];
    in_h = in_shape[kFormatNCHWIndH];
    in_w = in_shape[kFormatNCHWIndW];
    kernel_h = kernel_size[kFormatNCHWIndH];
    kernel_w = kernel_size[kFormatNCHWIndW];
    stride_h = strides[kFormatNCHWIndH];
    stride_w = strides[kFormatNCHWIndW];
  }
  int64_t out_h = abstract::Shape::kShapeDimAny;
  int64_t out_w = abstract::Shape::kShapeDimAny;
  if (pad_mode == VALID) {
    out_h = static_cast<int64_t>(std::ceil((in_h - (kernel_h - 1)) / static_cast<float>(stride_h)));
    out_w = static_cast<int64_t>(std::ceil((in_w - (kernel_w - 1)) / static_cast<float>(stride_w)));
  } else if (pad_mode == SAME) {
    out_h = static_cast<int64_t>(std::ceil(in_h / static_cast<float>(stride_h)));
    out_w = static_cast<int64_t>(std::ceil(in_w / static_cast<float>(stride_w)));
  }
  std::vector<int64_t> out_shape = {batch, channel, out_h, out_w};

  // Process attr mapping problems from mindspore to ai_cpu
  // kernel_size -> ksize
  // pad_mode -> padding
  if (format == NHWC) {
    std::vector<int64_t> ksize_NHWC = {kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3]};
    (void)primitive->AddAttr("ksize", MakeValue(ksize_NHWC));
    (void)primitive->AddAttr("data_format", MakeValue("NHWC"));
  } else if (format == NCHW) {
    std::vector<int64_t> ksize_NCHW = {kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3]};
    (void)primitive->AddAttr("ksize", MakeValue(ksize_NCHW));
    (void)primitive->AddAttr("data_format", MakeValue("NCHW"));
  }
  if (pad_mode == VALID) {
    (void)primitive->AddAttr("padding", MakeValue("VALID"));
  } else if (pad_mode == SAME) {
    (void)primitive->AddAttr("padding", MakeValue("SAME"));
  }

  if (NHWC == format) {
    out_shape = {batch, out_h, out_w, channel};
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr AvgPoolV1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> avgpool_v1_valid_types = {kFloat16, kFloat32, kFloat64};
  auto input_type = input_args[kInputIndex0]->BuildType();
  auto inferred_type = CheckAndConvertUtils::CheckTypeValid("value", input_type, avgpool_v1_valid_types, name);
  return inferred_type;
}
}  // namespace

AbstractBasePtr AvgPoolV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto avgpool_v1_infer_type = AvgPoolV1InferType(primitive, input_args);
  auto avgpool_v1_infer_shape = AvgPoolV1InferShape(primitive, input_args)->shape();
  return std::make_shared<abstract::AbstractTensor>(avgpool_v1_infer_type, avgpool_v1_infer_shape);
}
MIND_API_OPERATOR_IMPL(AvgPoolV1, BaseOperator);

// AG means auto generated
class MIND_API AGAvgPoolV1Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolV1InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolV1InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolV1Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AvgPoolV1, prim::kPrimAvgPoolV1, AGAvgPoolV1Infer, false);
}  // namespace ops
}  // namespace mindspore
