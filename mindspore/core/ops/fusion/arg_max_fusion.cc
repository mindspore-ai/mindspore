/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/fusion/arg_max_fusion.h"

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/fusion/arg_min_fusion.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ArgMaxFusion, Argmax);
void ArgMaxFusion::Init(const bool keep_dims, const bool out_max_value, const int64_t top_k, const int64_t axis) {
  set_axis(axis);
  set_keep_dims(keep_dims);
  set_out_max_value(out_max_value);
  set_top_k(top_k);
}

void ArgMaxFusion::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }
void ArgMaxFusion::set_out_max_value(const bool out_max_value) {
  (void)this->AddAttr(kOutMaxValue, api::MakeValue(out_max_value));
}
void ArgMaxFusion::set_top_k(const int64_t top_k) { (void)this->AddAttr(kTopK, api::MakeValue(top_k)); }

bool ArgMaxFusion::get_keep_dims() const {
  auto keep_dims = GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims);
  return GetValue<bool>(keep_dims);
}
bool ArgMaxFusion::get_out_max_value() const {
  auto out_maxv = GetAttr(kOutMaxValue);
  MS_EXCEPTION_IF_NULL(out_maxv);
  return GetValue<bool>(out_maxv);
}
int64_t ArgMaxFusion::get_top_k() const {
  auto topk = GetAttr(kTopK);
  MS_EXCEPTION_IF_NULL(topk);
  return GetValue<int64_t>(topk);
}

namespace {
BaseShapePtr ArgFusionInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kArgMaxInputNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kArgMaxInputNum,
                                           prim_name);
  auto x_base_shape = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_base_shape);
  auto x_shape = x_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto shape_vector = x_shape->shape();
  if (IsDynamic(shape_vector)) {
    return x_shape;
  }
  // Get rank of shape
  auto x_rank = shape_vector.size();
  // Get and calculate the real positive axis.
  auto axis_value = primitive->GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  auto axis = GetValue<int64_t>(axis_value);
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeLeft, {-1 * x_rank, x_rank}, prim_name);
  axis = axis < 0 ? axis + SizeToLong(x_rank) : axis;

  auto topk_value = primitive->GetAttr(kTopK);
  int64_t topk = topk_value == nullptr ? 1 : GetValue<int64_t>(topk_value);
  auto keep_dims_value = primitive->GetAttr(kKeepDims);
  bool keep_dims = keep_dims_value != nullptr && GetValue<bool>(keep_dims_value);
  auto out_shape_vector = shape_vector;
  if (topk == 1 && !keep_dims) {
    (void)out_shape_vector.erase(out_shape_vector.cbegin() + axis);
  } else {
    out_shape_vector[LongToSize(axis)] = topk;
  }

  auto out_max_value = primitive->GetAttr(kOutMaxValue);
  bool out_max = out_max_value != nullptr && GetValue<bool>(out_max_value);
  auto out_shape_ptr = std::make_shared<abstract::Shape>(out_shape_vector);
  if (out_max) {
    // two outputs: max indices, max value
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  } else {
    return out_shape_ptr;
  }
}

TypePtr ArgFusionInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kArgMaxInputNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kArgMaxInputNum,
                                           prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto x_type = x->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                            << ".";
  }

  auto out_max_value = primitive->GetAttr(kOutMaxValue);
  bool out_max = out_max_value != nullptr && GetValue<bool>(out_max_value);
  if (out_max) {
    // two outputs: max indices, max value
    return std::make_shared<Tuple>(std::vector<TypePtr>{kInt32, x_type});
  } else {
    return std::make_shared<TensorType>(kInt32);
  }
}
}  // namespace

class MIND_API ArgFusionInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgFusionInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ArgFusionInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ArgMaxFusion, prim::kPrimArgMaxFusion, ArgFusionInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ArgMinFusion, prim::kPrimArgMinFusion, ArgFusionInfer, false);
}  // namespace ops
}  // namespace mindspore
