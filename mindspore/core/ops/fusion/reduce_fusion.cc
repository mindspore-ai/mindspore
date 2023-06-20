/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/fusion/reduce_fusion.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/lite_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReduceFusion, Reduce);
void ReduceFusion::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }

void ReduceFusion::set_mode(const ReduceMode mode) {
  int64_t swi = mode;
  (void)this->AddAttr(kMode, api::MakeValue(swi));
}

void ReduceFusion::set_reduce_to_end(const bool reduce_to_end) {
  (void)this->AddAttr(kReduceToEnd, api::MakeValue(reduce_to_end));
}

void ReduceFusion::set_coeff(const float coeff) { (void)this->AddAttr(kCoeff, api::MakeValue(coeff)); }

bool ReduceFusion::get_keep_dims() const {
  auto value_ptr = GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

ReduceMode ReduceFusion::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ReduceMode(GetValue<int64_t>(value_ptr));
}

bool ReduceFusion::get_reduce_to_end() const {
  auto value_ptr = GetAttr(kReduceToEnd);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

float ReduceFusion::get_coeff() const {
  auto value_ptr = GetAttr(kCoeff);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void ReduceFusion::Init(const bool keep_dims, const ReduceMode mode, const bool reduce_to_end, const float coeff) {
  this->set_keep_dims(keep_dims);
  this->set_mode(mode);
  this->set_reduce_to_end(reduce_to_end);
  this->set_coeff(coeff);
}

namespace {
abstract::ShapePtr ReduceFusionInferShape(const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(primitive->name(), input_args, 0);
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto x_shape = shape_ptr->shape();

  auto keep_dims_value_ptr = primitive->GetAttr(kKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims_value_ptr);
  if (!keep_dims_value_ptr->isa<BoolImm>()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'keep_dims' must be Bool.";
  }
  bool keep_dims = GetValue<bool>(keep_dims_value_ptr);

  std::vector<int64_t> axis_value;
  int64_t axis_shape = 1;
  bool axis_is_dynamic = CheckAndGetAxisValue(input_args, &axis_value, &axis_shape, primitive);
  auto reduce_to_end_ptr = primitive->GetAttr(kReduceToEnd);
  bool reduce_to_end = reduce_to_end_ptr && GetValue<bool>(reduce_to_end_ptr);
  if (reduce_to_end) {
    if (axis_value.size() != 1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', if 'reduce_to_end' is Bool, the axis num should 1";
    }
    int64_t begin_axis = axis_value[0];
    for (int64_t i = begin_axis + 1; i < SizeToLong(x_shape.size()); ++i) {
      axis_value.push_back(i);
    }
    axis_shape = SizeToLong(x_shape.size()) - begin_axis;
    keep_dims = false;
  }

  ShapeVector out_shape = {};
  constexpr int dynamic_rank_value = -2;
  if (IsDynamicRank(x_shape)) {
    if (axis_shape == 0 && !keep_dims) {
      return std::make_shared<abstract::Shape>(out_shape);
    }
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  if (axis_shape == -1 && !keep_dims) {
    out_shape.push_back(dynamic_rank_value);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  ReduceFuncCheckAxisInferImpl(primitive, &axis_value, x_shape.size());

  if (axis_is_dynamic) {
    out_shape = ReduceFuncCalShapeAxisDyn(x_shape, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }
  out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, axis_value, keep_dims);
  return std::make_shared<abstract::Shape>(out_shape);
}
}  // namespace

class ReduceFusionInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const int64_t input_num = 1;
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                       primitive->name());
    return ReduceFusionInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    return x_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceFusion, prim::kPrimReduceFusion, ReduceFusionInfer, false);
}  // namespace ops
}  // namespace mindspore
