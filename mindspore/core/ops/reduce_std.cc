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

#include "ops/reduce_std.h"

#include <set>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
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
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ReduceStdInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto input_rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("input_rank", input_rank, kGreaterEqual, 1, prim_name);
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  auto temp_shape = input_shape;
  if (axis.size() == 0) {
    for (size_t i = 0; i < input_shape.size(); i++) {
      axis.push_back(i);
    }
  } else {
    for (size_t i = 0; i < axis.size(); ++i) {
      CheckAndConvertUtils::CheckInRange("axis value", axis[i], kIncludeLeft, {-input_rank, input_rank}, prim_name);
      if (axis[i] < 0) {
        axis[i] += input_rank;
      }
    }
  }
  for (size_t i = 0; i < axis.size(); ++i) {
    if (!keep_dims) {
      temp_shape[LongToSize(axis[i])] = -1;
    } else {
      temp_shape[LongToSize(axis[i])] = 1;
    }
  }
  if (!keep_dims) {
    for (std::vector<int64_t>::iterator iter = temp_shape.begin(); iter != temp_shape.end(); ++iter) {
      if (*iter == -1) {
        iter = temp_shape.erase(iter);
        iter -= 1;
      }
    }
  }
  abstract::ShapePtr output_shape = std::make_shared<abstract::Shape>(temp_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output_shape, output_shape});
}

TuplePtr ReduceStdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  auto type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", type, valid_types, name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(ReduceStd, BaseOperator);

void ReduceStd::Init(bool unbiased) { set_unbiased(unbiased); }

void ReduceStd::Init(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

void ReduceStd::Init(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

void ReduceStd::set_unbiased(bool unbiased) { (void)AddAttr(kUnbiased, api::MakeValue(unbiased)); }

bool ReduceStd::get_unbiased() const {
  auto value_ptr = GetAttr(kUnbiased);
  return GetValue<bool>(value_ptr);
}

void ReduceStd::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> ReduceStd::get_axis() const {
  std::vector<int64_t> axis;
  auto axis_value = GetAttr(kAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  if (axis_value->isa<api::ValueSequence>()) {
    axis = api::GetValue<std::vector<int64_t>>(axis_value);
  } else if (axis_value->isa<api::Int64Imm>()) {
    (void)axis.emplace_back(api::GetValue<int64_t>(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For ReduceStd, the type of attribute `axis` is invalid.";
  }
  return axis;
}

AbstractBasePtr ReduceStdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ReduceStdInferType(primitive, input_args);
  auto infer_shape = ReduceStdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGReduceStdInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ReduceStdInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ReduceStdInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ReduceStdInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceStd, prim::kPrimReduceStd, AGReduceStdInfer, false);
}  // namespace ops
}  // namespace mindspore
