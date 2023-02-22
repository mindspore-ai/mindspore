/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "ops/pad.h"

#include <algorithm>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
constexpr size_t kPaddingsItemSize = 2;

void CheckPaddings(size_t x_rank, const std::vector<std::vector<int64_t>> &paddings) {
  if (paddings.size() != x_rank) {
    MS_EXCEPTION(ValueError) << "For 'Pad', the length of 'paddings' must be equal to the rank of 'input_x', but got "
                             << paddings.size() << " vs " << x_rank;
  }
  for (size_t i = 0; i < paddings.size(); ++i) {
    if (paddings[i].size() != kPaddingsItemSize) {
      MS_EXCEPTION(ValueError) << "For 'Pad', the shape of 'paddings' must be (" << x_rank << ", " << kPaddingsItemSize
                               << "), but got paddings.shape[1] = " << paddings[i].size();
    }
    if (paddings[i][0] < 0 || paddings[i][1] < 0) {
      MS_EXCEPTION(ValueError) << "For 'Pad', all elements of 'paddings' must be >= 0, but got paddings[" << i
                               << "][0] = " << paddings[i][0] << ", paddings[" << i << "][1] = " << paddings[i][1];
    }
  }
}

abstract::ShapePtr PadInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  ShapeVector out_shape;
  auto prim_name = prim->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  auto x_shape = x_shape_ptr->shape();
  // Rank of input x is unknown
  if (std::any_of(x_shape.begin(), x_shape.end(), [](int64_t val) { return val < -1; })) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  // Rank of input x is known, but may has unknown dimension on some axes
  auto x_rank = x_shape.size();
  std::vector<std::vector<int64_t>> paddings = GetValue<std::vector<std::vector<int64_t>>>(prim->GetAttr(kPaddings));
  CheckPaddings(x_rank, paddings);
  for (size_t i = 0; i < x_rank; ++i) {
    auto sh = x_shape[i];
    if (x_shape[i] >= 0) {
      sh += (paddings[i][0] + paddings[i][1]);
    }
    out_shape.push_back(sh);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[kInputIndex0]->BuildType(), {kTensorType},
                                             prim_name);
}
}  // namespace

void Pad::Init(const std::vector<std::vector<int64_t>> &paddings) { this->set_paddings(paddings); }
void Pad::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, api::MakeValue(paddings));
}
std::vector<std::vector<int64_t>> Pad::get_paddings() const {
  return GetValue<std::vector<std::vector<int64_t>>>(GetAttr(kPaddings));
}

MIND_API_OPERATOR_IMPL(Pad, BaseOperator);
abstract::AbstractBasePtr PadInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PadInferType(primitive, input_args);
  auto infer_shape = PadInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGPadInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PadInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PadInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PadInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Pad, prim::kPrimPad, AGPadInfer, false);
}  // namespace ops
}  // namespace mindspore
