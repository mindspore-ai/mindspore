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

#include "ops/unique_with_pad.h"

#include <functional>
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <numeric>

#include "abstract/abstract_value.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kUniqueWithPadInputsNum = 2;
constexpr size_t kUniqueWithPadOutputsNum = 2;
constexpr int64_t kUniqueWithPadEmptyDim = 0;

void UniqueWithPadCheckEmptyTensor(const std::string &prim_name, const std::vector<ShapeVector> &shapes) {
  for (auto &shape : shapes) {
    if (IsDynamicRank(shape)) {
      continue;
    }
    auto is_empty_tensor =
      std::any_of(shape.begin(), shape.end(), [](int64_t x) -> bool { return x == kUniqueWithPadEmptyDim; });
    if (is_empty_tensor) {
      MS_EXCEPTION(ValueError) << "For [" << prim_name
                               << "], empty tensor(at least one dimension is zero) is not supported.";
    }
  }
  return;
}

abstract::TupleShapePtr UniqueWithPadInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto pad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto is_dynamic = IsDynamic(x_shape) || IsDynamic(pad_shape);

  UniqueWithPadCheckEmptyTensor(prim_name, {x_shape, pad_shape});

  size_t batch_rank = 0;
  if (primitive->HasAttr(ops::kBatchRank)) {
    auto value_ptr = primitive->GetAttr(ops::kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("input_shape_size", x_shape.size(), kEqual, batch_rank + 1, prim_name);
  }

  constexpr int64_t kNumZero = 0;
  if (!is_dynamic && batch_rank != kNumZero) {
    auto pad_num = std::accumulate(pad_shape.begin(), pad_shape.end(), 1, std::multiplies<int64_t>());
    auto input_batch = std::accumulate(x_shape.begin(), x_shape.begin() + batch_rank, 1, std::multiplies<int64_t>());
    (void)CheckAndConvertUtils::CheckInteger("elements num of input 'pad'", pad_num, kEqual, input_batch, prim_name);
  }
  auto x_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape_ptr, x_shape_ptr});
}

TuplePtr UniqueWithPadInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[0]->BuildType();

  std::set<TypePtr> valid_types = {kInt32, kInt64, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);

  TypePtr y_type = x_type;
  TypePtr idx_type = kInt32;

  abstract::AbstractTensorPtr x_ptr = input_args.at(kInputIndex0)->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_ptr->element());
  MS_EXCEPTION_IF_NULL(x_ptr->element()->GetTypeTrack());
  if (x_ptr->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    idx_type = kInt64;
  }

  return std::make_shared<Tuple>(std::vector<TypePtr>{y_type, idx_type});
}
}  // namespace

AbstractBasePtr UniqueWithPadInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kUniqueWithPadInputsNum, prim_name);
  auto infer_type = UniqueWithPadInferType(primitive, input_args);
  auto infer_shape = UniqueWithPadInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(UniqueWithPad, BaseOperator);

// AG means auto generated
class MIND_API AGUniqueWithPadInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueWithPadInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueWithPadInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return UniqueWithPadInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniqueWithPad, prim::kPrimUniqueWithPad, AGUniqueWithPadInfer, false);
}  // namespace ops
}  // namespace mindspore
