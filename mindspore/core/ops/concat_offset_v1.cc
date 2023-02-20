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

#include <map>
#include <set>
#include <string>
#include <memory>

#include "ops/concat_offset_v1.h"
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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ConcatOffsetV1InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kAxisRank = 0;
  const int64_t kXTensorRankOrElemNum = 1;
  const int64_t kXTensorNum = 2;
  // check axis shape
  auto axis = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(axis);
  auto axis_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(axis->BuildShape())[kShape];
  auto axis_rank = axis_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("input axis shape rank", SizeToLong(axis_rank), kEqual, kAxisRank,
                                           prim_name);
  // check x shape and infer y shape
  auto tensors = input_args[1]->isa<abstract::AbstractTuple>()
                   ? input_args[1]->cast<abstract::AbstractTuplePtr>()->elements()
                   : input_args[1]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("input x tensor num", SizeToLong(tensors.size()), kGreaterEqual, kXTensorNum,
                                           prim_name);
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(tensors.size())));
  auto tensor0 = tensors[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor0);
  auto tensor0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(tensor0->BuildShape())[kShape];
  auto tensor0_rank = tensor0_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("input x tensor0 shape rank", SizeToLong(tensor0_rank), kEqual,
                                           kXTensorRankOrElemNum, prim_name);
  auto tensor0_numelement = tensor0_shape[0];
  (void)CheckAndConvertUtils::CheckInteger("element num in input x tensor0", SizeToLong(tensor0_numelement),
                                           kGreaterEqual, kXTensorRankOrElemNum, prim_name);
  for (size_t i = 1; i < tensors.size(); ++i) {
    std::string tensori = "tensor" + std::to_string(i);
    auto tensori_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(tensors[i]->BuildShape())[kShape];
    auto tensori_rank = tensori_shape.size();
    (void)CheckAndConvertUtils::CheckInteger("input x " + tensori + " shape rank", SizeToLong(tensori_rank), kEqual,
                                             kXTensorRankOrElemNum, prim_name);
    auto tensori_numelement = tensori_shape[0];
    if (tensori_numelement != tensor0_numelement) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', tensor" << i << " element num in input x should be "
                               << tensor0_numelement << ", which is element num of tensor0"
                               << ", but got " << tensori_numelement << ".";
    }
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>(tensors.size(), tensor0->BuildShape()));
}

TuplePtr ConcatOffsetV1InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt32};
  // check axis type
  (void)CheckAndConvertUtils::CheckTensorTypeValid("axis", input_args[0]->BuildType(), valid_types, prim_name);
  // check x type and infer y type
  if (!input_args[1]->isa<abstract::AbstractTuple>() && !input_args[1]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "For 'ConcatOffsetV1', the input x must be list or tuple of tensors.";
  }
  auto tensors = input_args[1]->isa<abstract::AbstractTuple>()
                   ? input_args[1]->cast<abstract::AbstractTuplePtr>()->elements()
                   : input_args[1]->cast<abstract::AbstractListPtr>()->elements();
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < tensors.size(); ++i) {
    std::string tensori = "tensor" + std::to_string(i);
    (void)types.emplace(tensori, tensors[i]->BuildType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>(tensors.size(), tensors[0]->BuildType()));
}
}  // namespace

AbstractBasePtr ConcatOffsetV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = ConcatOffsetV1InferType(primitive, input_args);
  auto infer_shape = ConcatOffsetV1InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ConcatOffsetV1, BaseOperator);

// AG means auto generated
class MIND_API AGConcatOffsetV1Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatOffsetV1InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatOffsetV1InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatOffsetV1Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ConcatOffsetV1, prim::kPrimConcatOffsetV1, AGConcatOffsetV1Infer, false);
}  // namespace ops
}  // namespace mindspore
