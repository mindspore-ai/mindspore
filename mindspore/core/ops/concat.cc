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

#include <map>
#include <string>
#include "ops/concat.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ConcatInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  const int64_t kOneNum = 1;
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == kOneNum) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           prim_name);
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(elements.size())));
  (void)primitive->AddAttr("inputNums", MakeValue(SizeToLong(elements.size())));
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  if (IsDynamicRank(element0_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  auto element0_rank = element0_shape.size();
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange<int64_t>("Concat axis", axis_temp, kIncludeBoth,
                                              {-SizeToLong(element0_rank), SizeToLong(element0_rank) - kOneNum},
                                              prim_name);
  auto axis = axis_temp < 0 ? LongToSize(axis_temp + SizeToLong(element0_rank)) : LongToSize(axis_temp);
  int64_t all_shp = element0_shape[axis];
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    if (IsDynamicRank(elementi_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    (void)CheckAndConvertUtils::CheckInteger(elementi + " shape rank", SizeToLong(elementi_shape.size()), kEqual,
                                             SizeToLong(element0_shape.size()), prim_name);
    for (size_t j = 0; j < element0_rank; ++j) {
      if (j != axis && elementi_shape[j] != element0_shape[j] && elementi_shape[j] != -1 && element0_shape[j] != -1) {
        MS_EXCEPTION(ValueError)
          << "For '" << prim_name << "', element" << i
          << " shape in input can not concat with element0. To perform concat in the axis 0 "
             "direction, except for the 0th axis, all other axes must have the same shape. But got "
          << "element" << i << "_shape[" << j << "]: " << elementi_shape[j] << ", element0_shape[" << j
          << "]: " << element0_shape[j] << ".";
      }
    }
    all_shp = all_shp == -1 || elementi_shape[axis] == -1 ? -1 : all_shp + elementi_shape[axis];
  }
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr ConcatInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == 1) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    (void)types.emplace(elementi, elements[i]->BuildType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim_name);
  return elements[0]->BuildType();
}
}  // namespace

void Concat::Init(const int64_t axis) { this->set_axis(axis); }
int64_t Concat::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}
void Concat::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

MIND_API_OPERATOR_IMPL(Concat, BaseOperator);
AbstractBasePtr ConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, 1, primitive->name());
  auto infer_type = ConcatInferType(primitive, input_args);
  auto infer_shape = ConcatInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGConcatInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ConcatInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Concat, prim::kPrimConcat, AGConcatInfer, false);
}  // namespace ops
}  // namespace mindspore
