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
#include <string>
#include "ops/parallel_concat.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kOneNum = 1;
abstract::ShapePtr ParallelConcatInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           prim_name);
  (void)primitive->AddAttr("N", MakeValue(SizeToLong(elements.size())));
  (void)primitive->AddAttr("inputNums", MakeValue(SizeToLong(elements.size())));

  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape_map_i = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape());
    auto shape_i = shape_map_i[kShape];
    if (IsDynamicRank(shape_i)) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
  }
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  auto element0_rank = element0_shape.size();
  if (element0_rank < kOneNum) {
    MS_EXCEPTION(ValueError) << "For [" << prim_name
                             << "], the rank of input must greater than 1. But got:" << element0_rank << ".";
  }

  auto axis = 0;
  int64_t all_shp = static_cast<int64_t>(element0_shape[axis]);
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape_i = elements[i]->BuildShape();
    if (shape_i->IsDynamic()) {
      auto ret_shape = element0_shape;
      ret_shape[axis] = -1;
      return std::make_shared<abstract::Shape>(ret_shape);
    }
  }
  for (size_t i = 1; i < elements.size(); ++i) {
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("x" + std::to_string(i) + ".shape[0]", elementi_shape[0], kEqual, kOneNum,
                                             prim_name);
    if (elementi_shape.size() != element0_shape.size()) {
      MS_EXCEPTION(ValueError) << "For [" << prim_name << "], the rank of all elements should be the same, but got "
                               << "x0.rank [" << element0_shape.size() << "] and x" << std::to_string(i) << ".rank ["
                               << elementi_shape.size() << "].";
    }
    for (size_t j = 1; j < element0_rank; ++j) {
      if (elementi_shape[j] != element0_shape[j]) {
        MS_EXCEPTION(ValueError) << "For [" << prim_name << "], the shape of all elements should be the same, but got "
                                 << "x0.shape[" << std::to_string(j) << "] = [" << element0_shape[j] << "] and x"
                                 << std::to_string(i) << ".shape[" << std::to_string(j) << "] = [" << elementi_shape[j]
                                 << "].";
      }
    }

    all_shp = all_shp + elementi_shape[axis];
  }
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  (void)primitive->AddAttr("shape", MakeValue(ret_shape));
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr ParallelConcatInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the input must be a list or tuple of tensors. But got:" << input_args[0]->ToString()
                            << ".";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           prim_name);
  std::map<std::string, TypePtr> types;
  for (size_t i = 0; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    (void)types.emplace(elementi, elements[i]->BuildType());
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex_and_bool, prim_name);
  return elements[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(ParallelConcat, BaseOperator);
AbstractBasePtr ParallelConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = ParallelConcatInferType(primitive, input_args);
  auto infer_shape = ParallelConcatInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGParallelConcatInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ParallelConcatInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ParallelConcatInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ParallelConcatInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ParallelConcat, prim::kPrimParallelConcat, AGParallelConcatInfer, false);
}  // namespace ops
}  // namespace mindspore
