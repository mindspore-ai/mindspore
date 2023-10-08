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

#include "ops/tile.h"
#include <map>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> GetInferShape(const PrimitivePtr &prim, const std::vector<int64_t> &input_shape,
                                   const std::vector<int64_t> &multiples_v) {
  if (multiples_v.size() < input_shape.size()) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim->name()
      << "', 'multiples' length cannot be smaller than 'input_x' dimension length, but got 'multiples' length:"
      << multiples_v.size() << ", 'input_x' dimension length: " << input_shape.size() << ".";
  }
  int64_t len_sub = SizeToLong(multiples_v.size() - input_shape.size());
  std::vector<int64_t> infer_shape = input_shape;
  std::vector<int64_t> multiples_w;
  if (len_sub == 0) {
    multiples_w = multiples_v;
  }
  if (len_sub > 0) {
    for (int64_t i = 0; i < len_sub; i++) {
      (void)infer_shape.insert(infer_shape.begin(), 1);
    }
    multiples_w = multiples_v;
  }
  for (size_t i = 0; i < multiples_w.size(); i++) {
    if (infer_shape[i] == abstract::Shape::kShapeDimAny) {
      continue;
    }
    infer_shape[i] *= multiples_w[i];
  }
  return infer_shape;
}

abstract::ShapePtr TileInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, num, prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  std::vector<int64_t> multiples_v;
  auto multiple_value = input_args[1]->BuildValue();
  auto multiple_shape = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(multiple_value);
  if (input_args[1]->isa<abstract::AbstractTensor>()) {
    if (!IsValueKnown(multiple_value)) {
      auto shape = multiple_shape->cast<abstract::ShapePtr>()->shape();
      if (IsDynamic(shape)) {
        return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
      }
      return std::make_shared<abstract::Shape>(ShapeVector(shape[kIndex0], abstract::Shape::kShapeDimAny));
    }
    multiples_v = CheckAndConvertUtils::CheckTensorIntValue("multiples", multiple_value, prim_name);
  } else {
    auto tuple_abs = input_args[1]->cast<abstract::AbstractSequencePtr>();
    if (tuple_abs == nullptr) {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name
                              << "], the input[multiples] must be a tuple with all Int elements, but got "
                              << input_args[1]->BuildType()->ToString();
    }
    if (!IsValueKnown(multiple_value)) {
      if (tuple_abs->dynamic_len()) {
        return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
      }
      return std::make_shared<abstract::Shape>(
        ShapeVector(tuple_abs->elements().size(), abstract::Shape::kShapeDimAny));
    } else {
      multiples_v = CheckAndConvertUtils::CheckTupleInt("input[multiples]", multiple_value, prim_name);
    }
  }

  for (auto multiple : multiples_v) {
    if (multiple <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'multiples' must be an positive integer, but got "
                               << multiples_v << ".";
    }
  }

  auto infer_shape = GetInferShape(primitive, input_shape, multiples_v);
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr TileInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type_map = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_map);
  auto x_dtype = x_type_map->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_dtype);
  std::set<TypePtr> template_types = {kTensorType};
  return CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, template_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Tile, BaseOperator);

// AG means auto generated
class MIND_API AGTileInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return TileInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return TileInferType(primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Tile, prim::kPrimTile, AGTileInfer, false);
}  // namespace ops
}  // namespace mindspore
