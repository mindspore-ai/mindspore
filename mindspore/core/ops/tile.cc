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

#include <map>
#include "ops/tile.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> GetInferShape(const PrimitivePtr &prim, const std::vector<int64_t> &input_shape,
                                   const std::vector<int64_t> &multiples_v) {
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
  if (len_sub < 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim->name()
      << "', 'multiples' length cannot be smaller than 'input_x' dimension length, but got 'multiples' length:"
      << multiples_v.size() << ", 'input_x' dimension length: " << input_shape.size() << ".";
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
  const int INDEX = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, INDEX, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  std::vector<int64_t> multiples_v;
  auto multiple_value = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(multiple_value);
  if (multiple_value->isa<tensor::Tensor>()) {
    multiples_v = CheckAndConvertUtils::CheckTensorIntValue("multiples", multiple_value, prim_name);
  } else {
    multiples_v = CheckAndConvertUtils::CheckTupleInt("input[multiples]", multiple_value, prim_name);
  }

  for (auto multiple : multiples_v) {
    if (multiple <= 0) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', 'multiples' must be an positive integer, but got " << multiples_v
                        << ".";
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
AbstractBasePtr TileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(TileInferShape(primitive, input_args), TileInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameTile, Tile);
}  // namespace ops
}  // namespace mindspore
