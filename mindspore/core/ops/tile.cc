/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TileInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto tile_prim = primitive->cast<PrimTilePtr>();
  MS_EXCEPTION_IF_NULL(tile_prim);
  auto prim_name = tile_prim->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x shape", input_args[0]->BuildShape(), prim_name);
  auto multiples_v = GetValue<std::vector<int64_t>>(input_args[1]->cast<abstract::AbstractTuplePtr>()->BuildValue());
  int len_sub = input_shape.size() - multiples_v.size();
  std::vector<int64_t> infer_shape = input_shape;
  std::vector<int64_t> multiples_w;
  if (len_sub == 0) {
    multiples_w = multiples_v;
  }
  if (len_sub > 0) {
    for (int64_t i = 0; i < len_sub; i++) {
      infer_shape.insert(infer_shape.begin(), 1);
    }
    multiples_w = multiples_v;
  }
  if (len_sub < 0) {
    MS_EXCEPTION(ValueError) << "the length of multiples can not be smaller than the"
                                "length of dimension in input_x";
  }
  for (size_t i = 0; i < multiples_w.size(); i++) {
    infer_shape[i] *= multiples_w[i];
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr TileInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInteger("tile_prim_infer", input_args.size(), kEqual, 2, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType()->cast<TensorTypePtr>();
  std::set<TypePtr> template_types = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
  auto infer_dtype = x_dtype->element()->type_id();
  return TypeIdToType(infer_dtype);
}
}  // namespace

AbstractBasePtr TileInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(TileInferType(primitive, input_args),
                                                    TileInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameTile, Tile);
}  // namespace ops
}  // namespace mindspore
