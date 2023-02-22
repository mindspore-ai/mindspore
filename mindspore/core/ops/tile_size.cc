/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/tile_size.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr MakeTuple(int64_t num) {
  abstract::AbstractBasePtrList elems;
  for (int64_t i = 0; i < num; i++) {
    elems.emplace_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, kInt64));
  }

  return std::make_shared<abstract::AbstractTuple>(elems);
}
}  // namespace
MIND_API_OPERATOR_IMPL(TileSize, BaseOperator);
class TileSizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto ndim_abs = input_args[2]->cast<abstract::AbstractScalarPtr>();
    auto ndim_value = ndim_abs->BuildValue();
    if (IsValueKnown(ndim_value)) {
      auto ndim = GetValue<int64_t>(ndim_value);
      auto abs = MakeTuple(ndim);
      return abs->BuildShape();
    }

    return nullptr;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    auto ndim_abs = input_args[2]->cast<abstract::AbstractScalarPtr>();
    auto ndim_value = ndim_abs->BuildValue();
    if (IsValueKnown(ndim_value)) {
      auto ndim = GetValue<int64_t>(ndim_value);
      auto abs = MakeTuple(ndim);
      return abs->BuildType();
    }

    return nullptr;
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto ndim_abs = input_args[2]->cast<abstract::AbstractScalarPtr>();
    auto ndim_value = ndim_abs->BuildValue();
    if (IsValueKnown(ndim_value)) {
      auto ndim = GetValue<int64_t>(ndim_value);
      auto out_abs = MakeTuple(ndim);
      return out_abs;
    }

    return nullptr;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      return nullptr;
    }

    auto shape_value = input_args[0]->BuildValue();
    auto out_shape_value = input_args[1]->BuildValue();
    auto ndim_value = input_args[2]->BuildValue();
    if (!IsValueKnown(shape_value) || !IsValueKnown(out_shape_value) || !IsValueKnown(ndim_value)) {
      return nullptr;
    }
    auto shape = GetValue<ShapeVector>(shape_value);
    auto out_shape = GetValue<ShapeVector>(out_shape_value);
    auto ndim = GetValue<int64_t>(ndim_value);

    ShapeVector out(ndim, 1);
    abstract::AbstractBasePtrList elems;
    auto size = std::min(shape.size(), out_shape.size());
    for (size_t i = 0; i < size; i++) {
      if (shape[i] != out_shape[i]) {
        out[i] = out_shape[i];
      }
    }
    for (size_t i = 0; i < out.size(); i++) {
      auto scalar = std::make_shared<abstract::AbstractScalar>(out[i]);
      (void)elems.emplace_back(scalar);
    }

    return std::make_shared<abstract::AbstractTuple>(elems)->BuildValue();
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(TileSize, prim::kPrimTileSize, TileSizeInfer, true);
}  // namespace ops
}  // namespace mindspore
