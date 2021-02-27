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

#include <set>
#include "ops/embedding_lookup.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void EmbeddingLookup::Init(const bool setattr_flag) { this->set_setattr_flag(setattr_flag); }

void EmbeddingLookup::set_setattr_flag(const bool setattr_flag) {
  this->AddAttr(kSetattrFlag, MakeValue(setattr_flag));
}

bool EmbeddingLookup::get_setattr_flag() const {
  auto value_ptr = GetAttr(kSetattrFlag);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr EmbeddingLookupInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto params = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(params);
  auto indices = input_args[1]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(indices);
  const std::set<TypeId> int_valid_types = {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("indices type", indices->BuildType(), int_valid_types, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[2]->BuildType());
  auto offset_type = input_args[2]->BuildType()->type_id();
  if (int_valid_types.find(offset_type) == int_valid_types.end()) {
    MS_LOG(EXCEPTION) << "offset must be int.";
  }

  MS_EXCEPTION_IF_NULL(params->shape());
  auto params_shp = params->shape()->shape();
  MS_EXCEPTION_IF_NULL(indices->shape());
  auto indices_shp = indices->shape()->shape();
  ShapeVector shape;
  shape.insert(shape.end(), indices_shp.begin(), indices_shp.end());
  shape.insert(shape.end(), params_shp.begin() + 1, params_shp.end());
  auto indices_max_shape = indices->shape()->max_shape();
  ShapeVector max_shape;
  if (!indices_max_shape.empty()) {
    max_shape.insert(max_shape.end(), indices_max_shape.begin(), indices_max_shape.end());
    max_shape.insert(max_shape.end(), params_shp.begin() + 1, params_shp.end());
  } else {
    max_shape = shape;
  }
  auto indices_min_shape = indices->shape()->min_shape();
  ShapeVector min_shape;
  if (!indices_min_shape.empty()) {
    min_shape.insert(min_shape.end(), indices_min_shape.begin(), indices_min_shape.end());
    min_shape.insert(min_shape.end(), params_shp.begin() + 1, params_shp.end());
  } else {
    min_shape = shape;
  }

  return std::make_shared<abstract::AbstractTensor>(params->element(),
                                                    std::make_shared<abstract::Shape>(shape, min_shape, max_shape));
}
REGISTER_PRIMITIVE_C(kNameEmbeddingLookup, EmbeddingLookup);
}  // namespace ops
}  // namespace mindspore
