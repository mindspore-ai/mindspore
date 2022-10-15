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

#include "ops/dynamic_stitch.h"
#include <set>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DynamicStitchInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t args_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, args_size, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  // input0: indices
  auto input_tuple = input_args[0]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple);
  auto indices = input_tuple->elements();
  auto indices_size = input_tuple->size();

  // input1: data
  auto input_tuple_1 = input_args[1]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple_1);
  auto data = input_tuple_1->elements();
  auto data_size = input_tuple_1->size();
  // check tuple size
  if (indices_size != data_size) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', indices_size: " << indices_size
                             << " should be same as data_size: " << data_size;
  }
  if (data_size <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', data tuple size should be greater than 0";
  }

  // support input indices dynamic rank
  for (size_t i = 0; i < indices_size; i++) {
    auto indices_i = indices[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(indices_i);
    auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_i->BuildShape())[kShape];
    if (IsDynamicRank(indices_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
  }

  // support input data dynamic rank
  for (size_t i = 0; i < data_size; i++) {
    auto data_i = data[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(data_i);
    auto data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_i->BuildShape())[kShape];
    if (IsDynamicRank(data_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
  }

  bool first_dim_unknow = false;
  int64_t first_dim_size = 0;
  for (size_t i = 0; i < indices_size; i++) {
    auto indice_i = indices[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(indice_i);
    auto value_i = indice_i->BuildValue();
    MS_EXCEPTION_IF_NULL(value_i);
    if (!value_i->isa<tensor::Tensor>()) {
      first_dim_unknow = true;
      continue;
    }
    auto index_i_value = CheckAndConvertUtils::CheckTensorIntValue("indices", value_i, prim_name);
    auto index_i_max = std::max_element(index_i_value.begin(), index_i_value.end());
    first_dim_size = *index_i_max > first_dim_size ? *index_i_max : first_dim_size;
  }

  auto indices0 = indices[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(indices0);
  auto indices0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices0->BuildShape())[kShape];

  auto data0 = data[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(data0);
  auto data0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data0->BuildShape())[kShape];

  for (size_t i = 1; i < data.size(); ++i) {
    auto indicesi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices[i]->BuildShape())[kShape];
    auto datai_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data[i]->BuildShape())[kShape];
    if (indicesi_shape.size() > datai_shape.size()) {
      MS_LOG(EXCEPTION) << "The rank of indices[i] must be <= rank of data[i]!";
    }
  }

  ShapeVector out_shape;
  if (first_dim_unknow) {
    out_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    out_shape.push_back(first_dim_size + 1);
  }
  // support input data dynamic shape
  bool data_shape_is_dynamic = false;
  for (size_t i = 0; i < data_size; i++) {
    auto data_i = data[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(data_i);
    auto data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_i->BuildShape())[kShape];
    if (IsDynamic(data_shape)) {
      data_shape_is_dynamic = true;
      break;
    }
  }
  if (data_shape_is_dynamic) {
    for (size_t i = indices0_shape.size(); i < data0_shape.size(); ++i) {
      out_shape.push_back(abstract::Shape::kShapeDimAny);
    }
  } else {
    for (size_t i = indices0_shape.size(); i < data0_shape.size(); ++i) {
      out_shape.push_back(data0_shape[i]);
    }
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DynamicStitchInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto input_tuple_1 = input_args[1]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple_1);
  auto data_size = input_tuple_1->size();
  if (data_size <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', data tuple size should be greater than 0";
  }

  auto data = input_tuple_1->elements();
  auto data0 = data[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(data0);

  std::map<std::string, TypePtr> types;
  (void)types.emplace("data0", data0->BuildType());

  std::set<TypePtr> valid_types = ops::common_valid_types;
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicStitch, BaseOperator);
AbstractBasePtr DynamicStitchInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = DynamicStitchInferType(primitive, input_args);
  auto infer_shape = DynamicStitchInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(DynamicStitch, prim::kPrimDynamicStitch, DynamicStitchInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
