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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DynamicStitchFrontendInferShape(const PrimitivePtr &primitive,
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
    auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_i->GetShape())[kShape];
    if (IsDynamicRank(indices_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
  }

  // support input data dynamic rank
  for (size_t i = 0; i < data_size; i++) {
    auto data_i = data[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(data_i);
    auto data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_i->GetShape())[kShape];
    if (IsDynamicRank(data_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    }
  }

  bool first_dim_unknow = false;
  int64_t first_dim_size = 0;
  for (size_t i = 0; i < indices_size; i++) {
    auto indice_i = indices[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(indice_i);
    auto value_i = indice_i->GetValue();
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
  auto indices0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices0->GetShape())[kShape];

  auto data0 = data[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(data0);
  auto data0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data0->GetShape())[kShape];

  for (size_t i = 1; i < data.size(); ++i) {
    auto indicesi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices[i]->GetShape())[kShape];
    auto datai_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data[i]->GetShape())[kShape];
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
    auto data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_i->GetShape())[kShape];
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

TypePtr DynamicStitchFrontendInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
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
  (void)types.emplace("data0", data0->GetType());

  std::set<TypePtr> valid_types = ops::common_valid_types;
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicStitch, BaseOperator);
AbstractBasePtr DynamicStitchInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = DynamicStitchFrontendInferType(primitive, input_args);
  auto infer_shape = DynamicStitchFrontendInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDynamicStitchInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.size() < kSize2) {
      MS_LOG(EXCEPTION) << "Dynamic stitch inputs size must greater than 2. But got: " << input_args.size();
    }
    if (input_args.size() % 2 != 0) {
      MS_LOG(EXCEPTION) << "Dynamic stitch inputs size must be even. But got: " << input_args.size();
    }

    int64_t first_dim_size = 0;
    auto indices_size = input_args.size() / 2;
    for (size_t i = 0; i < indices_size; i++) {
      auto indice_i = input_args[i];
      MS_EXCEPTION_IF_NULL(indice_i);
      auto value_i = indice_i->GetValue();
      MS_EXCEPTION_IF_NULL(value_i);
      auto index_i_value =
        CheckAndConvertUtils::CheckTensorIntValue("indices", value_i, primitive->name(), indice_i->GetType());
      auto index_i_max = std::max_element(index_i_value.begin(), index_i_value.end());
      first_dim_size = *index_i_max > first_dim_size ? *index_i_max : first_dim_size;
    }

    MS_EXCEPTION_IF_NULL(input_args[0]);
    MS_EXCEPTION_IF_NULL(input_args[indices_size]);
    auto indices0_shape = input_args[0]->GetShape()->GetShapeVector();
    auto data0_shape = input_args[indices_size]->GetShape()->GetShapeVector();

    for (size_t i = 1; i < indices_size; ++i) {
      auto indicesi_shape = input_args[i]->GetShape()->GetShapeVector();
      auto datai_shape = input_args[indices_size + i]->GetShape()->GetShapeVector();
      if (indicesi_shape.size() > datai_shape.size()) {
        MS_LOG(EXCEPTION) << "The rank of indices[i] must be <= rank of data[i]!";
      }
    }
    ShapeVector out_shape;
    out_shape.push_back(first_dim_size + 1);
    // support input data dynamic shape
    for (size_t i = indices0_shape.size(); i < data0_shape.size(); ++i) {
      out_shape.push_back(data0_shape[i]);
    }
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto data_size = input_args.size();
    if (data_size < 2) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', data tuple size should be greater than 1";
    }
    auto indices_size = input_args.size() / 2;
    auto data0 = input_args[indices_size];
    MS_EXCEPTION_IF_NULL(data0);
    MS_EXCEPTION_IF_NULL(data0->GetType());
    return data0->GetType()->Clone();
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DynamicStitchInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicStitch, prim::kPrimDynamicStitch, AGDynamicStitchInfer, false);
}  // namespace ops
}  // namespace mindspore
