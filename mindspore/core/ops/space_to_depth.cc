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

#include "ops/space_to_depth.h"

#include <map>
#include <memory>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_name.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SpaceToDepth::Init(const int64_t block_size, const Format &format) {
  this->set_block_size(block_size);
  this->set_format(format);
}

void SpaceToDepth::set_block_size(const int64_t block_size) {
  CheckAndConvertUtils::Check(kBlockSize, block_size, kGreaterEqual, 2, this->name());
  (void)AddAttr(kBlockSize, api::MakeValue(block_size));
}

int64_t SpaceToDepth::get_block_size() const {
  auto value_ptr = GetAttr(kBlockSize);
  return GetValue<int64_t>(value_ptr);
}

void SpaceToDepth::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

Format SpaceToDepth::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

namespace {
abstract::ShapePtr SpaceToDepthInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto shapeMap = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = shapeMap[kShape];
  auto block_size = GetValue<int64_t>(primitive->GetAttr("block_size"));
  const int64_t c_of_nchw = 1;
  const int64_t h_of_nchw = 2;
  const int64_t w_of_nchw = 3;
  const int64_t min_block_size = 2;
  auto data_format_ptr = primitive->GetAttr("format");
  if (input_args.size() != 1) {
    MS_EXCEPTION(ValueError) << "For SpaceToDepth, the input number must be 1, but got " << input_args.size();
  }
  (void)primitive->AddAttr("data_format", data_format_ptr);
  if (block_size < min_block_size) {
    MS_EXCEPTION(ValueError) << "For SpaceToDepth, block_size must greater than 2, but got the block_size is "
                             << block_size;
  }
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  if (IsDynamicRank(x_shape) || x_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(x_shape.size()), kEqual, x_rank, prim_name);
  auto out_shape = x_shape;
  if (out_shape[c_of_nchw] != abstract::Shape::kShapeDimAny) {
    out_shape[c_of_nchw] *= block_size * block_size;
  }
  if (out_shape[h_of_nchw] != abstract::Shape::kShapeDimAny) {
    (void)CheckAndConvertUtils::CheckInteger("x_shape[2] % block_size", x_shape[h_of_nchw] % block_size, kEqual, 0,
                                             prim_name);
    out_shape[h_of_nchw] /= block_size;
  }
  if (out_shape[w_of_nchw] != abstract::Shape::kShapeDimAny) {
    (void)CheckAndConvertUtils::CheckInteger("x_shape[3] % block_size", x_shape[w_of_nchw] % block_size, kEqual, 0,
                                             prim_name);
    out_shape[w_of_nchw] /= block_size;
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr SpaceToDepthInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  std::set<TypePtr> valid_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, prim_name);
  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SpaceToDepth, BaseOperator);
AbstractBasePtr SpaceToDepthInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  auto infer_type = SpaceToDepthInferType(primitive, input_args);
  auto infer_shape = SpaceToDepthInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSpaceToDepthInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToDepthInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToDepthInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SpaceToDepthInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SpaceToDepth, prim::kPrimSpaceToDepth, AGSpaceToDepthInfer, false);
}  // namespace ops
}  // namespace mindspore
