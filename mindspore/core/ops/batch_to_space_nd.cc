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
#include "ops/batch_to_space_nd.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BatchToSpaceNDInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  constexpr int64_t ascend_len = 4;
  constexpr int64_t len = 2;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  (void)CheckAndConvertUtils::CheckInteger("input_x rank", SizeToLong(x_shape.size()),
                                           (is_ascend ? kEqual : kGreaterEqual), (is_ascend ? ascend_len : len),
                                           prim_name);
  if (IsDynamicShape(x_shape)) {
    std::vector<int64_t> res(x_shape.size(), abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(res);
  }
  auto out_shape = x_shape;
  int64_t block_shape_prod = 1;
  auto block_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr(kBlockShape));
  auto crops = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr(kCrops));
  size_t size = block_shape.size();
  if (x_shape.size() <= size) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the rank of x_shape must be greater than the rank of `block_shape`.";
  }
  size_t offset = x_shape.size() - size;
  for (size_t i = 0; i < size; i++) {
    block_shape_prod = block_shape_prod * block_shape[i];
    auto x_block_prod = out_shape[i + offset] * block_shape[i];
    auto crops_sum = crops[i][0] + crops[i][1];
    CheckAndConvertUtils::Check("x block shape prod", x_block_prod, kGreaterThan, crops_sum, prim_name);
    out_shape[i + offset] = x_block_prod - crops_sum;
  }
  if (out_shape[0] % block_shape_prod != 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', the first dim of 'input_x' must be divisible by 'block_shape_prod'. But got first dim of 'input_x': "
      << out_shape[0] << ", 'block_shape_prod' with value: " << block_shape_prod << ".";
  }
  out_shape[0] /= block_shape_prod;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr BatchToSpaceNDInferType(const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  // check_scalar_or_tensor_types_same
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, "BatchToSpaceND");
}
}  // namespace

MIND_API_OPERATOR_IMPL(BatchToSpaceND, BaseOperator);
void BatchToSpaceND::set_crops(std::vector<std::vector<int64_t>> crops) {
  const int64_t crop_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kCrops, SizeToLong(crops.size()), kEqual, crop_size, this->name());
  size_t h = crops.size();
  size_t w = crops[0].size();
  std::vector<size_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kCrops, {h, w}, kEqual, temp_w, this->name());
  for (size_t i = 0; i < h; i++) {
    for (size_t j = 0; j < w; j++) {
      (void)CheckAndConvertUtils::CheckInteger(kCrops, crops[i][j], kGreaterEqual, 0, this->name());
    }
  }
  (void)this->AddAttr(kCrops, api::MakeValue(crops));
}

std::vector<std::vector<int64_t>> BatchToSpaceND::get_crops() const {
  auto value_ptr = GetAttr(kCrops);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void BatchToSpaceND::set_block_shape(std::vector<int64_t> block_shape) {
  const int64_t block_size = 2;
  (void)CheckAndConvertUtils::CheckInteger(kBlockShape, SizeToLong(block_shape.size()), kEqual, block_size,
                                           this->name());
  for (size_t i = 0; i < block_shape.size(); i++) {
    (void)CheckAndConvertUtils::CheckInteger(kBlockShape, block_shape[i], kGreaterEqual, 1, this->name());
  }
  (void)this->AddAttr(kBlockShape, api::MakeValue(block_shape));
}

std::vector<int64_t> BatchToSpaceND::get_block_shape() const {
  auto value_ptr = GetAttr(kBlockShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void BatchToSpaceND::Init(const std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> crops) {
  this->set_crops(crops);
  this->set_block_shape(block_shape);
}

AbstractBasePtr BatchToSpaceNDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  const int64_t kInputNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  auto infer_type = BatchToSpaceNDInferType(input_args);
  auto infer_shape = BatchToSpaceNDInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBatchToSpaceNDInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDInferType(input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchToSpaceND, prim::kPrimBatchToSpaceND, AGBatchToSpaceNDInfer, false);
}  // namespace ops
}  // namespace mindspore
