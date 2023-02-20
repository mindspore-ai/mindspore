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
#include "ops/batch_to_space_nd_v2.h"

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <map>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BatchToSpaceNDV2InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto out_shape = x_shape;

  int64_t block_shape_prod = 1;
  if (input_args[1]->isa<abstract::AbstractTensor>() && !input_args[1]->BuildValue()->isa<tensor::Tensor>()) {
    std::vector<int64_t> res(out_shape.size(), -1);
    return std::make_shared<abstract::Shape>(res);
  }
  constexpr auto index2 = 2;
  if (input_args[index2]->isa<abstract::AbstractTensor>() && !input_args[index2]->BuildValue()->isa<tensor::Tensor>()) {
    std::vector<int64_t> res(out_shape.size(), -1);
    return std::make_shared<abstract::Shape>(res);
  }
  auto block_shape = CheckAndConvertUtils::CheckTensorIntValue(kBlockShape, input_args[1]->BuildValue(), prim_name);
  auto crops = CheckAndConvertUtils::CheckTensorIntValue(kCrops, input_args[index2]->BuildValue(), prim_name);
  size_t size = block_shape.size();
  size_t offset = x_shape.size() - size;
  for (size_t i = 0; i < size; i++) {
    block_shape_prod = block_shape_prod * block_shape[i];
    auto x_block_prod = out_shape[i + offset] * block_shape[i];
    auto crops_sum = crops[i * index2] + crops[i * index2 + 1];
    CheckAndConvertUtils::Check("x block shape prod", x_block_prod, kGreaterThan, crops_sum, prim_name);
    out_shape[i + offset] = x_block_prod - crops_sum;
  }
  if (out_shape[0] == -1) {
    return std::make_shared<abstract::Shape>(out_shape);
  }
  if (out_shape[0] % block_shape_prod != 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', the first dim of 'input_x' must be divisible by 'block_shape_prod'. But got first dim of 'input_x': "
      << out_shape[0] << ", 'block_shape_prod' with value: " << block_shape_prod << ".";
  }
  out_shape[0] = int64_t(floor(out_shape[0] / static_cast<float>(block_shape_prod)));

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr BatchToSpaceNDV2InferType(const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  // check_scalar_or_tensor_types_same
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, "BatchToSpaceNDV2");
}
}  // namespace

MIND_API_OPERATOR_IMPL(BatchToSpaceNDV2, BaseOperator);
AbstractBasePtr BatchToSpaceNDV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  const int64_t kInputNum = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           prim_name);
  auto infer_type = BatchToSpaceNDV2InferType(input_args);
  auto infer_shape = BatchToSpaceNDV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBatchToSpaceNDV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDV2InferType(input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BatchToSpaceNDV2Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchToSpaceNDV2, prim::kPrimBatchToSpaceNDV2, AGBatchToSpaceNDV2Infer, false);
}  // namespace ops
}  // namespace mindspore
