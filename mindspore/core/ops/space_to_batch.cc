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
#include "ops/space_to_batch.h"

#include <memory>
#include <set>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SpaceToBatch::set_paddings(const std::vector<std::vector<int64_t>> &paddings) {
  (void)this->AddAttr(kPaddings, api::MakeValue(paddings));
  int64_t h = SizeToLong(paddings.size());
  int64_t w = SizeToLong(paddings[0].size());
  std::vector<int64_t> temp_w = {2, 2};
  CheckAndConvertUtils::Check(kPaddings, {h, w}, kEqual, temp_w, this->name());
  for (size_t i = 0; i < LongToSize(h); i++) {
    for (size_t j = 0; j < LongToSize(w); j++) {
      (void)CheckAndConvertUtils::CheckInteger(kPadding, paddings[i][j], kGreaterEqual, 0, this->name());
    }
  }
}

std::vector<std::vector<int64_t>> SpaceToBatch::get_paddings() const {
  auto value_ptr = GetAttr(kPaddings);
  return GetValue<std::vector<std::vector<int64_t>>>(value_ptr);
}
void SpaceToBatch::set_block_size(const std::vector<int64_t> block_size) {
  (void)this->AddAttr(kBlockSize, api::MakeValue(block_size));
}

std::vector<int64_t> SpaceToBatch::get_block_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kBlockSize));
}

void SpaceToBatch::Init(const std::vector<int64_t> block_size, const std::vector<std::vector<int64_t>> &paddings) {
  this->set_paddings(paddings);
  this->set_block_size(block_size);
}
class SpaceToBatchInfer : public abstract::OpInferBase {
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto shapeMap = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto x_shape = shapeMap[kShape];
    if (IsDynamicRank(x_shape)) {
      return std::make_shared<abstract::Shape>(x_shape);
    }

    const int64_t rank_num = 4;
    const size_t DIM_0 = 0;
    const size_t DIM_1 = 1;
    const size_t DIM_2 = 2;
    const size_t DIM_3 = 3;
    (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(x_shape.size()), kEqual, rank_num, prim_name);

    auto out_shape = x_shape;
    auto block_size = GetValue<int64_t>(primitive->GetAttr("block_size"));
    auto paddings = GetValue<std::vector<std::vector<int64_t>>>(primitive->GetAttr("paddings"));

    if (out_shape[DIM_0] != abstract::Shape::kShapeDimAny) {
      out_shape[DIM_0] *= block_size * block_size;
    }

    if (out_shape[DIM_2] != abstract::Shape::kShapeDimAny) {
      auto padded_0 = out_shape[DIM_2] + paddings[DIM_0][DIM_0] + paddings[DIM_0][DIM_1];
      if (padded_0 % block_size != 0) {
        MS_EXCEPTION(ValueError) << "For SpaceToBatch, the x_shape[2] plus paddings must be divisible by "
                                    "'block_size', but got padded value: "
                                 << padded_0 << ", and block_size: " << block_size;
      }
      out_shape[DIM_2] = padded_0 / block_size;
    }

    if (out_shape[DIM_3] != abstract::Shape::kShapeDimAny) {
      auto padded_1 = out_shape[DIM_3] + paddings[DIM_1][DIM_0] + paddings[DIM_1][DIM_1];
      if (padded_1 % block_size != 0) {
        MS_EXCEPTION(ValueError) << "For SpaceToBatch, the x_shape[3] plus paddings must be divisible by "
                                    "'block_size', but got padded value: "
                                 << padded_1 << ", and block_size: " << block_size;
      }
      out_shape[DIM_3] = padded_1 / block_size;
    }

    return std::make_shared<abstract::Shape>(out_shape);
  }
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto input_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    const std::set<TypePtr> number_type = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,   kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex64};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, number_type, prim_name);
    return input_type;
  }
};

MIND_API_OPERATOR_IMPL(SpaceToBatch, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(SpaceToBatch, prim::kPrimSpaceToBatch, SpaceToBatchInfer, false);
}  // namespace ops
}  // namespace mindspore
