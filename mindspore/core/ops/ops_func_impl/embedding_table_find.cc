/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/embedding_table_find.h"
#include <vector>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/embedding_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingTableFindFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  CheckTensorScalarRank(primitive, input_args[0], "table_id");

  std::vector<int64_t> output_shape{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};

  auto keys_shape_ptr = input_args[1]->GetShape();
  MS_EXCEPTION_IF_NULL(keys_shape_ptr);
  const auto &keys_shape = keys_shape_ptr->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(keys_shape))) {
    MS_CHECK_VALUE(keys_shape.size() == 1,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of keys", SizeToLong(keys_shape.size()), kEqual,
                                                               int64_t(1), primitive));
    output_shape[0] = keys_shape[0];
  }

  auto embedding_dim_opt = GetScalarValue<int64_t>(input_args[2]->GetValue());
  if (MS_LIKELY(embedding_dim_opt.has_value())) {
    auto embedding_dim = embedding_dim_opt.value();
    MS_CHECK_VALUE(embedding_dim > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", embedding_dim,
                                                                                  kGreaterThan, int64_t(0), primitive));
    output_shape[1] = embedding_dim;
  }

  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

TypePtr EmbeddingTableFindFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  const auto &table_id_type = input_args[0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("table_id", table_id_type, {kInt32}, prim_name);

  const auto &keys_type = input_args[1]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  return kFloat32;
}

int32_t EmbeddingTableFindFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto ret = CheckEmbeddingOpsExtraArgs(primitive, {input_args[kInputIndex2], input_args[kInputIndex4],
                                                    input_args[kInputIndex1], input_args[kInputIndex5]});
  return ret;
}
}  // namespace ops
}  // namespace mindspore
