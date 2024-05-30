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
#include "ops/ops_func_impl/embedding_compute_var.h"

#include <vector>
#include <set>
#include <memory>
#include <string>
#include <functional>

#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace ops {
namespace {
void EmbeddingComputeVarCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  CheckTensorScalarRank(primitive, input_args[1], "ps_id");
}

void EmbeddingComputeVarCheckType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &prim_name = primitive->name();
  auto ps_id_type = input_args[kInputIndex1]->GetType();
  auto table_id_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(ps_id_type);
  MS_EXCEPTION_IF_NULL(table_id_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ps_id", ps_id_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("table_id", table_id_type, valid_types, prim_name);
}
}  // namespace

BaseShapePtr EmbeddingComputeVarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  EmbeddingComputeVarCheckShape(primitive, input_args);
  return std::make_shared<abstract::TensorShape>(ShapeVector(1));
}

TypePtr EmbeddingComputeVarFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  EmbeddingComputeVarCheckType(primitive, input_args);
  return kInt32;
}

int32_t EmbeddingComputeVarFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  const auto &table_id_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto table_name_opt = GetArrayValue<std::string>(primitive->GetAttr("table_name"));
  if (MS_UNLIKELY(IsDynamic(table_id_shape) || !table_name_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto table_name_size = table_name_opt.value().size();
  auto table_id_num =
    LongToSize(std::accumulate(table_id_shape.begin(), table_id_shape.end(), 1, std::multiplies<int64_t>()));
  if (MS_UNLIKELY(table_id_num != table_name_size)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", export table_id length " << table_id_num
                             << " doesn't match table_name length " << table_name_size << ".";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
