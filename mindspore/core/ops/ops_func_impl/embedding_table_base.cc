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
#include "ops/ops_func_impl/embedding_table_base.h"

#include <set>
#include <memory>
#include <functional>
#include <string>

#include "ops/op_utils.h"
#include "utils/shape_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
void EmbeddingTableBaseCheckShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  CheckTensorScalarRank(primitive, input_args[1], "ps_id");
  const auto &table_id_shape = input_args[2]->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(table_id_shape))) {
    MS_CHECK_VALUE(table_id_shape.size() == 1,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of table_id", SizeToLong(table_id_shape.size()),
                                                               kEqual, 1, primitive));
  }
}

void EmbeddingTableBaseCheckType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = primitive->name();
  auto ps_id_type = input_args[1]->GetType();
  MS_EXCEPTION_IF_NULL(ps_id_type);
  auto table_id_type = input_args[2]->GetType();
  MS_EXCEPTION_IF_NULL(table_id_type);
  const std::set<TypePtr> valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ps_id", ps_id_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("table_id", table_id_type, valid_types, prim_name);
}
}  // namespace
BaseShapePtr EmbeddingTableBaseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  EmbeddingTableBaseCheckShape(primitive, input_args);
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr EmbeddingTableBaseFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  EmbeddingTableBaseCheckType(primitive, input_args);
  return std::make_shared<TensorType>(kInt32);
}

int32_t EmbeddingTableBaseFuncImpl::CommonCheckValidation(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto file_type_opt = GetScalarValue<std::string>(primitive->GetAttr("file_type"));
  if (MS_UNLIKELY(!file_type_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto file_type = file_type_opt.value();
  if (MS_UNLIKELY(file_type != "bin")) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", does not support Non-bin file.";
  }

  const auto &table_id_shape = input_args[kInputIndex2]->GetShape()->GetShapeVector();
  auto table_name_opt = GetArrayValue<std::string>(primitive->GetAttr("table_name"));
  if (MS_UNLIKELY(IsDynamic(table_id_shape) || !table_name_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto table_name_size = table_name_opt.value().size();
  auto table_id_num =
    LongToSize(std::accumulate(table_id_shape.begin(), table_id_shape.end(), 1, std::multiplies<int64_t>()));
  if (MS_UNLIKELY(table_id_num <= kInputIndex0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", table_id should not be empty tensor.";
  }
  if (MS_UNLIKELY(table_id_num != table_name_size)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", export table_id length " << table_id_num
                             << " doesn't match table_name length " << table_name_size << ".";
  }

  auto embedding_dim_opt = GetArrayValue<int64_t>(primitive->GetAttr("embedding_dim"));
  auto value_total_len_opt = GetArrayValue<int64_t>(primitive->GetAttr("value_total_len"));
  if (MS_UNLIKELY(!value_total_len_opt.has_value() || !embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto embedding_dim_size = embedding_dim_opt.value().size();
  auto value_total_len_size = value_total_len_opt.value().size();
  if (MS_UNLIKELY(table_id_num != embedding_dim_size || table_id_num != value_total_len_size)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", table_id length[" << table_id_num
                             << "], embedding_dim length[" << embedding_dim_size << "], value_total_len length["
                             << value_total_len_size << "] are not equal.";
  }

  return OP_CHECK_SUCCESS;
}

int32_t EmbeddingTableBaseFuncImpl::SpecifiedCheckValidation(const PrimitivePtr &primitive,
                                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_LOG(EXCEPTION) << "The method 'SpecifiedCheckValidation()' doesn't implement.";
}

int32_t EmbeddingTableBaseFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto ret_common = CommonCheckValidation(primitive, input_args);
  auto ret_specified = SpecifiedCheckValidation(primitive, input_args);
  if (ret_common == OP_CHECK_RETRY || ret_specified == OP_CHECK_RETRY) {
    return OP_CHECK_RETRY;
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace mindspore::ops
