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

#include "ops/ops_func_impl/init_embedding_hashmap.h"

#include <vector>
#include <string>
#include <memory>

#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kNameFilterMode = "filter_mode";
constexpr auto kInputFilterModeIndex = 13;
bool CheckRank(const ShapeVector &shape, int64_t rank) {
  if (rank > INT32_MAX) {
    MS_LOG(ERROR) << "rank[" << rank << "] cannot exceed int32max.";
    return false;
  }
  if (IsDynamicRank(shape)) {
    return true;
  }

  int64_t existing = static_cast<int64_t>(shape.size());
  if (existing != rank) {
    MS_LOG(ERROR) << "rank[" << existing << "] must be [" << rank << "]";
    return false;
  }
  return true;
}
}  // namespace

BaseShapePtr InitEmbeddingHashmapFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  auto table_id_shape_ptr = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(table_id_shape_ptr);
  auto table_id_shape = table_id_shape_ptr->GetShapeVector();
  if (!CheckRank(table_id_shape, 0)) {
    MS_LOG(EXCEPTION) << "For " << op_name << " input:0 must be scalar.";
  }
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr InitEmbeddingHashmapFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("table_id", input_args[0]->GetType(), {kInt32}, primitive->name());
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
