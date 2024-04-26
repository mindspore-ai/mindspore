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
#include "ops/ops_func_impl/chunk.h"
#include <string>
#include <algorithm>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int64_t GetChunksNum(const ValuePtr &chunks_value, const std::string &prim_name) {
  auto chunks_opt = GetScalarValue<int64_t>(chunks_value);
  if (MS_UNLIKELY(!chunks_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", the chunk is a unknown variable, which is not supported now.";
  }
  return chunks_opt.value();
}

int64_t GetInputTensorRank(const ShapeVector &input_shape, const std::string &prim_name) {
  if (IsDynamicRank(input_shape)) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", the input tensor is with dynamic rank, which is not supported now.";
  }
  return SizeToLong(input_shape.size());
}

int64_t GetChunksDim(const ValuePtr &dim_value, const ShapeVector &input_shape, const std::string &prim_name) {
  auto dim_opt = GetScalarValue<int64_t>(dim_value);
  if (!dim_opt.has_value()) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", the dims is a unknown variable, which is not supported now.";
  }
  auto dim = dim_opt.value();
  if (dim < 0) {
    dim += SizeToLong(input_shape.size());
  }
  return dim;
}
}  // namespace

BaseShapePtr ChunkFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const auto &input_shape_ptr = input_args[kIndex0]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &chunks_value = input_args[kIndex1]->GetValue();
  const auto &dim_value = input_args[kIndex2]->GetValue();

  auto chunks = GetChunksNum(chunks_value, prim_name);
  auto rank = GetInputTensorRank(input_shape, prim_name);
  auto dim = GetChunksDim(dim_value, input_shape, prim_name);
  auto dim_size = input_shape[dim];
  if (dim_size == abstract::Shape::kShapeDimAny) {
    MS_LOG(EXCEPTION) << "For " << prim_name
                      << ", the dimension corresponds to the specified 'dims' is dynamic, which "
                         "is not supported now.";
  }
  int64_t each_size = (dim_size + chunks - 1) / chunks;

  std::vector<abstract::BaseShapePtr> output_list{};
  if (each_size == 0 && dim_size == 0) {
    for (int64_t i = 0; i < chunks; ++i) {
      output_list.push_back(std::make_shared<abstract::TensorShape>(std::vector<int64_t>(0)));
    }
    return std::make_shared<abstract::TupleShape>(std::move(output_list));
  }
  auto actual_chunks = std::max<int64_t>((dim_size + each_size - 1) / each_size, 1);
  auto last_split_size = each_size - (each_size * actual_chunks - dim_size);
  for (int64_t i = 0; i < actual_chunks - 1; i++) {
    std::vector<int64_t> each_shape{};
    for (int64_t j = 0; j < rank; j++) {
      if (j == dim) {
        (void)each_shape.emplace_back(each_size);
      } else {
        (void)each_shape.emplace_back(input_shape[j]);
      }
    }
    (void)output_list.emplace_back(std::make_shared<abstract::TensorShape>(each_shape));
  }
  // handle last split size
  std::vector<int64_t> last_shape{};
  for (int64_t j = 0; j < rank; j++) {
    if (j == dim) {
      (void)last_shape.emplace_back(last_split_size);
    } else {
      (void)last_shape.emplace_back(input_shape[j]);
    }
  }
  (void)output_list.emplace_back(std::make_shared<abstract::TensorShape>(last_shape));
  return std::make_shared<abstract::TupleShape>(std::move(output_list));
}

TypePtr ChunkFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const auto &input_shape_ptr = input_args[kIndex0]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &chunks_value = input_args[kIndex1]->GetValue();
  const auto &dim_value = input_args[kIndex2]->GetValue();

  auto chunks = GetChunksNum(chunks_value, prim_name);
  auto dim = GetChunksDim(dim_value, input_shape, prim_name);
  auto dim_size = input_shape[dim];
  if (dim_size == abstract::Shape::kShapeDimAny) {
    MS_LOG(EXCEPTION) << "For " << prim_name
                      << ", the dimension corresponds to the specified 'dims' is dynamic, which "
                         "is not supported now.";
  }
  int64_t each_size = (dim_size + chunks - 1) / chunks;

  const auto &infer_type = input_args[0]->GetType();
  if (each_size == 0 && dim_size == 0) {
    std::vector<TypePtr> output_type_list(chunks, infer_type->Clone());
    return std::make_shared<Tuple>(std::move(output_type_list));
  }
  auto actual_chunks = std::max<int64_t>((dim_size + each_size - 1) / each_size, 1);
  std::vector<TypePtr> output_type_list(actual_chunks, infer_type->Clone());
  return std::make_shared<Tuple>(std::move(output_type_list));
}

int32_t ChunkFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  int32_t check_status = OP_CHECK_SUCCESS;
  // Check chunks valid.
  auto chunks_value = input_args[kIndex1]->GetValue();
  auto chunks_opt = GetScalarValue<int64_t>(chunks_value);
  if (MS_UNLIKELY(!chunks_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto chunks = chunks_opt.value();
    if (chunks <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', chunks must be positive, but got " << chunks << ".";
    }
  }
  // Check dim valid.
  auto x_shape_ptr = input_args[kIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  // Skip to check dim valid if input is dynamic rank.
  if (IsDynamicRank(x_shape)) {
    return check_status;
  }
  auto rank = SizeToLong(x_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, primitive));

  auto dim_value = input_args[kIndex2]->GetValue();
  auto dim_opt = GetScalarValue<int64_t>(dim_value);
  if (MS_UNLIKELY(!dim_opt.has_value())) {
    check_status = OP_CHECK_RETRY;
  } else {
    const auto dim = dim_opt.value();
    if (dim >= rank || dim < -rank) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', dim must in [" << -rank << " , " << rank << "), but got "
                               << dim << ".";
    }
  }
  return check_status;
}

}  // namespace ops
}  // namespace mindspore
