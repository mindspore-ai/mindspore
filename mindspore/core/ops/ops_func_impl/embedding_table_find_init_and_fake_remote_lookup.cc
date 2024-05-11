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

#include <map>
#include <vector>

#include "ops/ops_func_impl/embedding_table_find_and_init.h"
#include "ops/ops_func_impl/fake_remote_lookup_uniqued.h"
#include "ops/ops_func_impl/embedding_utils.h"
#include "ops/op_utils.h"
#include "ops/op_enum.h"
#include "mindapi/base/types.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kKeyDimOne = 1;
constexpr int64_t kKeyDimTwo = 2;

int32_t CommonCheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                              const std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t> &indexes) {
  auto &[initializer_mode_idx, filter_mode_idx, optimizer_mode_idx, backward_mode_idx, backward_int_params_idx,
         backward_float_params_idx, backward_bool_params_idx] = indexes;
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  auto initializer_mode_opt = GetScalarValue<int64_t>(input_args[initializer_mode_idx]->GetValue());
  if (MS_UNLIKELY(!initializer_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  static std::set<int64_t> valid_initializer_mode_set{InitializerMode::TRUNCATED_NORMAL, InitializerMode::CONSTANT,
                                                      InitializerMode::RANDOM_UNIFORM};
  auto initializer_mode = initializer_mode_opt.value();
  if (MS_UNLIKELY(valid_initializer_mode_set.find(initializer_mode) == valid_initializer_mode_set.end())) {
    MS_EXCEPTION(ValueError)
      << "For " << prim_name
      << ", initializer_mode must be 'truncated_normal', 'constant' or 'random_uniform', but got " << initializer_mode
      << ".";
  }

  auto filter_mode_opt = GetScalarValue<int64_t>(input_args[filter_mode_idx]->GetValue());
  if (MS_UNLIKELY(!filter_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto filter_mode = filter_mode_opt.value();
  if (MS_UNLIKELY(filter_mode != FilterMode::NO_FILTER && filter_mode != FilterMode::COUNTER)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", filter_mode should be counter or no_filter, but got "
                             << filter_mode << ".";
  }

  auto optimizer_mode_opt = GetScalarValue<int64_t>(input_args[optimizer_mode_idx]->GetValue());
  if (MS_UNLIKELY(!optimizer_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  static std::set<int64_t> valid_optimizer_mode_set{OptimizerMode::DEFAULT, OptimizerMode::ADAM, OptimizerMode::ADAMW,
                                                    OptimizerMode::ADAGRAD};
  auto optimizer_mode = optimizer_mode_opt.value();
  if (MS_UNLIKELY(valid_optimizer_mode_set.find(optimizer_mode) == valid_optimizer_mode_set.end())) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", optimizer_mode must be 'adam', 'adamw', 'adagrad' or '', but got " << optimizer_mode
                             << ".";
  }

  auto backward_mode_opt = GetScalarValue<int64_t>(input_args[backward_mode_idx]->GetValue());
  if (MS_UNLIKELY(!backward_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto backward_mode = static_cast<BackwardMode>(backward_mode_opt.value());
  static std::set<int64_t> valid_backward_mode_set{BackwardMode::APPLYA_ADAM, BackwardMode::APPLYA_ADAMW,
                                                   BackwardMode::APPLY_ADA_GRAD, BackwardMode::APPLY_FTRL};
  if (MS_UNLIKELY(valid_backward_mode_set.find(backward_mode) == valid_backward_mode_set.end())) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", backward_mode must be 'adam', 'adamw', 'ftrl' or 'adagrad' , but got "
                             << backward_mode << ".";
  }

  static std::unordered_map<BackwardMode, std::tuple<size_t, size_t, size_t> > expect_backward_params_size{
    {BackwardMode::APPLYA_ADAM, std::make_tuple(1, 6, 1)},
    {BackwardMode::APPLYA_ADAMW, std::make_tuple(1, 7, 3)},
    {BackwardMode::APPLY_ADA_GRAD, std::make_tuple(1, 1, 1)},
    {BackwardMode::APPLY_FTRL, std::make_tuple(0, 4, 1)}};
  const auto &expect_size = expect_backward_params_size[backward_mode];

  auto int_params_opt = GetArrayValue<int64_t>(input_args[backward_int_params_idx]);
  auto float_params_opt = GetArrayValue<pyfloat>(input_args[backward_float_params_idx]);
  auto bool_params_opt = GetArrayValue<bool>(input_args[backward_bool_params_idx]);
  if (MS_UNLIKELY(!float_params_opt.has_value() || !bool_params_opt.has_value() || !int_params_opt.has_value())) {
    MS_LOG(EXCEPTION) << "For " << prim_name
                      << ", failed to get backward_int_params, backward_float_params or backward_bool_params.";
  }
  auto int_params_size = int_params_opt.value().size();
  auto float_params_size = float_params_opt.value().size();
  auto bool_params_size = bool_params_opt.value().size();
  if (MS_UNLIKELY(std::make_tuple(int_params_size, float_params_size, bool_params_size) != expect_size)) {
    MS_LOG(EXCEPTION)
      << "For " << prim_name
      << ", the length of backward_int_params, backward_float_params and backward_bool_params should be "
      << std::get<kIndex0>(expect_size) << ", " << std::get<kIndex1>(expect_size) << " and "
      << std::get<kIndex2>(expect_size) << ", but got " << int_params_size << ", " << float_params_size << " and "
      << bool_params_size << ".";
  }

  return OP_CHECK_SUCCESS;
}

BaseShapePtr CommonInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                              const std::tuple<size_t, size_t, size_t> &indexes) {
  auto &[table_id_idx, keys_idx, embedding_dim_idx] = indexes;

  CheckTensorScalarRank(primitive, input_args[table_id_idx], "table_id");

  std::vector<int64_t> output_shape{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
  auto keys_shape_ptr = input_args[keys_idx]->GetShape();
  MS_EXCEPTION_IF_NULL(keys_shape_ptr);
  const auto &keys_shape = keys_shape_ptr->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(keys_shape))) {
    auto keys_dims = SizeToLong(keys_shape.size());
    MS_CHECK_VALUE(keys_dims >= kKeyDimOne && keys_dims <= kKeyDimTwo,
                   CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>(
                     "rank of keys", SizeToLong(keys_shape.size()), kIncludeBoth, {kKeyDimOne, kKeyDimTwo}, primitive));
    output_shape[0] = keys_shape[0];
  }

  auto embedding_dim_opt = GetScalarValue<int64_t>(input_args[embedding_dim_idx]->GetValue());
  if (MS_LIKELY(embedding_dim_opt.has_value())) {
    auto embedding_dim = embedding_dim_opt.value();
    MS_CHECK_VALUE(embedding_dim > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", embedding_dim,
                                                                                  kGreaterThan, int64_t(0), primitive));
    output_shape[1] = embedding_dim;
  }

  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

TypePtr CommonInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                        const std::tuple<size_t, size_t> &indexes) {
  auto &[table_id_idx, keys_idx] = indexes;
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  const auto &table_id_type = input_args[table_id_idx]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("table_id", table_id_type, {kInt32}, prim_name);

  const auto &keys_type = input_args[keys_idx]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("keys", keys_type, {kInt64}, prim_name);

  return kFloat32;
}
}  // namespace

BaseShapePtr EmbeddingTableFindAndInitFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  return CommonInferShape(primitive, input_args,
                          std::make_tuple(indexes_.table_id, indexes_.keys, indexes_.embedding_dim));
}

TypePtr EmbeddingTableFindAndInitFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  return CommonInferType(primitive, input_args, std::make_tuple(indexes_.table_id, indexes_.keys));
}

int32_t EmbeddingTableFindAndInitFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto ret_normal = CommonCheckValidation(
    primitive, input_args,
    std::make_tuple(indexes_.initializer_mode, indexes_.filter_mode, indexes_.optimizer_mode, indexes_.backward_mode,
                    indexes_.backward_int_params, indexes_.backward_float_params, indexes_.backward_bool_params));
  auto ret_extra =
    CheckEmbeddingOpsExtraArgs(primitive, {input_args[indexes_.embedding_dim], input_args[indexes_._embedding_dim],
                                           input_args[indexes_.keys], input_args[indexes_._max_key_num]});
  if (ret_normal == OP_CHECK_RETRY || ret_extra == OP_CHECK_RETRY) {
    return OP_CHECK_RETRY;
  }
  return OP_CHECK_SUCCESS;
}

BaseShapePtr FakeRemoteLookupUniquedFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  return CommonInferShape(primitive, input_args,
                          std::make_tuple(indexes_.table_id, indexes_.keys, indexes_.embedding_dim));
}

TypePtr FakeRemoteLookupUniquedFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  return CommonInferType(primitive, input_args, std::make_tuple(indexes_.table_id, indexes_.keys));
}

int32_t FakeRemoteLookupUniquedFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto ret_normal = CommonCheckValidation(
    primitive, input_args,
    std::make_tuple(indexes_.initializer_mode, indexes_.filter_mode, indexes_.optimizer_mode, indexes_.backward_mode,
                    indexes_.backward_int_params, indexes_.backward_float_params, indexes_.backward_bool_params));
  auto ret_extra =
    CheckEmbeddingOpsExtraArgs(primitive, {input_args[indexes_.embedding_dim], input_args[indexes_._embedding_dim],
                                           input_args[indexes_.keys], input_args[indexes_._max_key_num]});
  if (ret_normal == OP_CHECK_RETRY || ret_extra == OP_CHECK_RETRY) {
    return OP_CHECK_RETRY;
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
