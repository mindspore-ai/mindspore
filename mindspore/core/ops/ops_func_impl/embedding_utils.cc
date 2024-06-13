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

#include "ops/ops_func_impl/embedding_utils.h"

#include <functional>

#include "ops/op_utils.h"
#include "ops/op_enum.h"
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
int32_t CheckEmbeddingOptimizerArgsValidation(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args,
                                              const std::tuple<size_t, size_t, size_t> &indexes,
                                              bool check_max_grad_norm,
                                              const std::tuple<size_t, size_t> &other_indexs) {
  auto &[embdding_dim_idx, keys_idx, grad_idx] = indexes;

  auto embedding_dim_opt = GetScalarValue<int64_t>(input_args[embdding_dim_idx]->GetValue());
  if (MS_UNLIKELY(!embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto embedding_dim = embedding_dim_opt.value();
  if (MS_UNLIKELY(embedding_dim <= 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", the value of embedding_dim must be greater than 0, but got " << embedding_dim << ".";
  }

  const auto &keys_shape = input_args[keys_idx]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(keys_shape))) {
    return OP_CHECK_RETRY;
  }

  auto keys_num = std::accumulate(keys_shape.begin(), keys_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (MS_UNLIKELY(keys_num <= 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", Num of keys should be greater than 0, but got "
                             << keys_num << ".";
  }

  const auto &grad_shape = input_args[grad_idx]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(grad_shape))) {
    return OP_CHECK_RETRY;
  }

  auto grad_num = std::accumulate(grad_shape.begin(), grad_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (MS_UNLIKELY(keys_num * embedding_dim != grad_num)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", Num of grad should be equal to embedding_dim * keys_num, but got grad_num: "
                             << grad_num << ", embedding_dim: " << embedding_dim << ", keys_num: " << keys_num << ".";
  }

  if (!check_max_grad_norm) {
    return OP_CHECK_SUCCESS;
  }

  auto &[amsgrad_idx, max_grad_norm_idx] = other_indexs;
  auto amsgrad_opt = GetScalarValue<bool>(input_args[amsgrad_idx]->GetValue());
  if (MS_UNLIKELY(!amsgrad_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto amsgrad = amsgrad_opt.value();
  if (amsgrad) {
    const auto &max_grad_norm_type = input_args[max_grad_norm_idx]->GetType();
    if (MS_UNLIKELY(max_grad_norm_type->isa<TypeNone>())) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << ", when amsgrad is true, max_grad_norm should not be None.";
    }

    const auto &max_grad_norm_shape = input_args[max_grad_norm_idx]->GetShape()->GetShapeVector();
    auto max_grad_norm_num =
      std::accumulate(max_grad_norm_shape.begin(), max_grad_norm_shape.end(), int64_t(1), std::multiplies<int64_t>());

    if (MS_UNLIKELY(keys_num * embedding_dim != max_grad_norm_num)) {
      MS_EXCEPTION(ValueError)
        << "For " << primitive->name()
        << ", Num of max_grad_norm should be equal to embedding_dim * keys_num, but got max_grad_norm_num: "
        << max_grad_norm_num << ", embedding_dim: " << embedding_dim << ", keys_num: " << keys_num << ".";
    }
  }

  return OP_CHECK_SUCCESS;
}

int32_t CheckEmbeddingOpsExtraArgs(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &embedding_dim_arg = input_args[kInputIndex0];
  const auto &_embedding_dim_arg = input_args[kInputIndex1];
  const auto &keys_arg = input_args[kInputIndex2];
  const auto &_max_key_num_arg = input_args[kInputIndex3];

  auto embdding_dim_opt = GetScalarValue<int64_t>(embedding_dim_arg->GetValue());
  auto _embedding_dim_opt = GetScalarValue<int64_t>(_embedding_dim_arg->GetValue());
  if (MS_UNLIKELY(!embdding_dim_opt.has_value() || !_embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto embedding_dim = embdding_dim_opt.value();
  auto _embedding_dim = _embedding_dim_opt.value();
  if (MS_UNLIKELY(embedding_dim != _embedding_dim)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", embedding_dim should be equal to _embedding_dimm, but got " << embedding_dim
                             << " and " << _embedding_dim << ".";
  }

  const auto &keys_shape = keys_arg->GetShape()->GetShapeVector();
  auto _max_key_num_opt = GetScalarValue<int64_t>(_max_key_num_arg->GetValue());
  if (MS_UNLIKELY(IsDynamic(keys_shape) || !_max_key_num_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto _max_key_num = _max_key_num_opt.value();
  auto keys_num = std::accumulate(keys_shape.begin(), keys_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (MS_UNLIKELY(keys_num != _max_key_num)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", _max_key_num should be equal to keys' num, but got "
                             << _max_key_num << " and " << keys_num << ".";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
