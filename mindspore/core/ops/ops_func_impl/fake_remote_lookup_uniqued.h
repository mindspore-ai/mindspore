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

#ifndef MINDSPORE_CORE_OPS_FUNC_IMPL_FAKE_REMOTE_LOOKUP_UNIQUED_H_
#define MINDSPORE_CORE_OPS_FUNC_IMPL_FAKE_REMOTE_LOOKUP_UNIQUED_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
struct FakeRemoteLookupUniquedIndexes {
  size_t table_id{0};
  size_t keys{1};
  size_t actual_keys_num{2};
  size_t unique_indices{3};
  size_t key_count{4};
  size_t max_grad_norm{5};
  size_t embedding_dim{6};
  size_t value_total_len{7};
  size_t initializer_mode{8};
  size_t constant_value{9};
  size_t min{10};
  size_t max{11};
  size_t mu{12};
  size_t sigma{13};
  size_t seed{14};
  size_t seed2{15};
  size_t filter_mode{16};
  size_t filter_freq{17};
  size_t default_key_or_value{18};
  size_t default_key{19};
  size_t default_value{20};
  size_t optimizer_mode{21};
  size_t optimizer_params{22};
  size_t _embedding_dim{23};
  size_t _max_key_num{24};
  size_t _use_counter_filter{25};
  size_t backward_mode{26};
  size_t backward_int_params{27};
  size_t backward_float_params{28};
  size_t backward_bool_params{29};
  size_t parameter{30};
};

class MIND_API FakeRemoteLookupUniquedFuncImpl final : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  int32_t CheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  FakeRemoteLookupUniquedIndexes indexes_;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FUNC_IMPL_FAKE_REMOTE_LOOKUP_UNIQUED_H_
