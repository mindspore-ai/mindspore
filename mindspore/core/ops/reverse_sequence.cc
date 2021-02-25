/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <set>
#include <vector>
#include <memory>
#include "ops/reverse_sequence.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void ReverseSequence::Init(const int64_t seq_dim, const int64_t batch_dim) {
  this->set_seq_dim(seq_dim);
  this->set_batch_dim(batch_dim);
}
void ReverseSequence::set_seq_dim(const int64_t seq_dim) { this->AddAttr(kSeqDim, MakeValue(seq_dim)); }
void ReverseSequence::set_batch_dim(const int64_t batch_dim) { this->AddAttr(kBatchDim, MakeValue(batch_dim)); }

int64_t ReverseSequence::get_seq_dim() const {
  auto value_ptr = this->GetAttr(kSeqDim);
  return GetValue<int64_t>(value_ptr);
}
int64_t ReverseSequence::get_batch_dim() const {
  auto value_ptr = this->GetAttr(kBatchDim);
  return GetValue<int64_t>(value_ptr);
}
AbstractBasePtr ReverseSequenceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto reverse_prim = primitive->cast<PrimReverseSequence>();
  MS_EXCEPTION_IF_NULL(reverse_prim);
  auto prim_name = reverse_prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  auto seq_lengths =
    CheckAndConvertUtils::ConvertShapePtrToShape("seq_lengths", input_args[1]->BuildShape(), prim_name);
  auto seq_dim = reverse_prim->get_seq_dim();
  auto batch_dim = reverse_prim->get_batch_dim();
  CheckAndConvertUtils::CheckInteger("seq_dim", seq_dim, kLessEqual, input_shape.size(), prim_name);
  CheckAndConvertUtils::CheckInteger("batch_dim", batch_dim, kLessEqual, input_shape.size(), prim_name);
  CheckAndConvertUtils::CheckInteger("batch_dim", batch_dim, kNotEqual, seq_dim, prim_name);
  CheckAndConvertUtils::CheckInteger("seq_lengths rank", seq_lengths.size(), kEqual, 1, prim_name);
  CheckAndConvertUtils::CheckInteger("seq_lengths vector size", seq_lengths[0], kEqual, input_shape[batch_dim],
                                     prim_name);
  // infer type
  std::set<TypeId> tmp(common_valid_types);
  tmp.insert(kNumberTypeBool);
  const std::set<TypeId> valid_x_types(tmp);
  const std::set<TypeId> valid_seq_types = {kNumberTypeInt32, kNumberTypeInt64};
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto seq_type = input_args[1]->BuildType()->cast<TensorTypePtr>()->element();
  CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, valid_x_types, prim_name);
  CheckAndConvertUtils::CheckTensorTypeValid("seq_type", seq_type, valid_seq_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(x_type, input_shape);
}
REGISTER_PRIMITIVE_C(kNameReverseSequence, ReverseSequence);
}  // namespace ops
}  // namespace mindspore
