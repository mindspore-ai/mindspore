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
void ReverseSequence::set_seq_dim(const int64_t seq_dim) { (void)this->AddAttr(kSeqDim, MakeValue(seq_dim)); }
void ReverseSequence::set_batch_dim(const int64_t batch_dim) { (void)this->AddAttr(kBatchDim, MakeValue(batch_dim)); }

int64_t ReverseSequence::get_seq_dim() const { return GetValue<int64_t>(GetAttr(kSeqDim)); }
int64_t ReverseSequence::get_batch_dim() const {
  auto value_ptr = this->GetAttr(kBatchDim);
  return GetValue<int64_t>(value_ptr);
}
AbstractBasePtr ReverseSequenceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto seq_lengths = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto seq_dim = GetValue<int64_t>(primitive->GetAttr(kSeqDim));
  auto batch_dim = GetValue<int64_t>(primitive->GetAttr(kBatchDim));
  (void)CheckAndConvertUtils::CheckInteger("seq_dim", seq_dim, kLessEqual, SizeToLong(input_shape.size()), prim_name);
  (void)CheckAndConvertUtils::CheckInteger("batch_dim", batch_dim, kLessEqual, SizeToLong(input_shape.size()),
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("batch_dim", batch_dim, kNotEqual, seq_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("seq_lengths rank", SizeToLong(seq_lengths.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("seq_lengths vector size", seq_lengths[0], kEqual,
                                           input_shape[LongToSize(batch_dim)], prim_name);
  // infer type
  std::set<TypePtr> valid_x_types(common_valid_types);
  (void)valid_x_types.emplace(kBool);
  const std::set<TypePtr> valid_seq_types = {kInt32, kInt64};
  auto x_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto seq_type = input_args[1]->BuildType()->cast<TensorTypePtr>()->element();
  auto infered_type = CheckAndConvertUtils::CheckTensorTypeValid("x_type", x_type, valid_x_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("seq_type", seq_type, valid_seq_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(infered_type, input_shape);
}
REGISTER_PRIMITIVE_C(kNameReverseSequence, ReverseSequence);
}  // namespace ops
}  // namespace mindspore
