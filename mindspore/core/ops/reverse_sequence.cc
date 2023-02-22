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
#include <algorithm>

#include "ops/reverse_sequence.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReverseSequence, BaseOperator);
void ReverseSequence::Init(const int64_t seq_dim, const int64_t batch_dim) {
  this->set_seq_dim(seq_dim);
  this->set_batch_dim(batch_dim);
}
void ReverseSequence::set_seq_dim(const int64_t seq_dim) { (void)this->AddAttr(kSeqDim, api::MakeValue(seq_dim)); }
void ReverseSequence::set_batch_dim(const int64_t batch_dim) {
  (void)this->AddAttr(kBatchDim, api::MakeValue(batch_dim));
}

int64_t ReverseSequence::get_seq_dim() const { return GetValue<int64_t>(GetAttr(kSeqDim)); }
int64_t ReverseSequence::get_batch_dim() const {
  auto value_ptr = this->GetAttr(kBatchDim);
  return GetValue<int64_t>(value_ptr);
}

namespace {
abstract::ShapePtr ReverseSequenceInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("ReverseSequence", input_args, 0);
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto seq_lengths_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("ReverseSequence", input_args, 1);
  MS_EXCEPTION_IF_NULL(seq_lengths_shape_ptr);
  auto x_shape = x_shape_ptr->shape();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto seq_lengths_shape = seq_lengths_shape_ptr->shape();

  auto seq_dim_ptr = primitive->GetAttr("seq_dim");
  MS_EXCEPTION_IF_NULL(seq_dim_ptr);
  auto seq_dim = GetValue<int64_t>(seq_dim_ptr);
  auto batch_dim_ptr = primitive->GetAttr("batch_dim");
  MS_EXCEPTION_IF_NULL(batch_dim_ptr);
  auto batch_dim = GetValue<int64_t>(batch_dim_ptr);

  if (seq_dim >= SizeToLong(x_shape.size()) || seq_dim < 0) {
    MS_EXCEPTION(ValueError) << "For 'ReverseSequence', the 'seq_dim' must be in range of [0, " << x_shape.size()
                             << "), but got " << seq_dim << " with type 'int'.";
  }
  if (batch_dim >= SizeToLong(x_shape.size()) || batch_dim < 0) {
    MS_EXCEPTION(ValueError) << "For 'ReverseSequence', the 'batch_dim' must be in range of [0, " << x_shape.size()
                             << "), but got " << batch_dim << " with type 'int'.";
  }
  if (batch_dim == seq_dim) {
    MS_EXCEPTION(ValueError) << "For 'ReverseSequence', the 'batch_dim' should be != seq_dim: " << seq_dim
                             << ", but got " << batch_dim << ".";
  }
  if (seq_lengths_shape.size() != 1) {
    MS_EXCEPTION(ValueError) << "For 'ReverseSequence', the 'seq_lengths' rank should be = expected: 1 , but got "
                             << seq_lengths_shape.size() << ".";
  }
  if (seq_lengths_shape[0] != x_shape[batch_dim]) {
    MS_EXCEPTION(ValueError)
      << "For 'ReverseSequence', the 'seq_lengths' vector size should be = input size along batch_dim: "
      << x_shape[batch_dim] << ", but got " << seq_lengths_shape[0] << ".";
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr ReverseSequenceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", the input args used for infer shape and type is necessary, but missing it.";
  }
  const std::set<TypePtr> seq_lengths_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("seq_lengths", input_args[1]->BuildType(), seq_lengths_valid_types,
                                                   prim->name());

  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16,    kUInt32,     kUInt64,
                                           kInt8,    kInt16,   kInt32,   kInt64, kComplex64, kComplex128, kBool};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), x_valid_types, prim->name());
}
}  // namespace
AbstractBasePtr ReverseSequenceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ReverseSequenceInferType(primitive, input_args);
  auto infer_shape = ReverseSequenceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGReverseSequenceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ReverseSequenceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ReverseSequenceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ReverseSequenceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ReverseSequence, prim::kPrimReverseSequence, AGReverseSequenceInfer, false);
}  // namespace ops
}  // namespace mindspore
