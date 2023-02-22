/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/ctc_greedy_decoder.h"

#include <map>
#include <string>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void CTCGreedyDecoder::Init(const bool merge_repeated) { this->set_merge_repeated(merge_repeated); }

void CTCGreedyDecoder::set_merge_repeated(const bool merge_repeated) {
  (void)this->AddAttr(kMergeRepeated, api::MakeValue(merge_repeated));
}

bool CTCGreedyDecoder::get_merge_repeated() const {
  auto value_ptr = this->GetAttr(kMergeRepeated);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr CTCGreedyDecoderInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  const int64_t kInputsRank = 3;
  const int64_t kSeqLenRank = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto inputs_x_ptr = abstract::CheckArg<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto inputs_x_dtype = input_args[kInputIndex0]->BuildType();
  auto sequence_length_dtype = input_args[kInputIndex1]->BuildType();
  auto inputs_x_shape_ptr = input_args[0]->BuildShape();
  auto sequence_length_shape_ptr = input_args[1]->BuildShape();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("inputs type", inputs_x_dtype, {kFloat32, kFloat64}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sequence length dtype", sequence_length_dtype, {kInt32}, prim_name);

  auto inputs_x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto sequence_length_shape_map =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto inputs_x_shape = inputs_x_shape_map[kShape];
  auto sequence_length_shape = sequence_length_shape_map[kShape];
  if (inputs_x_shape.size() != kInputsRank && !IsDynamicRank(inputs_x_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', inputs's dim must be 3, but got: " << inputs_x_shape.size()
                             << ".";
  }

  if (sequence_length_shape.size() != kSeqLenRank && !IsDynamicRank(sequence_length_shape)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', sequence_length's dims must be 1, but got: " << sequence_length_shape.size() << ".";
  }
  ShapeVector decoded_indices_shape = {-1, 2};
  ShapeVector decoded_indices_max_shape = {inputs_x_shape[0] * inputs_x_shape[1], 2};
  ShapeVector decoded_values_shape = {-1};
  ShapeVector decoded_values_max_shape = {inputs_x_shape[0] * inputs_x_shape[1]};
  ShapeVector decoded_shape_shape = {2};
  ShapeVector log_probability_shape = {inputs_x_shape[1], 1};

  auto decoded_indices = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<mindspore::abstract::Shape>(decoded_indices_shape, decoded_indices_max_shape));
  auto decoded_values = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<mindspore::abstract::Shape>(decoded_values_shape, decoded_values_max_shape));
  auto decoded_shape = std::make_shared<abstract::AbstractTensor>(kInt64, decoded_shape_shape);
  auto log_probability =
    std::make_shared<abstract::AbstractTensor>(inputs_x_ptr->element()->BuildType(), log_probability_shape);

  AbstractBasePtrList ret = {decoded_indices, decoded_values, decoded_shape, log_probability};
  if (inputs_x_shape_ptr->IsDynamic() || sequence_length_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::AbstractTuple>(ret);
  }

  if (inputs_x_shape[1] != sequence_length_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', inputs batch_size must be the same with sequence_length batch_size, "
                             << "but now inputs batch_size: " << inputs_x_shape[1]
                             << " and sequence_length batch_size: " << sequence_length_shape[0] << ".";
  }

  return std::make_shared<abstract::AbstractTuple>(ret);
}
MIND_API_OPERATOR_IMPL(CTCGreedyDecoder, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CTCGreedyDecoder, prim::kPrimCTCGreedyDecoder, CTCGreedyDecoderInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
