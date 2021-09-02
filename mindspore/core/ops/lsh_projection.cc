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

#include "ops/lsh_projection.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void LshProjection::Init(const LshProjectionType &type) { set_type(type); }

void LshProjection::set_type(const LshProjectionType &type) {
  int64_t swi = (int64_t)type;
  (void)AddAttr(kType, MakeValue(swi));
}

LshProjectionType LshProjection::get_type() const { return LshProjectionType(GetValue<int64_t>(GetAttr(kType))); }

AbstractBasePtr LshProjectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  const int64_t input0_size = 2;
  const int64_t input0_last_dim = 32;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);
  auto input0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto input1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input0 rank", SizeToLong(input0.size()), kEqual, input0_size, op_name);
  (void)CheckAndConvertUtils::CheckInteger("input0_shape_dimen_1", input0[1], kLessEqual, input0_last_dim, op_name);
  (void)CheckAndConvertUtils::CheckInteger("input1 rank", SizeToLong(input1.size()), kGreaterEqual, 1, op_name);

  if (input_args.size() == 3) {
    auto input2 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("input2 rank", SizeToLong(input2.size()), kEqual, 1, op_name);
    (void)CheckAndConvertUtils::CheckInteger("input2_shape_dimen_0", input2[0], kEqual, input1[0], op_name);
  }

  std::vector<int64_t> out_shape;
  switch ((int64_t)LshProjectionType(GetValue<int64_t>(primitive->GetAttr(kType)))) {
    case (int64_t)LshProjectionType::SPARSE:
      out_shape.push_back(input0[0]);
      break;
    case (int64_t)LshProjectionType::DENSE:
      out_shape.push_back(input0[0] * input0[1]);
      break;
  }
  return std::make_shared<abstract::AbstractTensor>(kInt32, out_shape);
}
REGISTER_PRIMITIVE_C(kNameLshProjection, LshProjection);
}  // namespace ops
}  // namespace mindspore
