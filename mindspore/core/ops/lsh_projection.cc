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
  AddAttr(kType, MakeValue(swi));
}

LshProjectionType LshProjection::get_type() const {
  auto value_ptr = GetAttr(kType);
  return LshProjectionType(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr LshProjectionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto LshProjection_prim = primitive->cast<PrimLshProjectionPtr>();
  MS_EXCEPTION_IF_NULL(LshProjection_prim);
  //  if (input_args.size() != 2 && input_args.size() != 3) {
  //    MS_LOG(ERROR) << "inputs to LshProjection operator should be 2 or 3, but " << input_args.size() << " is given.";
  //  }
  auto op_name = LshProjection_prim->name();
  auto input0 = CheckAndConvertUtils::ConvertShapePtrToShape("input0_shape", input_args[0]->BuildShape(), op_name);
  auto input1 = CheckAndConvertUtils::ConvertShapePtrToShape("input1_shape", input_args[1]->BuildShape(), op_name);
  CheckAndConvertUtils::CheckInteger("input0_shape", input0.size(), kEqual, 2, op_name);
  CheckAndConvertUtils::CheckInteger("input0_shape_dimen_1", input0[1], kLessEqual, 32, op_name);
  CheckAndConvertUtils::CheckInteger("input1_shape", input1.size(), kGreaterEqual, 1, op_name);

  if (input_args.size() == 3) {
    auto input2 = CheckAndConvertUtils::ConvertShapePtrToShape("input2_shape", input_args[2]->BuildShape(), op_name);
    CheckAndConvertUtils::CheckInteger("input2_shape", input2.size(), kEqual, 1, op_name);
    CheckAndConvertUtils::CheckInteger("input2_shape_dimen_0", input2[0], kEqual, input1[0], op_name);
  }

  std::vector<int64_t> out_shape;
  switch ((int64_t)LshProjection_prim->get_type()) {
    case (int64_t)LshProjectionType::SPARSE:
      out_shape.push_back(input0[0]);
      break;
    case (int64_t)LshProjectionType::DENSE:
      out_shape.push_back(input0[0] * input0[1]);
      break;
  }
  TypePtr infer_type = TypeIdToType(kNumberTypeInt32);
  return std::make_shared<abstract::AbstractTensor>(infer_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameLshProjection, LshProjection);
}  // namespace ops
}  // namespace mindspore
