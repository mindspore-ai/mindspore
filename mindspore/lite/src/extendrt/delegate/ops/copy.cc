/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ops/copy.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"
#include "src/common/utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Copy, BaseOperator);

void Copy::set_copy_format(CopyFormatType format) { this->AddAttr(kCopyFormat, api::MakeValue(format)); }

int Copy::get_copy_format() const {
  auto value_ptr = GetAttr(kCopyFormat);
  return static_cast<int>(GetValue<int64_t>(value_ptr));
}

class CopyInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override;
};

BaseShapePtr CopyInfer::InferShape(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->BuildShape();
}

TypePtr CopyInfer::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto format_ptr = primitive->GetAttr(kCopyFormat);
  auto oper = static_cast<int>(GetValue<int64_t>(format_ptr));
  TypePtr in_type = input_args[kInputIndex0]->BuildType();
  TypePtr res = in_type;
  if ((in_type == kFloat32) && (oper == Copy::CopyFormatType::HOST_DEVICE)) {
    res = kFloat16;
  }
  if ((in_type == kFloat16) && (oper == Copy::CopyFormatType::DEVICE_HOST)) {
    res = kFloat32;
  }
  return res;
}

AbstractBasePtr CopyInfer::InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto type = InferType(primitive, input_args);
  auto shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

GVAR_DEF(PrimitivePtr, kPrimCopy, std::make_shared<Primitive>(kNameCopy));
REGISTER_PRIMITIVE_OP_INFER_IMPL(Copy, kPrimCopy, CopyInfer, false);
}  // namespace ops
}  // namespace mindspore
