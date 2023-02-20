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

#include "ops/data_format_vec_permute.h"

#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
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
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DataFormatVecPermuteInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  if (IsDynamic(x_shape)) {
    return x_shape_ptr;
  }
  std::vector<int64_t> shape1 = {4};
  std::vector<int64_t> shape2 = {4, 2};
  if (x_shape != shape1 && x_shape != shape2) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", input shape must be (4, ) or (4, 2), but got " << x_shape
                             << ".";
  }
  return x_shape_ptr;
}

TypePtr DataFormatVecPermuteInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
}
}  // namespace

void DataFormatVecPermute::Init(const std::string &src_format, const std::string &dst_format) {
  this->set_src_format(src_format);
  this->set_dst_format(dst_format);
}

void DataFormatVecPermute::set_src_format(const std::string &src_format) {
  (void)CheckAndConvertUtils::CheckString(kSrcFormat, src_format, {"NHWC", "NCHW"}, this->name());
  (void)this->AddAttr(kSrcFormat, api::MakeValue(src_format));
}

void DataFormatVecPermute::set_dst_format(const std::string &dst_format) {
  (void)CheckAndConvertUtils::CheckString(kSrcFormat, dst_format, {"NHWC", "NCHW"}, this->name());
  (void)this->AddAttr(kDstFormat, api::MakeValue(dst_format));
}

std::string DataFormatVecPermute::get_src_format() const {
  auto value_ptr = this->GetAttr(kSrcFormat);
  return GetValue<std::string>(value_ptr);
}

std::string DataFormatVecPermute::get_dst_format() const {
  auto value_ptr = this->GetAttr(kDstFormat);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(DataFormatVecPermute, BaseOperator);
AbstractBasePtr DataFormatVecPermuteInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = DataFormatVecPermuteInferType(primitive, input_args);
  auto infer_shape = DataFormatVecPermuteInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGDataFormatVecPermuteInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DataFormatVecPermuteInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DataFormatVecPermuteInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DataFormatVecPermuteInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DataFormatVecPermute, prim::kPrimDataFormatVecPermute, AGDataFormatVecPermuteInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
