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
#include "ops/grad/gather_d_grad_v2.h"
#include <memory>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::BaseShapePtr GatherDGradV2InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const size_t gather_d_grad_v2_input_num = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == gather_d_grad_v2_input_num,
                             "GatherDGradV2's input size should be 3 but got " + std::to_string(input_args.size()));
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  return input_args[kInputIndex0]->BuildShape();
}

TypePtr GatherDGradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const size_t gather_d_grad_v2_input_num = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == gather_d_grad_v2_input_num,
                             "GatherDGradV2's input size should be 3 but got " + std::to_string(input_args.size()));
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), {kTensorType},
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64},
                                                   prim_name);
  auto out_type =
    CheckAndConvertUtils::CheckTensorTypeValid("grad", input_args[kInputIndex2]->BuildType(), {kTensorType}, prim_name);
  return out_type;
}
}  // namespace

void GatherDGradV2::Init(int64_t dim) { this->set_dim(dim); }

void GatherDGradV2::set_dim(const int64_t dim) { (void)this->AddAttr(kDim, api::MakeValue(dim)); }

int64_t GatherDGradV2::get_dim() const {
  auto value_ptr = this->GetAttr(kDim);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr GatherDGradV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = GatherDGradV2InferType(primitive, input_args);
  auto infer_shape = GatherDGradV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(GatherDGradV2, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(GatherDGradV2, prim::kPrimGatherDGradV2, GatherDGradV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
