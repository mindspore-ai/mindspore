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
#include "ops/grad/gather_d_grad.h"

#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(GatherDGrad, BaseOperator);
class GatherDGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();

    const size_t gather_d_grad_input_num = 2;
    MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == gather_d_grad_input_num,
                               "GatherDGrad's input size should be 2 but got " + std::to_string(input_args.size()));
    for (auto item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    auto shape_attr = primitive->GetAttr(kShape);
    auto shape = CheckAndConvertUtils::CheckIntOrTupleInt("shape", shape_attr, prim_name);
    return std::make_shared<abstract::Shape>(shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const size_t gather_d_grad_input_num = 2;
    MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == gather_d_grad_input_num,
                               "GatherDGrad's input size should be 2 but got " + std::to_string(input_args.size()));
    for (auto item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[kInputIndex0]->BuildType(), {kInt32, kInt64},
                                                     prim_name);
    auto out_type = CheckAndConvertUtils::CheckTensorTypeValid("grad", input_args[kInputIndex1]->BuildType(),
                                                               {kTensorType}, prim_name);
    return out_type;
  }
};

void GatherDGrad::Init(int64_t dim, const std::vector<int64_t> &shape) {
  this->set_dim(dim);
  this->set_shape(shape);
}

void GatherDGrad::set_dim(const int64_t dim) { (void)this->AddAttr(kDim, api::MakeValue(dim)); }

void GatherDGrad::set_shape(const std::vector<int64_t> &shape) { (void)this->AddAttr(kShape, api::MakeValue(shape)); }

int64_t GatherDGrad::get_dim() const {
  auto value_ptr = this->GetAttr(kDim);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> GatherDGrad::get_shape() const {
  auto value_ptr = this->GetAttr(kShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(GatherDGrad, prim::kPrimGatherDGrad, GatherDGradInfer, false);
}  // namespace ops
}  // namespace mindspore
