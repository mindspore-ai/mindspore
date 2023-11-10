/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/pooling_grad.h"

#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "utils/check_convert_utils.h"
#include "ops/conv_pool_ops.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(PoolingGrad, BaseOperator);
void PoolingGrad::Init(const PoolMode &pool_mode, const std::vector<int64_t> &window,
                       const std::vector<int64_t> &stride, const PadMode &pad_mode,
                       const std::vector<int64_t> &pad_list, const RoundMode &round_mode, const Format &format,
                       const bool global) {
  set_pool_mode(pool_mode);
  set_window(window);
  set_stride(stride);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_round_mode(round_mode);
  set_format(format);
  set_global(global);
}

void PoolingGrad::set_pool_mode(const PoolMode &pool_mode) {
  int64_t swi = pool_mode;
  (void)this->AddAttr(kPoolMode, api::MakeValue(swi));
}

PoolMode PoolingGrad::get_pool_mode() const {
  auto value_ptr = GetAttr(kPoolMode);
  return PoolMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_window(const std::vector<int64_t> &window) {
  (void)this->AddAttr(kWindow, api::MakeValue(window));
}

std::vector<int64_t> PoolingGrad::get_window() const {
  auto value_ptr = GetAttr(kWindow);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_stride(const std::vector<int64_t> &stride) {
  (void)this->AddAttr(kStride, api::MakeValue(stride));
}

std::vector<int64_t> PoolingGrad::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode PoolingGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return PadMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_pad_list(const std::vector<int64_t> &pad_list) {
  (void)this->AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> PoolingGrad::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void PoolingGrad::set_round_mode(const RoundMode &round_mode) {
  int64_t swi = round_mode;
  (void)this->AddAttr(kRoundMode, api::MakeValue(swi));
}

RoundMode PoolingGrad::get_round_mode() const {
  auto value_ptr = GetAttr(kRoundMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return RoundMode(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_format(const Format &format) {
  int64_t swi = format;
  (void)this->AddAttr(kFormat, api::MakeValue(swi));
}

Format PoolingGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return Format(GetValue<int64_t>(value_ptr));
}

void PoolingGrad::set_global(const bool global) { (void)this->AddAttr(kGlobal, api::MakeValue(global)); }

bool PoolingGrad::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

class MIND_API PoolingGradInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: three tensors(y, dy, x).
    constexpr auto kPoolingGradInputNum = 3;
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, kPoolingGradInputNum);
    return input_args[kIndex2]->GetShape()->Clone();
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: three tensors(y, dy, x).
    constexpr auto kPoolingGradInputNum = 3;
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, kPoolingGradInputNum);
    return input_args[kIndex1]->GetType()->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: three tensors(y, dy, x).
    constexpr auto kPoolingGradInputNum = 3;
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, kPoolingGradInputNum);
    auto out_y = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, kIndex0);
    auto d_out = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, kIndex1);
    auto input_x = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, kIndex2);
    (void)abstract::CheckTensorsDTypeSame({out_y, d_out, input_x}, {kInt, kUInt, kFloat},
                                          op_name + "evaluator three inputs should be %s");

    AbstractBasePtr ret = d_out->Broaden();
    auto x_shape = dyn_cast<abstract::TensorShape>(input_args[2]->GetShapeTrack());
    MS_EXCEPTION_IF_NULL(x_shape);
    ret->set_shape(x_shape);
    return ret;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PoolingGrad, prim::kPrimPoolingGrad, PoolingGradInfer, false);
}  // namespace ops
}  // namespace mindspore
