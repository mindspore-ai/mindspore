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

#include "src/litert/delegate/coreml/op/arithmetic_coreml.h"
namespace mindspore::lite {
int ArithmeticCoreMLOp::IsSupport() {
  MS_CHECK_TRUE_MSG(in_tensors_.size() == kInputSize1, RET_NOT_SUPPORT, "Arithmetic op only support two inputs.");
  auto input_a = in_tensors_.at(0);
  auto input_b = in_tensors_.at(1);
  if ((input_a.IsConst() && input_a.ElementNum() == 1) || (input_b.IsConst() && input_b.ElementNum() == 1)) {
    use_normal_ = true;
    int const_idx = input_a.IsConst() ? 0 : 1;
    auto org_data = in_tensors_[const_idx].MutableData();
    CHECK_NULL_RETURN(org_data);
    const_value_ = reinterpret_cast<float *>(org_data)[0];
  }
  return RET_OK;
}

int ArithmeticCoreMLOp::BuildLayer() {
  if (use_normal_) {
    auto ret = BuildNormalArithmetic();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Build normal arithmetic layer failed for op: " << name_;
      return RET_ERROR;
    }
    return RET_OK;
  }
  auto ret = BuildBroadcastableArithmetic();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build broadcastable arithmetic layer failed for op: " << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticCoreMLOp::BuildNormalArithmetic() {
  MS_ASSERT(op_ != nullptr);
  switch (type_) {
    case schema::PrimitiveType_AddFusion: {
      auto add_param = op_->mutable_add();
      add_param->set_alpha(const_value_);
      break;
    }
    case schema::PrimitiveType_SubFusion: {
      auto add_param = op_->mutable_add();
      add_param->set_alpha(-const_value_);
      break;
    }
    case schema::PrimitiveType_MulFusion: {
      auto mul_param = op_->mutable_multiply();
      mul_param->set_alpha(const_value_);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported arithmetic type.";
      return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticCoreMLOp::BuildBroadcastableArithmetic() {
  MS_ASSERT(op_ != nullptr);
  switch (type_) {
    case schema::PrimitiveType_AddFusion:
      (void)op_->mutable_addbroadcastable();
      break;
    case schema::PrimitiveType_SubFusion:
      (void)op_->mutable_subtractbroadcastable();
      break;
    case schema::PrimitiveType_MulFusion:
      (void)op_->mutable_multiplybroadcastable();
      break;
    default:
      MS_LOG(ERROR) << "Unsupported arithmetic type.";
      return RET_ERROR;
  }
  for (const auto &in_tensor : in_tensors_) {
    if (in_tensor.IsConst()) {
      auto ret = SetConstInput(in_tensor);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set const input failed for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

void ArithmeticCoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  for (const auto &in_tensor : in_tensors_) {
    if (in_tensor.IsConst() && !use_normal_) {
      // note that const op has not input
      const_ops_[in_tensor.Name()]->add_output(in_tensor.Name());
    }
    if (!(in_tensor.IsConst() && use_normal_)) {
      op_->add_input(in_tensor.Name());
    }
  }
  op_->add_output(out_tensors_[0].Name());
}
}  // namespace mindspore::lite
