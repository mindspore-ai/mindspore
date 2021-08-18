/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/op/argmax_npu.h"
#include <memory>

namespace mindspore {
int ArgmaxNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  argmax_ = new (std::nothrow) hiai::op::ArgMaxExt2(name_);
  if (argmax_ == nullptr) {
    MS_LOG(ERROR) << "New argmax npu operator for " << name_ << " failed.";
    return RET_ERROR;
  }
  auto argmax_prim = primitive->value_as_ArgMaxFusion();
  if (argmax_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }

  axis_const_ = new (std::nothrow) hiai::op::Const(name_ + "_axis");
  if (axis_const_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  std::vector<int> axis = {static_cast<int>(argmax_prim->axis())};
  ge::TensorDesc tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
  std::shared_ptr<ge::Tensor> ge_tensor =
    std::make_shared<ge::Tensor>(tensor_desc, reinterpret_cast<const uint8_t *>(axis.data()), sizeof(int));
  if (ge_tensor == nullptr) {
    MS_LOG(ERROR) << "new ge_tensor failed.";
    return RET_ERROR;
  }
  axis_const_->set_attr_value(ge_tensor);
  argmax_->set_input_axis(*axis_const_);

  argmax_->set_attr_keep_dims(argmax_prim->keep_dims());
  argmax_->set_attr_outmaxval(argmax_prim->out_max_value());
  argmax_->set_attr_topk(argmax_prim->top_k());
  return RET_OK;
}

int ArgmaxNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors,
                              const std::vector<ge::Operator *> &npu_inputs) {
  argmax_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *ArgmaxNPUOp::GetNPUOp() { return argmax_; }

ArgmaxNPUOp::~ArgmaxNPUOp() {
  if (argmax_ != nullptr) {
    delete argmax_;
    argmax_ = nullptr;
  }
  if (axis_const_ != nullptr) {
    delete axis_const_;
    axis_const_ = nullptr;
  }
}
}  // namespace mindspore
