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

#include "src/runtime/kernel/npu/argmax_npu.h"
#include <memory>
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_ArgMaxFusion;

namespace mindspore::kernel {
int ArgmaxNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  return RET_OK;
}

int ArgmaxNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::ArgMaxExt2(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "New argmax npu operator for " << name_ << " failed.";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  auto axis_const_ = new (std::nothrow) hiai::op::Const(name_ + "_axis");
  if (axis_const_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  ge::TensorDesc tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
  std::shared_ptr<ge::Tensor> ge_tensor =
    std::make_shared<ge::Tensor>(tensor_desc, reinterpret_cast<const uint8_t *>(&(param_->axis_)), sizeof(int));
  if (ge_tensor == nullptr) {
    MS_LOG(ERROR) << "new ge_tensor failed.";
    return RET_ERROR;
  }
  axis_const_->set_attr_value(ge_tensor);
  op_->set_input_axis(*axis_const_);
  op_->set_attr_keep_dims(param_->keep_dims_);
  op_->set_attr_outmaxval(param_->out_value_);
  op_->set_attr_topk(param_->topk_);

  return RET_OK;
}

ge::Operator *mindspore::kernel::ArgmaxNPUKernel::GetNPUOp() { return op_; }

ArgmaxNPUKernel::~ArgmaxNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (axis_const_ != nullptr) {
    delete axis_const_;
    axis_const_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_ArgMaxFusion, NPUKernelCreator<ArgmaxNPUKernel>)
}  // namespace mindspore::kernel
