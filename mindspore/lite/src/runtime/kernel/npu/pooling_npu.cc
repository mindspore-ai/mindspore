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

#include "src/runtime/kernel/npu/pooling_npu.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_AvgPoolFusion;
using mindspore::schema::PrimitiveType_MaxPoolFusion;

namespace mindspore::kernel {
int PoolingNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                OpParameter *opParameter) {
  if (pooling_param_->pad_l_ > pooling_param_->stride_w_ || pooling_param_->pad_u_ > pooling_param_->stride_h_) {
    MS_LOG(ERROR) << "Npu pooling does not support pad > stride.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingNPUKernel::SetPoolingParam() {
  if (pooling_param_->pool_mode_ == PoolMode_MaxPool) {
    pooling_->set_attr_mode(0);
  } else if (pooling_param_->pool_mode_ == PoolMode_AvgPool) {
    pooling_->set_attr_mode(1);
  } else {
    pooling_->set_attr_mode(2);
  }
  pooling_->set_attr_global_pooling(pooling_param_->global_);
  pooling_->set_attr_window({pooling_param_->window_h_, pooling_param_->window_w_});
  pooling_->set_attr_stride({pooling_param_->stride_h_, pooling_param_->stride_w_});
  if (pooling_param_->pad_mode_ == Pad_same) {
    pooling_->set_attr_pad_mode(6);
    pooling_->set_attr_pad({0, 0, 0, 0});
  } else if (pooling_param_->pad_mode_ == Pad_valid) {
    pooling_->set_attr_pad_mode(5);
    pooling_->set_attr_pad({0, 0, 0, 0});
  } else {
    pooling_->set_attr_pad_mode(0);
    pooling_->set_attr_pad(
      {pooling_param_->pad_u_, pooling_param_->pad_d_, pooling_param_->pad_l_, pooling_param_->pad_r_});
  }

  if (pooling_param_->round_mode_ == RoundMode_Floor) {  // no use in cpu
    pooling_->set_attr_ceil_mode(0);
    pooling_->set_attr_data_mode(1);
  } else {
    pooling_->set_attr_ceil_mode(1);
    pooling_->set_attr_data_mode(0);
  }
  return RET_OK;
}

int PoolingNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs,
                                   const std::vector<ge::Operator *> &npu_inputs) {
  pooling_ = new (std::nothrow) hiai::op::PoolingD(name_ + "_pooling");
  if (pooling_ == nullptr) {
    MS_LOG(ERROR) << "New pooling npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto ret = SetPoolingParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  pooling_->set_input_x(*npu_inputs[0]);

  if (pooling_param_->act_type_ != ActType_No) {
    ret = SetActivation(pooling_, pooling_param_->act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::PoolingNPUKernel::GetNPUOp() {
  if (pooling_param_->act_type_ == ActType_No) {
    return pooling_;
  } else {
    return act_;
  }
}

PoolingNPUKernel::~PoolingNPUKernel() {
  if (pooling_ != nullptr) {
    delete pooling_;
    pooling_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_MaxPoolFusion, NPUKernelCreator<PoolingNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_AvgPoolFusion, NPUKernelCreator<PoolingNPUKernel>)
}  // namespace mindspore::kernel
