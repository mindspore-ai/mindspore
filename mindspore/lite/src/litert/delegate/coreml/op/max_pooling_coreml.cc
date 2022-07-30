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

#include "src/litert/delegate/coreml/op/max_pooling_coreml.h"
namespace mindspore::lite {
int MaxPoolingCoreMLOp::InitParams() {
  pooling_prim_ = op_primitive_->value_as_MaxPoolFusion();
  if (pooling_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int MaxPoolingCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  auto pooling_param = op_->mutable_pooling();
  pooling_param->set_type(CoreML::Specification::PoolingLayerParams::MAX);
  if (pooling_prim_->global()) {
    pooling_param->set_globalpooling(true);
    pooling_param->mutable_valid();
    return RET_OK;
  }
  auto kernel_h = static_cast<int>(*(pooling_prim_->kernel_size()->begin()));
  auto kernel_w = static_cast<int>(*(pooling_prim_->kernel_size()->begin() + 1));
  auto stride_h = static_cast<int>(*(pooling_prim_->strides()->begin()));
  auto stride_w = static_cast<int>(*(pooling_prim_->strides()->begin() + 1));
  pooling_param->add_stride(stride_h);
  pooling_param->add_stride(stride_w);
  pooling_param->add_kernelsize(kernel_h);
  pooling_param->add_kernelsize(kernel_w);
  if (pooling_prim_->pad_mode() == schema::PadMode_SAME) {
    pooling_param->mutable_same();
  } else {
    pooling_param->mutable_valid();
    if (pooling_prim_->pad() != nullptr) {
      auto pad_u = static_cast<int>(*(pooling_prim_->pad()->begin() + PAD_UP));
      auto pad_d = static_cast<int>(*(pooling_prim_->pad()->begin() + PAD_DOWN));
      auto pad_l = static_cast<int>(*(pooling_prim_->pad()->begin() + PAD_LEFT));
      auto pad_r = static_cast<int>(*(pooling_prim_->pad()->begin() + PAD_RIGHT));
      auto ret = SetPadding({pad_u, pad_d, pad_l, pad_r});
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Fail to set padding for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  auto act_type = pooling_prim_->activation_type();
  if (act_type != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(act_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set pooling activation failed for op: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
