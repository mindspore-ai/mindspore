/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/base/depth_to_space_base.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;

namespace mindspore::kernel {
int DepthToSpaceBaseCPUKernel::ReSize() {
  if (in_tensors_.at(0)->format() != schema::Format::Format_NHWC) {
    MS_LOG(ERROR) << "depth_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  if (param_->block_size_ <= 0) {
    MS_LOG(ERROR) << "Input block_size should > 0!";
    return RET_PARAM_INVALID;
  }
  auto shape_size = in_tensors_.at(0)->shape().size();
  if (shape_size != DIMENSION_4D) {
    MS_LOG(ERROR) << "Input shape size should be " << DIMENSION_4D;
    return RET_PARAM_INVALID;
  }
  int32_t in_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(in_tensors_.at(0)->shape().data()), in_strides, shape_size);
  param_->in_stride_dim0_ = in_strides[0];
  param_->in_stride_dim1_ = in_strides[1];
  param_->in_stride_dim2_ = in_strides[2];
  int32_t out_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(out_tensors_.at(0)->shape().data()), out_strides, shape_size);
  param_->out_stride_dim0_ = out_strides[0];
  param_->out_stride_dim1_ = out_strides[1];
  param_->out_stride_dim2_ = out_strides[2];
  return RET_OK;
}
}  // namespace mindspore::kernel
