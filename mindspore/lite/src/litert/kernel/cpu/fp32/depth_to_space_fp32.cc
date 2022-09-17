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
#include "src/litert/kernel/cpu/fp32/depth_to_space_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/base/depth_to_space_base.h"
#include "nnacl/nnacl_common.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {
int DepthToSpaceCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  param_->data_type_size_ = sizeof(float);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DepthToSpaceCPUKernel::ReSize() {
  if (in_tensors_[0]->format() != mindspore::NHWC) {
    MS_LOG(ERROR) << "depth_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  if (param_->block_size_ <= 0) {
    MS_LOG(ERROR) << "Input block_size should > 0!";
    return RET_PARAM_INVALID;
  }
  auto shape_size = in_tensors_[0]->shape().size();
  if (shape_size != DIMENSION_4D) {
    MS_LOG(ERROR) << "Input shape size should be " << DIMENSION_4D;
    return RET_PARAM_INVALID;
  }
  if (out_tensors_[0]->shape().size() != DIMENSION_4D) {
    MS_LOG(ERROR) << "OutPut shape size should be " << DIMENSION_4D;
    return RET_PARAM_INVALID;
  }
  if (out_tensors_[kOutputIndex]->shape().size() != DIMENSION_4D) {
    MS_LOG(ERROR) << "Output shape size should be " << DIMENSION_4D;
    return RET_ERROR;
  }
  int32_t in_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(in_tensors_[0]->shape().data()), in_strides, shape_size);
  param_->in_stride_dim0_ = in_strides[0];
  param_->in_stride_dim1_ = in_strides[1];
  param_->in_stride_dim2_ = in_strides[2];
  int32_t out_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(out_tensors_[0]->shape().data()), out_strides, shape_size);
  param_->out_stride_dim0_ = out_strides[0];
  param_->out_stride_dim1_ = out_strides[1];
  param_->out_stride_dim2_ = out_strides[2];
  return RET_OK;
}

int DepthToSpaceCPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const void *input_data = input->data();
  void *output_data = output->data();
  auto in_shape = input->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == DIMENSION_4D, RET_ERROR, "input shape should be 4!");
  if (input->format() == mindspore::NHWC) {
    DepthToSpaceForNHWC(input_data, output_data, in_shape.data(), param_);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Depth_to_space only support NHWC now!";
    return RET_ERROR;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DepthToSpace, LiteKernelCreator<DepthToSpaceCPUKernel>)
}  // namespace mindspore::kernel
