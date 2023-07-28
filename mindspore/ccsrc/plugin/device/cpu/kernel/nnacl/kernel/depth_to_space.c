/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/depth_to_space.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/depth_to_space_parameter.h"
#include "nnacl/base/depth_to_space_base.h"

int DepthToSpaceResize(KernelBase *self) {
  DepthToSpaceStruct *depth_to_space = (DepthToSpaceStruct *)self;
  DepthToSpaceArgs *args = &depth_to_space->args_;

  TensorC *input = self->in_[FIRST_INPUT];
  int32_t in_strides[DIMENSION_4D] = {0};
  ComputeStrides(input->shape_, in_strides, input->shape_size_);
  args->in_stride_dim0_ = in_strides[Index0];
  args->in_stride_dim1_ = in_strides[Index1];
  args->in_stride_dim2_ = in_strides[Index2];

  TensorC *output = self->out_[OUTPUT_INDEX];
  int32_t out_strides[DIMENSION_4D] = {0};
  ComputeStrides(output->shape_, out_strides, output->shape_size_);
  args->out_stride_dim0_ = out_strides[Index0];
  args->out_stride_dim1_ = out_strides[Index1];
  args->out_stride_dim2_ = out_strides[Index2];
  return NNACL_OK;
}

int DepthToSpaceCompute(KernelBase *self) {
  DepthToSpaceStruct *depth_to_space = (DepthToSpaceStruct *)self;
  int mode = ((DepthToSpaceParameter *)self->param_)->mode_;

  TensorC *input = self->in_[FIRST_INPUT];
  TensorC *output = self->out_[OUTPUT_INDEX];

  if (mode == 0) {
    // RCD
    DepthToSpaceForNHWC(input->data_, output->data_, input->shape_, &depth_to_space->args_);
  } else if (mode == 1) {
    // CRD
    DepthToSpaceCRDForNHWC(input->data_, output->data_, input->shape_, &depth_to_space->args_);
  } else {
    return NNACL_DEPTH_TO_SPACE_INVALID_MODE;
  }
  return NNACL_OK;
}

KernelBase *CreateDepthToSpace(OpParameter *param, int data_type) {
  DepthToSpaceStruct *depth_to_space = (DepthToSpaceStruct *)malloc(sizeof(DepthToSpaceStruct));
  NNACL_CHECK_NULL_RETURN_NULL(depth_to_space);
  memset(depth_to_space, 0, sizeof(DepthToSpaceStruct));

  depth_to_space->args_.data_type_size_ = DataTypeCSize(data_type);
  depth_to_space->args_.block_size_ = ((DepthToSpaceParameter *)param)->block_size_;
  depth_to_space->base_.Release = DefaultRelease;
  depth_to_space->base_.Prepare = DefaultPrepare1In1Out;
  depth_to_space->base_.Resize = DepthToSpaceResize;
  depth_to_space->base_.Compute = DepthToSpaceCompute;
  return (KernelBase *)depth_to_space;
}

REG_KERNEL_CREATOR(PrimType_DepthToSpace, kNumberTypeFloat32, CreateDepthToSpace)
REG_KERNEL_CREATOR(PrimType_DepthToSpace, kNumberTypeFloat16, CreateDepthToSpace)
