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
#include "src/litert/kernel/cpu/fp32/space_to_depth_fp32.h"
#include "nnacl/errorcode.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/base/space_to_depth_base.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::kernel {
int SpaceToDepthCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kInputIndex]);
  CHECK_NULL_RETURN(out_tensors_[kOutputIndex]);

  if (param_->block_size_ <= 0) {
    MS_LOG(ERROR) << "Input block_size should > 0!";
    return RET_PARAM_INVALID;
  }
  param_->date_type_len =
    in_tensors_[kInputIndex]->data_type() == kNumberTypeFloat16 ? FP16_DATA_TYPE_LEN : sizeof(float);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToDepthCPUKernel::ReSize() {
  if (in_tensors_[kInputIndex]->format() != mindspore::NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  return RET_OK;
}

int SpaceToDepthCPUKernel::SpaceToDepth(int task_id) {
  auto input = in_tensors_[kInputIndex];
  auto output = out_tensors_[kOutputIndex];
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  auto input_data = input->data();
  auto output_data = output->data();
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(output_data);
  auto ret =
    SpaceToDepthForNHWC(input_data, output_data, in_shape.data(), out_shape.data(), in_shape.size(), param_, task_id);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "do SpaceToDepthForNHWC failed. " << this->name();
    return RET_ERROR;
  }
  return RET_OK;
}

int SpaceToDepthRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = static_cast<SpaceToDepthCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->SpaceToDepth(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SpaceToDepth failed, ret: " << ret;
  }
  return ret;
}

int SpaceToDepthCPUKernel::Run() {
  auto ret = ParallelLaunch(ms_context_, SpaceToDepthRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed, ret: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToDepth, LiteKernelCreator<SpaceToDepthCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SpaceToDepth, LiteKernelCreator<SpaceToDepthCPUKernel>)
#endif
}  // namespace mindspore::kernel
