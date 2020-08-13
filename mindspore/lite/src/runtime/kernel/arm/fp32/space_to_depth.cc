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

#include "src/runtime/kernel/arm/fp32/space_to_depth.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32/space_to_depth.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::kernel {

int SpaceToDepthCPUKernel::Init() {
  SpaceToDepthParameter *param = reinterpret_cast<SpaceToDepthParameter *>(op_parameter_);
  if (param->block_size_ <= 0) {
    MS_LOG(ERROR) << "Input block_size should > 0!";
    return RET_PARAM_INVALID;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToDepthCPUKernel::ReSize() {
  if (in_tensors_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return RET_FORMAT_ERR;
  }

  num_unit_ = static_cast<int>(in_tensors_[0]->shape().at(kNHWC_H));
  thread_h_num_ = MSMIN(op_parameter_->thread_num_, num_unit_);
  thread_h_stride_ = UP_DIV(num_unit_, thread_h_num_);
  return RET_OK;
}

int SpaceToDepthCPUKernel::SpaceToDepth(int task_id) {
  int num_unit_thread = MSMIN(thread_h_stride_, num_unit_ - task_id * thread_h_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_h_stride_;
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  SpaceToDepthParameter *param = reinterpret_cast<SpaceToDepthParameter *>(op_parameter_);
  auto ret = SpaceToDepthForNHWC(input_ptr_, output_ptr_, in_shape.data(), out_shape.data(), in_shape.size(),
                                 param->block_size_, thread_offset, thread_offset + num_unit_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SpaceToDepth error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SpaceToDepthRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<SpaceToDepthCPUKernel *>(cdata);
  auto ret = g_kernel->SpaceToDepth(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SpaceToDepthRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SpaceToDepthCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  input_ptr_ = reinterpret_cast<float *>(in_tensors_[0]->Data());
  output_ptr_ = reinterpret_cast<float *>(out_tensors_[0]->Data());
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC) {
    ret = LiteBackendParallelLaunch(SpaceToDepthRun, this, thread_h_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SpaceToDepth error error_code[" << ret << "]";
      return ret;
    }
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Only support NHWC now!";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSpaceToDepthFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SpaceToDepthCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SpaceToDepthCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SpaceToDepth, CpuSpaceToDepthFp32KernelCreator)
}  // namespace mindspore::kernel
