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

#include "src/runtime/kernel/arm/fp32/resize.h"
#include "schema/model_generated.h"
#include "nnacl/fp32/resize.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ResizeCPUKernel::Init() {
  auto ret = ResizeBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto resize = reinterpret_cast<ResizeCPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeCPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto input_data = reinterpret_cast<float *>(input->Data());
  if (input_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();
  if (context_ == nullptr) {
    return RET_NULL_PTR;
  }

  int ret = 0;
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_BILINEAR): {
      ret = ResizeBilinear(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(), align_corners_,
                           task_id, context_->thread_num_);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST_NEIGHBOR): {
      if (align_corners_) {
        MS_LOG(ERROR) << "ResizeNearestNeighbor not support align_corners.";
        return RET_ERROR;
      }
      ret = ResizeNearestNeighbor(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(), task_id,
                                  context_->thread_num_);
      break;
    }
    case schema::ResizeMethod_UNKNOW:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      ret = RET_ERROR;
    }
  }
  return ret;
}

int ResizeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  int error_code = LiteBackendParallelLaunch(ResizeImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
