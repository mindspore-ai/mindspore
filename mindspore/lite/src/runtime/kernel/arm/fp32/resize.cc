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

#include <algorithm>
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

int ResizeCPUKernel::ReSize() {
  int ret = RET_OK;
  if (method_ == static_cast<int>(schema::ResizeMethod_BILINEAR)) {
    FreeTmpBuffer();
    ret = MallocTmpBuffer();
    if (ret != RET_OK) {
      FreeTmpBuffer();
      return ret;
    }

    auto input = in_tensors_.at(0);
    auto input_shape = input->shape();
    ret = PrepareResizeBilinear(input_shape.data(), out_tensors_[0]->shape().data(), align_corners_, y_bottoms_,
                                y_tops_, x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_);
    if (ret != RET_OK) {
      FreeTmpBuffer();
    }
  }
  return ret;
}

int ResizeCPUKernel::MallocTmpBuffer() {
  int c = in_tensors_.at(0)->Channel();
  int h = new_height_;
  int w = new_width_;
  y_bottoms_ = reinterpret_cast<int *>(malloc(sizeof(int) * h));
  if (y_bottoms_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  y_tops_ = reinterpret_cast<int *>(malloc(sizeof(int) * h));
  if (y_tops_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  y_bottom_weights_ = reinterpret_cast<float *>(malloc(sizeof(float) * h));
  if (y_bottom_weights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }

  x_lefts_ = reinterpret_cast<int *>(malloc(sizeof(int) * w));
  if (x_lefts_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  x_rights_ = reinterpret_cast<int *>(malloc(sizeof(int) * w));
  if (x_rights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  x_left_weights_ = reinterpret_cast<float *>(malloc(sizeof(float) * w));
  if (x_left_weights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  line_buffer_ = reinterpret_cast<float *>(malloc(sizeof(float) * w * c * 2 * context_->thread_num_));
  if (line_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }

  return RET_OK;
}
void ResizeCPUKernel::FreeTmpBuffer() {
  if (y_bottoms_ != nullptr) {
    free(y_bottoms_);
    y_bottoms_ = nullptr;
  }
  if (y_tops_ != nullptr) {
    free(y_tops_);
    y_tops_ = nullptr;
  }
  if (y_bottom_weights_ != nullptr) {
    free(y_bottom_weights_);
    y_bottom_weights_ = nullptr;
  }

  if (x_lefts_ != nullptr) {
    free(x_lefts_);
    x_lefts_ = nullptr;
  }
  if (x_rights_ != nullptr) {
    free(x_rights_);
    x_rights_ = nullptr;
  }
  if (x_left_weights_ != nullptr) {
    free(x_left_weights_);
    x_left_weights_ = nullptr;
  }
  if (line_buffer_ != nullptr) {
    free(line_buffer_);
    line_buffer_ = nullptr;
  }
}

int ResizeImpl(void *cdata, int task_id) {
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
      int n_h_begin, n_h_end;
      int n = out_tensors_.at(0)->shape()[0];
      int h = new_height_;
      int unit = UP_DIV(n * h, context_->thread_num_);
      n_h_begin = unit * task_id;
      n_h_end = std::min(n_h_begin + unit, n * h);
      int c = in_tensors_.at(0)->shape()[3];
      line0_ = line_buffer_ + new_width_ * c * 2 * task_id;
      line1_ = line0_ + new_width_ * c;
      ret = ResizeBilinear2(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(), y_bottoms_,
                            y_tops_, x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_, line0_, line1_, n_h_begin,
                            n_h_end);

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
  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, ResizeImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
