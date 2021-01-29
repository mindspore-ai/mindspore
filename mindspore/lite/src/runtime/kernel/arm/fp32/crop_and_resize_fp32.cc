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

#include "src/runtime/kernel/arm/fp32/crop_and_resize_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/fp32/resize_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CropAndResize;

namespace mindspore::kernel {
int CropAndResizeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CropAndResizeCPUKernel::ReSize() {
  new_height_ = out_tensors_.at(0)->shape()[1];
  new_width_ = out_tensors_.at(0)->shape()[2];
  return RET_OK;
}

int CropAndResizeCPUKernel::MallocTmpBuffer() {
  batch_ = out_tensors_[0]->Batch();
  // Malloc buffer to save coordinate.
  // For mode CROP_AND_RESIZE, different output batches require different cache coordinates.
  int c = in_tensors_.at(0)->Channel();
  y_bottoms_ = reinterpret_cast<int *>(context_->allocator->Malloc(sizeof(int) * new_height_ * batch_));
  if (y_bottoms_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  y_tops_ = reinterpret_cast<int *>(context_->allocator->Malloc(sizeof(int) * new_height_ * batch_));
  if (y_tops_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  y_bottom_weights_ = reinterpret_cast<float *>(context_->allocator->Malloc(sizeof(float) * new_height_ * batch_));
  if (y_bottom_weights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }

  x_lefts_ = reinterpret_cast<int *>(context_->allocator->Malloc(sizeof(int) * new_width_ * batch_));
  if (x_lefts_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  x_rights_ = reinterpret_cast<int *>(context_->allocator->Malloc(sizeof(int) * new_width_ * batch_));
  if (x_rights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  x_left_weights_ = reinterpret_cast<float *>(context_->allocator->Malloc(sizeof(float) * new_width_ * batch_));
  if (x_left_weights_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  line_buffer_ =
    reinterpret_cast<float *>(context_->allocator->Malloc(sizeof(float) * new_width_ * c * 2 * context_->thread_num_));
  if (line_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_NULL_PTR;
  }
  return RET_OK;
}

void CropAndResizeCPUKernel::FreeTmpBuffer() {
  context_->allocator->Free(y_bottoms_);
  context_->allocator->Free(y_tops_);
  context_->allocator->Free(y_bottom_weights_);
  context_->allocator->Free(x_lefts_);
  context_->allocator->Free(x_rights_);
  context_->allocator->Free(x_left_weights_);
  context_->allocator->Free(line_buffer_);
}

int CropAndResizeImpl(void *cdata, int task_id) {
  auto resize = reinterpret_cast<CropAndResizeCPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "CropAndResize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropAndResizeCPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto input_data = reinterpret_cast<float *>(input->data_c());
  if (input_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->data_c());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();
  int unit = UP_DIV(new_height_, context_->thread_num_);
  int h_begin = unit * task_id;
  int h_end = MSMIN(h_begin + unit, new_height_);
  if (h_end <= h_begin) {
    return RET_OK;
  }
  int c = in_tensors_.at(0)->shape().at(3);
  float *line0 = line_buffer_ + new_width_ * c * 2 * task_id;
  float *line1 = line0 + new_width_ * c;
  auto ret = CropAndResizeBilinear(input_data, output_data, input_shape.data(), out_tensors_.at(0)->shape().data(),
                                   y_bottoms_, y_tops_, x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_, line0,
                                   line1, h_begin, h_end);
  return ret;
}

int CropAndResizeCPUKernel::Run() {
  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto input_shape = input->shape();
  auto boxes = reinterpret_cast<float *>(in_tensors_.at(1)->data_c());
  auto box_idx = reinterpret_cast<int32_t *>(in_tensors_.at(2)->data_c());
  ret = PrepareCropAndResizeBilinear(input_shape.data(), boxes, box_idx, out_tensors_.at(0)->shape().data(), y_bottoms_,
                                     y_tops_, x_lefts_, x_rights_, y_bottom_weights_, x_left_weights_);
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, CropAndResizeImpl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "CropAndResize run error, error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_CropAndResize, LiteKernelCreator<CropAndResizeCPUKernel>)
}  // namespace mindspore::kernel
