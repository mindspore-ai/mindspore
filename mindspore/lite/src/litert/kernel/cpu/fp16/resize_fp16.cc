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
#include "src/litert/kernel/cpu/fp16/resize_fp16.h"
#include <map>
#include <utility>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::CoordinateTransformMode_ALIGN_CORNERS;
using mindspore::schema::CoordinateTransformMode_ASYMMETRIC;
using mindspore::schema::CoordinateTransformMode_HALF_PIXEL;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::kernel {
int ResizeFp16CPUKernel::ResizePrepare() {
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  const auto &input_shape = in_tensors_.front()->shape();
  const auto &output_shape = out_tensors_.front()->shape();
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    return PrepareResizeBilinearFp16(input_shape.data(), output_shape.data(), calculate_, coordinate_.y_bottoms_,
                                     coordinate_.y_tops_, coordinate_.x_lefts_, coordinate_.x_rights_,
                                     static_cast<float16_t *>(y_weights_), static_cast<float16_t *>(x_weights_));
  }
  if (method_ == static_cast<int>(schema::ResizeMethod_CUBIC)) {
    auto cubic_coeff = reinterpret_cast<ResizeParameter *>(op_parameter_)->cubic_coeff_;
    return PrepareResizeBicubicFp16(input_shape.data(), output_shape.data(), calculate_, coordinate_.y_tops_,
                                    coordinate_.x_lefts_, static_cast<float16_t *>(y_weights_),
                                    static_cast<float16_t *>(x_weights_), cubic_coeff);
  }
  return RET_OK;
}

int ResizeFp16CPUKernel::DataTypeLen() { return sizeof(float16_t); }

int ResizeFp16CPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.front();
  auto output = out_tensors_.front();
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  auto input_data = static_cast<float16_t *>(input->data());
  auto output_data = static_cast<float16_t *>(output->data());
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(output_data);
  int unit = UP_DIV(new_height_, op_parameter_->thread_num_);
  int h_begin = unit * task_id;
  int h_end = std::min(h_begin + unit, new_height_);
  int channel = input->Channel();
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      float16_t *line0 = static_cast<float16_t *>(line_buffer_) + new_width_ * channel * 2 * task_id;
      float16_t *line1 = line0 + new_width_ * channel;
      return ResizeBilinearFp16(input_data, output_data, input->shape().data(), output->shape().data(),
                                coordinate_.y_bottoms_, coordinate_.y_tops_, coordinate_.x_lefts_,
                                coordinate_.x_rights_, static_cast<float16_t *>(y_weights_),
                                static_cast<float16_t *>(x_weights_), line0, line1, h_begin, h_end);
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      return ResizeNearestNeighborFp16(input_data, output_data, input->shape().data(), output->shape().data(),
                                       calculate_, coordinate_transform_mode_, task_id, op_parameter_->thread_num_);
    }
    case static_cast<int>(schema::ResizeMethod_CUBIC): {
      float16_t *line_buffer = static_cast<float16_t *>(line_buffer_) + new_width_ * channel * 4 * task_id;
      return ResizeBicubicFp16(input_data, output_data, input->shape().data(), output->shape().data(),
                               coordinate_.y_tops_, coordinate_.x_lefts_, static_cast<float16_t *>(y_weights_),
                               static_cast<float16_t *>(x_weights_), line_buffer, h_begin, h_end);
    }
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Resize, LiteKernelCreator<ResizeFp16CPUKernel>)
}  // namespace mindspore::kernel
