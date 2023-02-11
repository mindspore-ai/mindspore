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

#include <map>
#include <utility>
#include "src/litert/kernel/cpu/fp32/resize_fp32.h"
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
namespace {
constexpr int kResizeSizeDouble = 2;
}  // namespace

int ResizeCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = ResizeBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  ret = SelectCalculatorFunc();
  if (ret != RET_OK) {
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeCPUKernel::ReSize() {
  if (method_ == static_cast<int>(schema::ResizeMethod_NEAREST)) {
    return RET_OK;
  }

  if (!const_shape_) {
    new_height_ = out_tensors_.at(0)->shape()[kNHWC_H];
    new_width_ = out_tensors_.at(0)->shape()[kNHWC_W];
  }

  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  ret = ResizePrepare();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }
  return RET_OK;
}

// Bilinear interpolation :
// Bilinear interpolation considers the closest 2x2 neighborhood of known pixel values surrounding the unknown pixel.
// It takes a weighted average of these 4 pixels to arrive at its final interpolated value. Thus, we need to reserve
// twice bigger space than coordinates arrays for weight arrays. It means x_weight_len is twice as much as x_len in
// detail.
//
// Bicubic interpolation:
// Bicubic goes one step beyond bilinear by considering the closest 4x4 neighborhood of known pixels --- for a total of
// 16 pixels. Since these are at various distances from the unknown pixel, closer pixels are given a higher weighting in
// the calculation.
void ResizeCPUKernel::CalTmpBufferLen(int *x_len, int *y_len, int *x_weight_len, int *y_weight_len) const {
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    *x_len = new_width_;
    *y_len = new_height_;
    *x_weight_len = new_width_;
    *y_weight_len = new_height_;
  }
  if (method_ == static_cast<int>(schema::ResizeMethod_CUBIC)) {
    *x_len = new_width_ * C4NUM;
    *y_len = new_height_ * C4NUM;
    *x_weight_len = new_width_ * C4NUM;
    *y_weight_len = new_height_ * C4NUM;
  }
}

// If resize method is bicubic, x_lefts_ array stores four elements (index - 1, index, index + 1, index + 2) for every
// output coordinate index.
int ResizeCPUKernel::MallocTmpBuffer() {
  if (method_ != static_cast<int>(schema::ResizeMethod_LINEAR) &&
      method_ != static_cast<int>(schema::ResizeMethod_CUBIC)) {
    return RET_OK;
  }
  // make sure y_bottoms_, y_tops_, etc. are null before malloc
  FreeTmpBuffer();

  int x_len = 0, y_len = 0, x_weight_len = 0, y_weight_len = 0;
  CalTmpBufferLen(&x_len, &y_len, &x_weight_len, &y_weight_len);

  // malloc memory for x, y coordinates
  {
    MS_CHECK_LE((static_cast<int64_t>(sizeof(int)) * x_len), MAX_MALLOC_SIZE, RET_ERROR);
    coordinate_.x_lefts_ = reinterpret_cast<int *>(malloc(static_cast<int>(sizeof(int)) * x_len));
    CHECK_MALLOC_RES(coordinate_.x_lefts_, RET_NULL_PTR);
    MS_CHECK_LE((static_cast<int64_t>(sizeof(int)) * y_len), MAX_MALLOC_SIZE, RET_ERROR);
    coordinate_.y_tops_ = reinterpret_cast<int *>(malloc(static_cast<int>(sizeof(int)) * y_len));
    CHECK_MALLOC_RES(coordinate_.y_tops_, RET_NULL_PTR);
    if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
      coordinate_.x_rights_ = reinterpret_cast<int *>(malloc(static_cast<int>(sizeof(int)) * x_len));
      CHECK_MALLOC_RES(coordinate_.x_rights_, RET_NULL_PTR);
      coordinate_.y_bottoms_ = reinterpret_cast<int *>(malloc(static_cast<int>(sizeof(int)) * y_len));
      CHECK_MALLOC_RES(coordinate_.y_bottoms_, RET_NULL_PTR);
    }
  }

  // malloc memory for weights of x, y axes
  {
    MS_CHECK_LE((static_cast<int64_t>(DataTypeLen()) * x_weight_len), MAX_MALLOC_SIZE, RET_ERROR);
    x_weights_ = malloc(x_weight_len * DataTypeLen());
    CHECK_MALLOC_RES(x_weights_, RET_NULL_PTR);
    MS_CHECK_LE((static_cast<int64_t>(DataTypeLen()) * y_weight_len), MAX_MALLOC_SIZE, RET_ERROR);
    y_weights_ = malloc(y_weight_len * DataTypeLen());
    CHECK_MALLOC_RES(y_weights_, RET_NULL_PTR);
  }

  {
    MS_CHECK_LE((static_cast<int64_t>(DataTypeLen()) * x_len * in_tensors_.at(0)->Channel() * kResizeSizeDouble *
                 op_parameter_->thread_num_),
                MAX_MALLOC_SIZE, RET_ERROR);
    MS_CHECK_TRUE_RET(in_tensors_.at(0)->Channel() > 0, RET_ERROR);
    line_buffer_ =
      malloc(DataTypeLen() * x_len * in_tensors_.at(0)->Channel() * kResizeSizeDouble * op_parameter_->thread_num_);
    CHECK_MALLOC_RES(line_buffer_, RET_NULL_PTR);
  }
  return RET_OK;
}

void ResizeCPUKernel::FreeTmpBuffer() {
  coordinate_.FreeData();
  if (y_weights_ != nullptr) {
    free(y_weights_);
    y_weights_ = nullptr;
  }
  if (x_weights_ != nullptr) {
    free(x_weights_);
    x_weights_ = nullptr;
  }
  if (line_buffer_ != nullptr) {
    free(line_buffer_);
    line_buffer_ = nullptr;
  }
}

int ResizeImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
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
  auto input_data = reinterpret_cast<float *>(input->data());
  auto output_data = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  MSLITE_CHECK_PTR(ms_context_);
  MSLITE_CHECK_PTR(input_data);
  MSLITE_CHECK_PTR(output_data);

  auto input_shape = input->shape();
  int unit = UP_DIV(new_height_, op_parameter_->thread_num_);
  int h_begin = unit * task_id;
  int h_end = std::min(h_begin + unit, new_height_);
  int c = input_shape.at(kNHWC_C);
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      float *line0 = static_cast<float *>(line_buffer_) + new_width_ * c * C2NUM * task_id;
      float *line1 = line0 + new_width_ * c;
      return ResizeBilinear(input_data, output_data, input_shape.data(), out_tensors_.at(0)->shape().data(),
                            coordinate_.y_bottoms_, coordinate_.y_tops_, coordinate_.x_lefts_, coordinate_.x_rights_,
                            static_cast<float *>(y_weights_), static_cast<float *>(x_weights_), line0, line1, h_begin,
                            h_end);
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      return ResizeNearestNeighbor(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(),
                                   calculate_, coordinate_transform_mode_, task_id, op_parameter_->thread_num_);
    }
    case static_cast<int>(schema::ResizeMethod_CUBIC): {
      float *line_buffer = static_cast<float *>(line_buffer_) +
                           static_cast<size_t>(new_width_ * c) * sizeof(float) * static_cast<size_t>(task_id);
      return ResizeBicubic(input_data, output_data, input_shape.data(), out_tensors_.at(0)->shape().data(),
                           coordinate_.y_tops_, coordinate_.x_lefts_, static_cast<float *>(y_weights_),
                           static_cast<float *>(x_weights_), line_buffer, h_begin, h_end);
    }
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }
}

int ResizeCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ResizeImpl, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeCPUKernel::ResizePrepare() {
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  const auto &input_shape = in_tensors_.front()->shape();
  const auto &output_shape = out_tensors_.front()->shape();
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    return PrepareResizeBilinear(input_shape.data(), output_shape.data(), calculate_, coordinate_.y_bottoms_,
                                 coordinate_.y_tops_, coordinate_.x_lefts_, coordinate_.x_rights_,
                                 static_cast<float *>(y_weights_), static_cast<float *>(x_weights_));
  }
  if (method_ == static_cast<int>(schema::ResizeMethod_CUBIC)) {
    auto cubic_coeff = reinterpret_cast<ResizeParameter *>(op_parameter_)->cubic_coeff_;
    return PrepareResizeBicubic(input_shape.data(), output_shape.data(), calculate_, coordinate_.y_tops_,
                                coordinate_.x_lefts_, static_cast<float *>(y_weights_),
                                static_cast<float *>(x_weights_), cubic_coeff);
  }
  return RET_OK;
}

int ResizeCPUKernel::SelectCalculatorFunc() {
  std::map<int, CalculateOriginalCoordinate> cal_fuc_list = {
    std::make_pair(CoordinateTransformMode_ASYMMETRIC, CalculateAsymmetric),
    std::make_pair(CoordinateTransformMode_ALIGN_CORNERS, CalculateAlignCorners),
    std::make_pair(CoordinateTransformMode_HALF_PIXEL, CalculateHalfPixel)};

  auto iter = cal_fuc_list.find(coordinate_transform_mode_);
  if (iter != cal_fuc_list.end()) {
    calculate_ = iter->second;
  } else {
    MS_LOG(ERROR) << "Do not support coordinate transform mode. Mode is"
                  << schema::EnumNameCoordinateTransformMode(
                       static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode_));
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Resize, LiteKernelCreator<ResizeCPUKernel>)
}  // namespace mindspore::kernel
