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

#include "coder/opcoders/nnacl/fp32/resize_fp32_coder.h"
#include <string>
#include <map>
#include <utility>
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::CoordinateTransformMode_ALIGN_CORNERS;
using mindspore::schema::CoordinateTransformMode_ASYMMETRIC;
using mindspore::schema::CoordinateTransformMode_HALF_PIXEL;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::lite::micro::nnacl {
int ResizeFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ResizeBaseCoder::Init(), "ResizeBaseCoder::Init failed");
  MS_CHECK_RET_CODE(SelectCalculatorFunc(), "SelectCalculatorFunc failed");
  MS_CHECK_RET_CODE(ReSize(), "ReSize failed");
  return RET_OK;
}

int ResizeFP32Coder::SelectCalculatorFunc() {
  const std::map<int, std::pair<CalculateOriginalCoordinate, std::string>> cal_fuc_list = {
    std::make_pair(CoordinateTransformMode_ASYMMETRIC, std::make_pair(CalculateAsymmetric, "CalculateAsymmetric")),
    std::make_pair(CoordinateTransformMode_ALIGN_CORNERS,
                   std::make_pair(CalculateAlignCorners, "CalculateAlignCorners")),
    std::make_pair(CoordinateTransformMode_HALF_PIXEL, std::make_pair(CalculateHalfPixel, "CalculateHalfPixel")),
  };

  auto fun_pair = cal_fuc_list.find(coordinate_transform_mode_);
  if (fun_pair != cal_fuc_list.end()) {
    calculate_ = fun_pair->second.first;
    calculate_str_ = fun_pair->second.second;
  } else {
    MS_LOG(ERROR) << "Do not support coordinate transform mode. Mode is"
                  << schema::EnumNameCoordinateTransformMode(
                       static_cast<schema::CoordinateTransformMode>(coordinate_transform_mode_));
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeFP32Coder::ReSize() {
  if (method_ == static_cast<int>(schema::ResizeMethod_NEAREST)) {
    return RET_OK;
  }

  if (!const_shape_) {
    new_height_ = output_tensor_->shape().at(kNHWC_H);
    new_width_ = output_tensor_->shape().at(kNHWC_W);
  }

  MS_CHECK_RET_CODE_WITH_EXE(MallocTmpBuffer(), "MallocTmpBuffer failed", FreeTmpBuffer());
  MS_CHECK_RET_CODE_WITH_EXE(ResizePrepare(), "ResizePrepare failed", FreeTmpBuffer());

  return RET_OK;
}

// Bilinear interpolation considers the closest 2x2 neighborhood of known pixel values surrounding the unknown pixel.
// It takes a weighted average of these 4 pixels to arrive at its final interpolated value. Thus, we need to reserve
// twice bigger space than coordinates arrays for weight arrays. It means x_weight_len is twice as much as x_len in
// detail.
// Bicubic goes one step beyond bilinear by considering the closest 4x4 neighborhood of known pixels --- for a total of
// 16 pixels. Since these are at various distances from the unknown pixel, closer pixels are given a higher weighting in
// the calculation.
void ResizeFP32Coder::CalTmpBufferLen() {
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    x_len_ = new_width_;
    y_len_ = new_height_;
    x_weight_len_ = new_width_;
    y_weight_len_ = new_height_;
  }
  if (method_ == static_cast<int>(schema::ResizeMethod_CUBIC)) {
    x_len_ = new_width_ * kFour;
    y_len_ = new_height_ * kFour;
    x_weight_len_ = new_width_ * kFour;
    y_weight_len_ = new_height_ * kFour;
  }
}

int ResizeFP32Coder::MallocTmpBuffer() {
  if (method_ != static_cast<int>(schema::ResizeMethod_LINEAR) &&
      method_ != static_cast<int>(schema::ResizeMethod_CUBIC)) {
    return RET_OK;
  }
  // make sure y_bottoms_, y_tops_, etc. are null before malloc
  FreeTmpBuffer();

  CalTmpBufferLen();

  // malloc memory for x, y coordinates
  {
    coordinate_.x_lefts_ = reinterpret_cast<int *>(malloc(sizeof(int) * x_len_));
    CHECK_MALLOC_RES(coordinate_.x_lefts_, RET_NULL_PTR);
    coordinate_.y_tops_ = reinterpret_cast<int *>(malloc(sizeof(int) * y_len_));
    CHECK_MALLOC_RES(coordinate_.y_tops_, RET_NULL_PTR);
    if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
      coordinate_.x_rights_ = reinterpret_cast<int *>(malloc(sizeof(int) * x_len_));
      CHECK_MALLOC_RES(coordinate_.x_rights_, RET_NULL_PTR);
      coordinate_.y_bottoms_ = reinterpret_cast<int *>(malloc(sizeof(int) * y_len_));
      CHECK_MALLOC_RES(coordinate_.y_bottoms_, RET_NULL_PTR);
    }
  }

  // malloc memory for weights of x, y axes
  {
    x_weights_ = reinterpret_cast<float *>(malloc(sizeof(float) * x_weight_len_));
    CHECK_MALLOC_RES(x_weights_, RET_NULL_PTR);
    y_weights_ = reinterpret_cast<float *>(malloc(sizeof(float) * y_weight_len_));
    CHECK_MALLOC_RES(y_weights_, RET_NULL_PTR);
  }

  {
    size_t line_buffer_size = DataTypeLen() * x_len_ * input_tensor_->Channel() * kTwo * kMaxThreadNumSupported;
    line_buffer_ = allocator_->Malloc(kNumberTypeUInt8, line_buffer_size, kWorkspace);
    CHECK_MALLOC_RES(line_buffer_, RET_NULL_PTR);
  }
  return RET_OK;
}

void ResizeFP32Coder::FreeTmpBuffer() { coordinate_.FreeData(); }

int ResizeFP32Coder::ResizePrepare() {
  auto input_shape = input_tensor_->shape();
  if (method_ == static_cast<int>(schema::ResizeMethod_LINEAR)) {
    return PrepareResizeBilinear(input_shape.data(), output_tensor_->shape().data(), calculate_, coordinate_.y_bottoms_,
                                 coordinate_.y_tops_, coordinate_.x_lefts_, coordinate_.x_rights_, y_weights_,
                                 x_weights_);
  }
  if (method_ == static_cast<int>(schema::ResizeMethod_CUBIC)) {
    auto resize_parameter = reinterpret_cast<ResizeParameter *>(parameter_);
    MS_CHECK_PTR(resize_parameter);
    auto cubic_coeff = resize_parameter->cubic_coeff_;
    return PrepareResizeBicubic(input_shape.data(), output_tensor_->shape().data(), calculate_, coordinate_.y_tops_,
                                coordinate_.x_lefts_, y_weights_, x_weights_, cubic_coeff);
  }
  return RET_OK;
}

int ResizeFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp32/resize_fp32.h",
          },
          {
            "resize_fp32.c",
          });
  Serializer code;
  code.CodeArray("input_shape", input_tensor_->shape().data(), input_tensor_->shape().size(), true);
  code.CodeArray("output_shape", output_tensor_->shape().data(), output_tensor_->shape().size(), true);

  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      code.CodeArray("y_bottoms", coordinate_.y_bottoms_, y_len_, true);
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("x_rights", coordinate_.x_rights_, x_len_, true);
      code.CodeArray("y_weights", y_weights_, y_weight_len_, true);
      code.CodeArray("x_weights", x_weights_, x_weight_len_, true);

      int c = input_tensor_->shape().at(kNHWC_C);
      code << "float *line0 = " << MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_) << ";\n";
      code << "float *line1 = line0 + " << new_width_ << " * " << c << ";\n";
      code.CodeFunction("ResizeBilinear", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_bottoms",
                        "y_tops", "x_lefts", "x_rights", "y_weights", "x_weights", "line0", "line1", 0, new_height_);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      code.CodeFunction("ResizeNearestNeighbor", input_tensor_, output_tensor_, "input_shape", "output_shape",
                        calculate_str_, coordinate_transform_mode_, kDefaultTaskId, kDefaultThreadNum);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_CUBIC): {
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("y_weights", y_weights_, y_weight_len_, true);
      code.CodeArray("x_weights", x_weights_, x_weight_len_, true);
      auto buffer_str = "(float *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_);
      code.CodeFunction("ResizeBicubic", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_tops",
                        "x_lefts", "y_weights", "x_weights", buffer_str, 0, new_height_);
      break;
    }
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Resize, CPUOpCoderCreator<ResizeFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
