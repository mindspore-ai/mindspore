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

#include "coder/opcoders/nnacl/fp16/resize_fp16_coder.h"
#include <string>
#include <map>
#include <utility>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"
#include "nnacl/fp32/resize_fp32.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::lite::micro::nnacl {
int ResizeFP16Coder::DataTypeLen() { return sizeof(uint16_t); }

int ResizeFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/resize_fp16.h",
            "nnacl/fp32/resize_fp32.h",
          },
          {
            "resize_fp16.c",
            "resize_fp32.c",
          });
  nnacl::NNaclFp32Serializer code;
  code.CodeArray("input_shape", input_tensor_->shape().data(), input_tensor_->shape().size(), true);
  code.CodeArray("output_shape", output_tensor_->shape().data(), output_tensor_->shape().size(), true);
  float16 *x_weights_fp16 = reinterpret_cast<float16 *>(malloc(DataTypeLen() * x_weight_len_));
  MS_CHECK_PTR_WITH_EXE(x_weights_fp16, free(x_weights_fp16));
  for (size_t i = 0; i < x_weight_len_; i++) {
    x_weights_fp16[i] = float16(x_weights_[i]);
  }
  float16 *y_weights_fp16 = reinterpret_cast<float16 *>(malloc(DataTypeLen() * y_weight_len_));
  MS_CHECK_PTR_WITH_EXE(y_weights_fp16, free(y_weights_fp16));
  for (size_t i = 0; i < y_weight_len_; i++) {
    y_weights_fp16[i] = float16(y_weights_[i]);
  }
  int unit = UP_DIV(new_height_, kDefaultThreadNum);
  int h_begin = unit * kDefaultTaskId;
  int h_end = std::min(h_begin + unit, new_height_);
  int channel = input_tensor_->Channel();

  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      auto ret = memset_s(coordinate_.y_bottoms_, y_len_ * sizeof(int), 0, y_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(coordinate_.y_tops_, y_len_ * sizeof(int), 0, y_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(coordinate_.x_lefts_, x_len_ * sizeof(int), 0, x_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(coordinate_.x_rights_, x_len_ * sizeof(int), 0, x_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(y_weights_fp16, y_weight_len_ * DataTypeLen(), 0, y_weight_len_ * DataTypeLen());
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(x_weights_fp16, x_weight_len_ * DataTypeLen(), 0, x_weight_len_ * DataTypeLen());
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      code.CodeArray("y_bottoms", coordinate_.y_bottoms_, y_len_, true);
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("x_rights", coordinate_.x_rights_, x_len_, true);
      code.CodeArray("y_weights", y_weights_fp16, y_weight_len_, true);
      code.CodeArray("x_weights", x_weights_fp16, x_weight_len_, true);

      code.CodeFunction("PrepareResizeBilinearFp16", "input_shape", "output_shape", calculate_str_, "(int *)y_bottoms",
                        "(int *)y_tops", "(int *)x_lefts", "(int *)x_rights", "(float16_t *)y_weights",
                        "(float16_t *)x_weights");
      code << "    float16_t *line0 = (float16_t *)" << MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_)
           << " + " << new_width_ << " * 2 * " << kDefaultTaskId << ";\n";
      code << "    float16_t *line1 = line0 + " << new_width_ << " * " << channel << ";\n";
      code.CodeFunction("ResizeBilinearFp16", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_bottoms",
                        "y_tops", "x_lefts", "x_rights", "(float16_t *)y_weights", "(float16_t *)x_weights", "line0",
                        "line1", h_begin, h_end);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      code.CodeFunction("ResizeNearestNeighborFp16", input_tensor_, output_tensor_, "input_shape", "output_shape",
                        calculate_str_, coordinate_transform_mode_, kDefaultTaskId, kDefaultThreadNum);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_CUBIC): {
      auto ret = memset_s(coordinate_.y_tops_, y_len_ * sizeof(int), 0, y_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(coordinate_.x_lefts_, x_len_ * sizeof(int), 0, x_len_ * sizeof(int));
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(y_weights_fp16, y_weight_len_ * DataTypeLen(), 0, y_weight_len_ * DataTypeLen());
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      ret = memset_s(x_weights_fp16, x_weight_len_ * DataTypeLen(), 0, x_weight_len_ * DataTypeLen());
      MS_CHECK_RET_CODE(ret, "memset_s failed");
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("y_weights", y_weights_fp16, y_weight_len_, true);
      code.CodeArray("x_weights", x_weights_fp16, x_weight_len_, true);
      auto resize_parameter = reinterpret_cast<ResizeParameter *>(parameter_);
      MS_CHECK_PTR(resize_parameter);
      auto cubic_coeff_str = "(float16_t)" + std::to_string(resize_parameter->cubic_coeff_);
      code.CodeFunction("PrepareResizeBicubicFp16", "input_shape", "output_shape", calculate_str_, "(int *)y_tops",
                        "(int *)x_lefts", "(float16_t *)y_weights", "(float16_t *)x_weights", cubic_coeff_str);
      auto buffer_str = "(float16_t *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_) + " + " +
                        std::to_string(new_width_ * channel * 4 * kDefaultTaskId);

      code.CodeFunction("ResizeBicubicFp16", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_tops",
                        "x_lefts", "(float16_t *)y_weights", "(float16_t *)x_weights", buffer_str, h_begin, h_end);
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

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Resize, CPUOpCoderCreator<ResizeFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Resize, CPUOpCoderCreator<ResizeFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
