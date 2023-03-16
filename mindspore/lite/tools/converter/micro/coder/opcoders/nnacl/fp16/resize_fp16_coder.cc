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
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::lite::micro::nnacl {
int ResizeFP16Coder::DataTypeLen() { return sizeof(uint16_t); }

int ResizeFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/resize_fp16.h",
          },
          {
            "resize_fp16.c",
          });
  Serializer code;
  code.CodeArray("input_shape", input_tensor_->shape().data(), input_tensor_->shape().size(), true);
  code.CodeArray("output_shape", output_tensor_->shape().data(), output_tensor_->shape().size(), true);
  std::vector<uint16_t> y_weights(y_weight_len_);
  Float32ToFp16(y_weights_, y_weights.data(), y_weight_len_);
  std::vector<uint16_t> x_weights(x_weight_len_);
  Float32ToFp16(x_weights_, x_weights.data(), x_weight_len_);

  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      code.CodeArray("y_bottoms", coordinate_.y_bottoms_, y_len_, true);
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("x_rights", coordinate_.x_rights_, x_len_, true);
      code.CodeArray("y_weights", y_weights.data(), y_weight_len_, true);
      code.CodeArray("x_weights", x_weights.data(), x_weight_len_, true);

      int c = input_tensor_->shape().at(kNHWC_C);
      code << "float16_t *line0 = (float16_t *)" << MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_)
           << ";\n";
      code << "float16_t *line1 = line0 + " << new_width_ << " * " << c << ";\n";
      code.CodeFunction("ResizeBilinearFp16", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_bottoms",
                        "y_tops", "x_lefts", "x_rights", "(float16_t *)y_weights", "(float16_t *)x_weights", "line0",
                        "line1", 0, new_height_);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      code.CodeFunction("ResizeNearestNeighborFp16", input_tensor_, output_tensor_, "input_shape", "output_shape",
                        calculate_str_, coordinate_transform_mode_, kDefaultTaskId, kDefaultThreadNum);
      break;
    }
    case static_cast<int>(schema::ResizeMethod_CUBIC): {
      code.CodeArray("y_tops", coordinate_.y_tops_, y_len_, true);
      code.CodeArray("x_lefts", coordinate_.x_lefts_, x_len_, true);
      code.CodeArray("y_weights", y_weights.data(), y_weight_len_, true);
      code.CodeArray("x_weights", x_weights.data(), x_weight_len_, true);
      auto buffer_str = "(float16_t *)" + MemoryAllocator::GetInstance()->GetRuntimeAddr(line_buffer_);
      code.CodeFunction("ResizeBicubicFp16", input_tensor_, output_tensor_, "input_shape", "output_shape", "y_tops",
                        "x_lefts", "(float16_t *)y_weights", "(float16_t *)x_weights", buffer_str, 0, new_height_);
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

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Resize, CPUOpCoderCreator<ResizeFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
