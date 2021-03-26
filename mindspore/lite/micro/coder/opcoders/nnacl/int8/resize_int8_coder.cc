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

#include "coder/opcoders/nnacl/int8/resize_int8_coder.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "securec/include/securec.h"
#include "nnacl/int8/quantize.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::lite::micro::nnacl {
ResizeInt8Coder::~ResizeInt8Coder() {
  delete quant_out_;
  quant_out_ = nullptr;
  delete quant_in_;
  quant_in_ = nullptr;
  delete multiplier_;
  multiplier_ = nullptr;
}

int ResizeInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ResizeBaseCoder::Init(), "init resize base failed");
  quant_in_ = new (std::nothrow)::QuantArg;
  quant_out_ = new (std::nothrow)::QuantArg;
  multiplier_ = new (std::nothrow) QuantMulArg;
  MS_CHECK_PTR(quant_in_);
  MS_CHECK_PTR(quant_out_);
  MS_CHECK_PTR(multiplier_);
  quant_in_->zp_ = input_tensor_->quant_params().at(0).zeroPoint;
  quant_in_->scale_ = input_tensor_->quant_params().at(0).scale;
  quant_out_->zp_ = output_tensor_->quant_params().at(0).zeroPoint;
  quant_out_->scale_ = output_tensor_->quant_params().at(0).scale;

  QuantizeRoundParameterWithDoublePrecision(quant_in_->scale_ / quant_out_->scale_, &multiplier_->multiplier_,
                                            &multiplier_->left_shift_, &multiplier_->right_shift_);
  return ReSize();
}

int ResizeInt8Coder::ReSize() {
  if (method_ == schema::ResizeMethod_LINEAR) {
    MS_LOG(ERROR) << "unsupported resize linear currently";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8Coder::DoCode(CoderContext *const context) {
  std::vector<std::string> headers = {"nnacl/int8/resize_int8.h", "wrapper/int8/resize_int8_wrapper.h"};
  std::vector<std::string> cFiles = {"resize_int8.c", "common_func.c", "resize_int8_wrapper.c"};
  Collect(context, headers, cFiles);

  nnacl::NNaclInt8Serializer code;
  code.CodeArray("input_shape", input_tensor_->shape().data(), input_tensor_->shape().size(), true);
  code.CodeArray("output_shape", output_tensor_->shape().data(), output_tensor_->shape().size(), true);
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_LINEAR): {
      MS_LOG(ERROR) << "unsupported: " << schema::EnumNameResizeMethod(static_cast<schema::ResizeMethod>(method_));
      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST): {
      bool same_zp = quant_in_->zp_ == quant_out_->zp_;
      bool same_scale = abs(quant_out_->scale_ - quant_in_->scale_) < 1e-6;
      bool align_corners = coordinate_transform_mode_ == schema::CoordinateTransformMode_ALIGN_CORNERS;
      if (same_zp && same_scale) {
        code.CodeBaseStruct("ResizeInt8Args", kRunArgs, input_tensor_, output_tensor_, "input_shape", "output_shape",
                            align_corners, gThreadNum);
        if (support_parallel_) {
          code.CodeFunction(kParallelLaunch, gThreadPool, "ResizeInt8Run", kRunArgsAddr, gThreadNum);
        } else {
          code.CodeFunction("ResizeInt8Run", kRunArgsAddr, kDefaultTaskId);
        }
      } else {
        MS_LOG(WARNING) << "unsupported parallel launch currently";
        code.CodeStruct("quant_in", *quant_in_);
        code.CodeStruct("quant_out", *quant_out_);
        code.CodeStruct("multiplier", *multiplier_);
        code.CodeFunction("ResizeNearestNeighborInt8", input_tensor_, output_tensor_, "input_shape", "output_shape",
                          align_corners, "multiplier", "quant_in", "quant_out", kDefaultTaskId, gThreadNum);
      }
      break;
    }
    case schema::ResizeMethod_UNKNOWN:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      return RET_ERROR;
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Resize, CPUOpCoderCreator<ResizeInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
