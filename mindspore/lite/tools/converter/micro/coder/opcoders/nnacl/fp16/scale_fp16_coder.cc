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
#include "coder/opcoders/nnacl/fp16/scale_fp16_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::lite::micro::nnacl {
int ScaleFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }

  this->scale_param_ = reinterpret_cast<ScaleParameter *>(parameter_);
  if (input_tensors_.size() < DIMENSION_2D || input_tensors_.size() > DIMENSION_3D) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << input_tensors_.size() << " is given.";
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(ScaleFP32Coder::InitScaleOffset(), "Scale fp16 InitScaleOffset failed.");
  MS_CHECK_RET_CODE(ScaleFP32Coder::CalculateParameter(), "Scale fp16 CalculateParameter failed.");
  return RET_OK;
}

int ScaleFP16Coder::DoCode(CoderContext *const context) {
  // init struct ScaleParameters
  Collect(context,
          {
            "nnacl/scale_parameter.h",
            "nnacl/kernel/scale.h",
            "nnacl/fp16/scale_fp16.h",
          },
          {
            "scale_fp32_wrapper.c",
            "scale_fp16.c",
          });

  NNaclFp32Serializer code;
  code.CodeStruct("scale_struct", scale_struct_);
  code << "    scale_struct.base_.thread_nr_ = " << thread_num_ << "; \n";

  auto scale = allocator_->GetRuntimeAddr(input_tensors_.at(kWeightIndex), const_scale_);
  std::string offset{"NULL"};
  if (input_tensors_.size() == DIMENSION_3D) {
    offset = allocator_->GetRuntimeAddr(input_tensors_.at(kBiasIndex), const_offset_);
  }
  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU6:
      code.CodeFunction("DoScaleRelu6Fp16", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId,
                        "&scale_struct");
      break;
    case schema::ActivationType_RELU: {
      code.CodeFunction("Fp16DoScaleRelu", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId,
                        "&scale_struct");
    }
    case schema::ActivationType_NO_ACTIVATION:
      code.CodeFunction("DoScaleFp16", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId, "&scale_struct");
      break;
    default:
      MS_LOG(ERROR) << "Scale does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_ScaleFusion, CPUOpCoderCreator<ScaleFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_ScaleFusion, CPUOpCoderCreator<ScaleFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
