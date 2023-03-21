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
#include "coder/opcoders/nnacl/fp32/scale_fp32_coder.h"
#include <string>
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::lite::micro::nnacl {
int ScaleFP32Coder::InitScaleOffset() {
  Tensor *scale_tensor = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(scale_tensor);
  if (scale_tensor->data() != nullptr) {
    scale_param_->const_scale_ = true;
  } else {
    scale_param_->const_scale_ = false;
  }

  if (input_tensors_.size() == DIMENSION_3D && input_tensors_.at(kBiasIndex)->data() != nullptr) {
    scale_param_->const_offset_ = true;
  } else {
    scale_param_->const_offset_ = false;
  }
  return RET_OK;
}

int ScaleFP32Coder::CalculateParameter() {
  std::vector<int> in_shape = input_tensor_->shape();
  Tensor *scale_tensor = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(scale_tensor);
  std::vector<int> scale_shape = scale_tensor->shape();

  if (scale_param_->axis_ < 0) {
    scale_param_->axis_ = scale_param_->axis_ + in_shape.size();
  }
  if (scale_shape.size() + scale_param_->axis_ > in_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }
  scale_param_->outer_size_ = 1;
  scale_param_->axis_size_ = 1;
  scale_param_->inner_size_ = 1;
  for (int i = 0; i < scale_param_->axis_; i++) {
    scale_param_->outer_size_ *= in_shape.at(i);
  }
  for (size_t i = 0; i < scale_shape.size(); i++) {
    if (in_shape.at(i + scale_param_->axis_) != scale_shape.at(i)) {
      MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
      return RET_ERROR;
    }
    scale_param_->axis_size_ *= in_shape.at(i + scale_param_->axis_);
  }
  for (size_t i = scale_param_->axis_ + scale_shape.size(); i < in_shape.size(); i++) {
    scale_param_->inner_size_ *= in_shape.at(i);
  }
  scale_param_->op_parameter_.thread_num_ = MSMIN(scale_param_->op_parameter_.thread_num_, scale_param_->outer_size_);
  return RET_OK;
}

int ScaleFP32Coder::Prepare(CoderContext *const context) {
  this->scale_param_ = reinterpret_cast<ScaleParameter *>(parameter_);
  if (input_tensors_.size() < DIMENSION_2D || input_tensors_.size() > DIMENSION_3D) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << input_tensors_.size() << " is given.";
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(InitScaleOffset(), "Scale fp32 InitScaleOffset failed.");
  return ReSize();
}

int ScaleFP32Coder::ReSize() {
  MS_CHECK_RET_CODE(CalculateParameter(), "Scale fp32 CalculateParameter failed.");
  return RET_OK;
}

int ScaleFP32Coder::DoCode(CoderContext *const context) {
  // init struct ScaleParameters
  Collect(context,
          {
            "wrapper/fp32/scale_fp32_wrapper.h",
            "nnacl/scale.h",
            "nnacl/fp32/scale_fp32.h",
          },
          {
            "scale_fp32_wrapper.c",
            "scale_fp32.c",
          });

  NNaclFp32Serializer code;
  code.CodeStruct("scale_parameter", *scale_param_);
  auto scale = allocator_->GetRuntimeAddr(input_tensors_.at(kWeightIndex), scale_param_->const_scale_);
  std::string offset{"NULL"};
  if (input_tensors_.size() == DIMENSION_3D) {
    offset = allocator_->GetRuntimeAddr(input_tensors_.at(kBiasIndex), scale_param_->const_offset_);
  }
  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU6:
      code.CodeFunction("DoScaleRelu6", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId,
                        "&scale_parameter");
      break;
    case schema::ActivationType_RELU: {
      if (!support_parallel_) {
        code.CodeFunction("DoScaleRelu", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId,
                          "&scale_parameter");
      } else {
        code.CodeBaseStruct("ScaleFp32Args", kRunArgs, input_tensor_, output_tensor_, scale, offset,
                            "&scale_parameter");
        code.CodeFunction(kParallelLaunch, "DoScaleReluRun", kRunArgsAddr, "scale_parameter.op_parameter_.thread_num_");
      }
      break;
    }
    case schema::ActivationType_NO_ACTIVATION:
      code.CodeFunction("DoScale", input_tensor_, output_tensor_, scale, offset, kDefaultTaskId, "&scale_parameter");
      break;
    default:
      MS_LOG(ERROR) << "Scale does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ScaleFusion, CPUOpCoderCreator<ScaleFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
