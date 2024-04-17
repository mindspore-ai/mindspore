/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/scale_dynamic_fp16_coder.h"
#include <string>
#include <algorithm>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::lite::micro::nnacl {
int ScaleDynamicFP16Coder::Prepare(CoderContext *const context) {
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Input tensor data type should be fp16, now is " << input_tensors_[i]->data_type());
  }
  MS_CHECK_TRUE_MSG(output_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Output tensor data type should be fp16, now is " << output_tensor_->data_type());

  scale_param_ = reinterpret_cast<ScaleParameter *>(parameter_);
  MS_CHECK_PTR(scale_param_);
  MS_CHECK_TRUE_MSG(memset_s(&scale_struct_, sizeof(scale_struct_), 0, sizeof(scale_struct_)) == EOK, RET_ERROR,
                    "memset_s fail.");
  scale_struct_.base_.param_ = parameter_;
  if (input_tensors_.size() < DIMENSION_2D || input_tensors_.size() > DIMENSION_3D) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << input_tensors_.size() << " is given.";
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(InitScaleOffset(), "Scale fp16 InitScaleOffset failed.");
  MS_CHECK_RET_CODE(CalculateParameter(), "Scale fp16 CalculateParameter failed.");
  return RET_OK;
}

int ScaleDynamicFP16Coder::DoCode(CoderContext *const context) {
  // init struct ScaleParameters
  Collect(context,
          {
            "nnacl/kernel/scale.h",
            "nnacl/fp16/scale_fp16.h",
          },
          {
            "scale_fp16.c",
          });

  NNaclFp32Serializer code;
  scale_struct_.scale_ = nullptr;
  scale_struct_.offset_ = nullptr;
  scale_struct_.input_ = nullptr;
  scale_struct_.output_ = nullptr;
  code.CodeStruct("scale_struct", scale_struct_, dynamic_param_);

  auto scale = GetTensorAddr(scale_tensor_, const_scale_, dynamic_mem_manager_, allocator_);
  std::string offset{"NULL"};
  if (input_tensors_.size() == DIMENSION_3D) {
    auto offset_tensor = input_tensors_.at(kBiasIndex);
    offset = GetTensorAddr(offset_tensor, const_offset_, dynamic_mem_manager_, allocator_);
  }
  std::string input_str =
    "(float16_t *)(" + GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  std::string output_str =
    "(float16_t *)(" + GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU6:
      code.CodeFunction("DoScaleRelu6Fp16", input_str, output_str, scale, offset, kDefaultTaskId, "&scale_struct");
      break;
    case schema::ActivationType_RELU:
      code.CodeFunction("Fp16DoScaleRelu", input_str, output_str, scale, offset, kDefaultTaskId, "&scale_struct");
      break;
    case schema::ActivationType_NO_ACTIVATION:
      code.CodeFunction("DoScaleFp16", input_str, output_str, scale, offset, kDefaultTaskId, "&scale_struct");
      break;
    default:
      MS_LOG(ERROR) << "Scale does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int ScaleDynamicFP16Coder::InitScaleOffset() {
  scale_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(scale_tensor_);
  if (scale_tensor_->data() != nullptr) {
    const_scale_ = true;
  }
  if (input_tensors_.size() == DIMENSION_3D && input_tensors_.at(kBiasIndex)->data() != nullptr) {
    const_offset_ = true;
  }
  return RET_OK;
}

int ScaleDynamicFP16Coder::CalculateParameter() {
  auto in_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::vector<std::string> scale_shape;
  if (scale_tensor_->IsConst()) {
    auto tensor_shape = scale_tensor_->shape();
    (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(scale_shape),
                         [](const auto &dim) { return std::to_string(dim); });
  } else {
    scale_shape = shape_info_container_->GetTemplateShape(scale_tensor_);
  }
  scale_struct_.axis_ =
    scale_param_->axis_ < 0 ? scale_param_->axis_ + static_cast<int>(in_shape.size()) : scale_param_->axis_;
  if (scale_shape.size() + scale_struct_.axis_ > in_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }
  dynamic_param_.outer_size_ = AccumulateShape(in_shape, 0, scale_struct_.axis_);
  if (scale_tensor_->IsConst() && scale_tensor_->shape().size() == 1) {
    dynamic_param_.axis_size_ = in_shape.at(scale_struct_.axis_);
  } else {
    dynamic_param_.axis_size_ = "{";
    for (size_t i = 0; i < scale_shape.size(); i++) {
      if (in_shape.at(i + scale_struct_.axis_) != scale_shape.at(i)) {
        MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
        return RET_ERROR;
      }
      dynamic_param_.axis_size_ += in_shape.at(i + scale_struct_.axis_) + ", ";
    }
    dynamic_param_.axis_size_ += "}";
  }
  dynamic_param_.inner_size_ = AccumulateShape(in_shape, scale_struct_.axis_ + scale_shape.size(), in_shape.size());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_ScaleFusion,
                           CPUOpCoderCreator<ScaleDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_ScaleFusion,
                           CPUOpCoderCreator<ScaleDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
