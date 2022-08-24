/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/cpu/base/scale_base.h"

using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::lite::micro::nnacl {
int ScaleFP32Coder::InitScaleOffset() {
  if (input_tensors_.size() == kInputSize2) {
    return RET_OK;
  }
  offset_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat32, scale_param_->axis_size_ * sizeof(float), kOfflinePackWeight));
  if (offset_ == nullptr) {
    MS_LOG(ERROR) << "Scale: malloc buffer for offset failed.";
    return RET_NULL_PTR;
  }
  MS_CHECK_RET_CODE(
    memcpy_s(offset_, scale_param_->axis_size_ * sizeof(float), 0, scale_param_->axis_size_ * sizeof(float)),
    "Scale: do memcpy_s for offset failed!");
  return RET_OK;
}

int ScaleFP32Coder::ComputeThreadCuttingInfo() {
  split_points_ = {0};
  int ele_num = output_tensor_->ElementsNum();
  if (!support_parallel_ || scale_param_->op_parameter_.thread_num_ <= 1) {
    scale_param_->op_parameter_.thread_num_ = 1;
    split_points_.push_back(ele_num);
    return RET_OK;
  }
  int block = ele_num / scale_param_->op_parameter_.thread_num_;
  int remain = ele_num - block * scale_param_->op_parameter_.thread_num_;
  int split = 0;
  while (split < ele_num) {
    split += block;
    split = remain > 0 ? (--remain, split + 1) : split;
    if (split > ele_num) {
      split = ele_num;
    }
    split_points_.push_back(split);
  }
  split_points_bak_ = reinterpret_cast<int *>(
    allocator_->Malloc(kNumberTypeInt32, split_points_.size() * sizeof(int), kOfflinePackWeight));
  if (split_points_bak_ == nullptr) {
    MS_LOG(ERROR) << "Scale: malloc buffer for split-points info failed.";
    return RET_NULL_PTR;
  }
  MS_CHECK_RET_CODE(memcpy_s(split_points_bak_, split_points_.size() * sizeof(int), split_points_.data(),
                             split_points_.size() * sizeof(int)),
                    "Scale: do memcpy_s for split-points failed!");
  scale_param_->op_parameter_.thread_num_ = split_points_.size() - 1;
  return RET_OK;
}

int ScaleFP32Coder::Prepare(CoderContext *const context) {
  this->scale_param_ = reinterpret_cast<ScaleParameter *>(parameter_);
  if (input_tensors_.size() < DIMENSION_2D || input_tensors_.size() > DIMENSION_3D) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << input_tensors_.size() << " is given.";
    return RET_ERROR;
  }
  if (scale_param_->activation_type_ != schema::ActivationType_NO_ACTIVATION &&
      scale_param_->activation_type_ != schema::ActivationType_RELU &&
      scale_param_->activation_type_ != schema::ActivationType_RELU6) {
    MS_LOG(ERROR) << "Scale: activation_type only support relu and relu6, but now is "
                  << scale_param_->activation_type_;
    return RET_ERROR;
  }
  return ReSize();
}

int ScaleFP32Coder::ReSize() {
  MS_CHECK_RET_CODE(mindspore::kernel::ScaleBaseCPUKernel::CalculateParameter(input_tensors_, scale_param_),
                    "Scale fp32 CalculateParameter failed.");
  MS_CHECK_RET_CODE(InitScaleOffset(), "Scale fp32 InitScaleOffset failed.");
  MS_CHECK_RET_CODE(ComputeThreadCuttingInfo(), "Scale fp32 ComputeThreadCuttingInfo failed.");
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

  if (!support_parallel_ || scale_param_->op_parameter_.thread_num_ == 1) {
    code << "\t\tint block[2] = {" << split_points_.front() << ", " << split_points_.back() << "};\n";
    if (input_tensors_.size() == kInputSize1) {
      code.CodeFunction("DoScaleFp32", input_tensor_, input_tensors_[kWeightIndex], offset_, output_tensor_,
                        "&scale_parameter", "block");
    } else {
      code.CodeFunction("DoScaleFp32", input_tensor_, input_tensors_[kWeightIndex], input_tensors_[kBiasIndex],
                        output_tensor_, "&scale_parameter", "block");
    }
  } else {
    if (input_tensors_.size() == kInputSize1) {
      code.CodeBaseStruct("ScaleFp32Args", kRunArgs, input_tensor_, output_tensor_, input_tensors_[kWeightIndex],
                          offset_, split_points_bak_, "&scale_parameter");
    } else {
      code.CodeBaseStruct("ScaleFp32Args", kRunArgs, input_tensor_, output_tensor_, input_tensors_[kWeightIndex],
                          input_tensors_[kBiasIndex], split_points_bak_, "&scale_parameter");
    }
    code.CodeFunction(kParallelLaunch, "DoScaleRun", kRunArgsAddr, "scale_parameter.op_parameter_.thread_num_");
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ScaleFusion, CPUOpCoderCreator<ScaleFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
