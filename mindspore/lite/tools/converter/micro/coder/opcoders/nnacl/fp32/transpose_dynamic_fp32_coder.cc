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

#include "coder/opcoders/nnacl/fp32/transpose_dynamic_fp32_coder.h"
#include <vector>
#include <unordered_set>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {
int TransposeDynamicFp32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE_MSG(input_tensor_->data_type() == kNumberTypeInt32 || input_tensor_->data_type() == kNumberTypeFloat32,
                    RET_INPUT_PARAM_INVALID, "Input tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32, RET_INPUT_PARAM_INVALID,
                    "Perm tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(
    output_tensor_->data_type() == kNumberTypeInt32 || output_tensor_->data_type() == kNumberTypeFloat32,
    RET_INPUT_PARAM_INVALID, "Output tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "The second input of transpose is non-const.");
  thread_num_ = 1;
  MS_CHECK_RET_CODE(Init(), "init failed");
  return RET_OK;
}

int TransposeDynamicFp32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/transpose_parameter.h",
            "nnacl/errorcode.h",
            "nnacl/fp32/transpose_fp32.h",
          },
          {
            "transpose_fp32.c",
          });

  NNaclFp32Serializer code;
  dims_ = static_cast<int>(out_shapes_.size());
  code << "const int32_t output_shape = [" << dims_ << "] = {";
  for (size_t i = 0; i < out_shapes_.size(); ++i) {
    code << out_shapes_[i] << ", ";
  }
  code << "};\n";
  code.CodeStruct("trans_param", *param_, dynamic_param_);
  auto input_str = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  auto output_str = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  if (param_->num_axes_ > DIMENSION_6D) {
    code.CodeFunction("TransposeDimsFp32", input_str, output_str, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.num_axes_", kDefaultTaskId,
                      kDefaultThreadNum);
  } else {
    code.CodeFunction("DoTransposeFp32", input_str, output_str, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.data_num_",
                      "trans_param.num_axes_");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int TransposeDynamicFp32Coder::Init() {
  param_ = reinterpret_cast<TransposeParameter *>(parameter_);
  MS_CHECK_PTR(param_);
  param_->num_axes_ = 0;
  if (input_tensors_.size() == C2NUM) {
    param_->num_axes_ = input_tensors_[SECOND_INPUT]->ElementsNum();
  }
  if (input_tensor_->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    return RET_OK;
  }
  // get perm data
  auto ret = ResetStatus();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do transpose reset failed.";
    return ret;
  }

  ret = ComputeOfflineInfo();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do compute transpose offline info failed.";
    return ret;
  }
  return RET_OK;
}

int TransposeDynamicFp32Coder::ResetStatus() {
  auto in_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  if (in_shape.size() > MAX_TRANSPOSE_DIM_SIZE) {
    MS_LOG(ERROR) << "input shape out of range.";
    return RET_ERROR;
  }
  int trans_nd[MAX_TRANSPOSE_DIM_SIZE] = {0, 2, 1};
  int *perm_data{nullptr};
  if (in_shape.size() != static_cast<size_t>(param_->num_axes_)) {
    perm_data = trans_nd;
    if (in_shape.size() == C3NUM && param_->num_axes_ == C4NUM) {
      param_->num_axes_ = C3NUM;
    }
    if (param_->num_axes_ == 0) {
      for (int i = 0; i < static_cast<int>(in_shape.size()); ++i) {
        trans_nd[i] = static_cast<int>(in_shape.size()) - 1 - i;
      }
      param_->num_axes_ = static_cast<int>(in_shape.size());
    }
  } else {
    if (input_tensors_.size() != C2NUM) {
      MS_LOG(ERROR) << "input tensors size is not equal to 2.";
      return RET_ERROR;
    }
    auto perm_tensor = input_tensors_.at(SECOND_INPUT);
    perm_data = reinterpret_cast<int *>(perm_tensor->data());
    MSLITE_CHECK_PTR(perm_data);
    std::vector<int> perm(perm_data, perm_data + input_tensors_[SECOND_INPUT]->ElementsNum());
    if (perm.size() != std::unordered_set<int>(perm.cbegin(), perm.cend()).size()) {
      MS_LOG(ERROR) << "Invalid perm, the same element exits in perm.";
      return RET_ERROR;
    }
  }
  MS_CHECK_TRUE_MSG(param_->num_axes_ <= MAX_TRANSPOSE_DIM_SIZE, RET_ERROR, "transpose perm is invalid.");
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }
  return RET_OK;
}

int TransposeDynamicFp32Coder::ComputeOfflineInfo() {
  in_shapes_ = shape_info_container_->GetTemplateShape(input_tensor_);
  out_shapes_ = shape_info_container_->GetTemplateShape(output_tensor_);
  const int ori_stride = 1;
  dynamic_param_.strides_ = std::to_string(ori_stride) + ", ";
  dynamic_param_.out_strides_ = std::to_string(ori_stride) + ", ";
  dynamic_param_.data_num_ = AccumulateShape(in_shapes_);
  std::vector<std::string> strides(param_->num_axes_);
  std::vector<std::string> out_strides(param_->num_axes_);
  strides[param_->num_axes_ - 1] = "1";
  out_strides[param_->num_axes_ - 1] = "1";
  for (int i = param_->num_axes_ - C2NUM; i >= 0; --i) {
    strides[i] = in_shapes_[i + 1] + " * " + strides[i + 1];
    out_strides[i] = out_shapes_[i + 1] + " * " + out_strides[i + 1];
  }
  dynamic_param_.strides_ = "{";
  dynamic_param_.out_strides_ = "{";
  for (int i = 0; i < param_->num_axes_; ++i) {
    dynamic_param_.strides_ += strides[i] + ", ";
    dynamic_param_.out_strides_ += out_strides[i] + ", ";
  }
  dynamic_param_.strides_ += "}";
  dynamic_param_.out_strides_ += "}";
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat32, PrimitiveType_Transpose,
                           CPUOpCoderCreator<TransposeDynamicFp32Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeInt32, PrimitiveType_Transpose,
                           CPUOpCoderCreator<TransposeDynamicFp32Coder>)
}  // namespace mindspore::lite::micro::nnacl
