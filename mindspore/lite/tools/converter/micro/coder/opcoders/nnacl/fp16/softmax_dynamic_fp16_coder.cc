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

#include "coder/opcoders/nnacl/fp16/softmax_dynamic_fp16_coder.h"
#include <map>
#include <algorithm>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"
#include "tools/common/string_util.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_LogSoftmax;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::lite::micro::nnacl {
int SoftmaxDynamicFP16Coder::Prepare(CoderContext *const context) {
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Input tensor data type is invalid");
  }
  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(output_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Output tensor data type is invalid");
  }
  MS_CHECK_RET_CODE(Init(), "Init failed!");
  return RET_OK;
}

int SoftmaxDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/softmax_fp16.h",
            "nnacl/fp16/log_softmax_fp16.h",
          },
          {
            "softmax_fp16.c",
            "log_softmax_fp16.c",
            "exp_fp16.c",
          });

  auto ret = ComputeWorkSpace();
  MS_CHECK_RET_CODE(ret, "ComputeWorkSpace failed!");
  NNaclFp32Serializer code;
  sum_data_str_ = "(float16_t *)(" + buffer_start_ + ")";
  auto primitive_type = param_->op_parameter_.type_;
  std::string input_data =
    "(float16_t *)(" + GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  std::string output_data =
    "(float16_t *)(" + GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_) + ")";
  code << "    int input_shape[" << input_shape_.size() << "] = " << dynamic_param_.input_shape_ << ";\n";
  if (primitive_type == schema::PrimitiveType_Softmax) {
    code.CodeFunction("SoftmaxFp16", input_data, output_data, sum_data_str_, softmax_struct_.axis_,
                      softmax_struct_.n_dim_, "input_shape");
  } else {
    code.CodeFunction("LogSoftmaxFp16", input_data, output_data, sum_data_str_, "input_shape", softmax_struct_.n_dim_,
                      softmax_struct_.axis_);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int SoftmaxDynamicFP16Coder::Init() {
  param_ = reinterpret_cast<SoftmaxParameter *>(parameter_);
  MS_CHECK_PTR(param_);
  softmax_struct_.base_.param_ = parameter_;
  input_shape_ = shape_info_container_->GetTemplateShape(input_tensor_);
  size_t in_dims = input_shape_.size();
  softmax_struct_.n_dim_ = in_dims;
  softmax_struct_.axis_ = param_->axis_ < 0 ? param_->axis_ + softmax_struct_.n_dim_ : param_->axis_;
  dynamic_param_.element_size_ = AccumulateShape(input_shape_, 0, input_shape_.size());
  dynamic_param_.input_shape_ = "{";
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    dynamic_param_.input_shape_ += input_shape_[i] + ", ";
  }
  dynamic_param_.input_shape_ += "}";
  return RET_OK;
}

int SoftmaxDynamicFP16Coder::ComputeWorkSpace() {
  std::map<std::string, std::vector<int>> real_nums;
  size_t scene_num = 0;
  for (auto &dim_template : input_shape_) {
    auto dim_nums = shape_info_container_->GetRealNums(dim_template);
    MS_CHECK_TRUE_MSG(!dim_nums.empty(), RET_ERROR, "Dynamic shape's num must be greater than 0.");
    real_nums[dim_template] = dim_nums;
    scene_num = std::max(scene_num, dim_nums.size());
  }
  for (size_t i = 0; i < scene_num; ++i) {
    std::vector<int> real_shape(input_shape_.size());
    for (size_t j = 0; j < input_shape_.size(); ++j) {
      if (IsNumber(input_shape_[j])) {
        real_shape[j] = std::stoi(input_shape_[j]);
      } else {
        real_shape[j] = real_nums[input_shape_[j]][i % real_nums[input_shape_[j]].size()];
      }
    }
    int out_plane_size = 1;
    for (int j = 0; j < softmax_struct_.axis_; ++j) {
      MS_CHECK_INT_MUL_NOT_OVERFLOW(out_plane_size, real_shape[j], RET_ERROR);
      out_plane_size *= real_shape[j];
    }
    int in_plane_size = 1;
    for (int j = softmax_struct_.axis_ + 1; j < softmax_struct_.n_dim_; ++j) {
      MS_CHECK_INT_MUL_NOT_OVERFLOW(in_plane_size, real_shape[j], RET_ERROR);
      in_plane_size *= real_shape[j];
    }
    int workspace = out_plane_size * in_plane_size * sizeof(float16);
    buffer_start_ = dynamic_mem_manager_->AllocWorkSpace(workspace, i);
    MS_CHECK_TRUE_MSG(!buffer_start_.empty(), RET_ERROR, "Softmax cannot alloc workspace.");
  }
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Softmax,
                           CPUOpCoderCreator<SoftmaxDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Softmax,
                           CPUOpCoderCreator<SoftmaxDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogSoftmax,
                           CPUOpCoderCreator<SoftmaxDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogSoftmax,
                           CPUOpCoderCreator<SoftmaxDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
