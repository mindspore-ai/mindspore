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

#include "coder/opcoders/nnacl/fp16/slice_dynamic_fp16_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::lite::micro::nnacl {
int SliceDynamicFP16Coder::Prepare(CoderContext *const context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C3NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  CHECK_NULL_RETURN(input_tensors_[FIRST_INPUT]);
  CHECK_NULL_RETURN(input_tensors_[SECOND_INPUT]);
  CHECK_NULL_RETURN(input_tensors_[THIRD_INPUT]);
  CHECK_NULL_RETURN(output_tensor_);
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->IsConst() && input_tensors_[THIRD_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "The second and third input of slice is non-const.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32 &&
                      input_tensors_[THIRD_INPUT]->data_type() == kNumberTypeInt32,
                    RET_INPUT_PARAM_INVALID, "second or third input tensor data type need to be int32.");
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  return Init();
}

int SliceDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/slice_base.h",
          },
          {
            "slice_base.c",
          });
  NNaclFp32Serializer code;
  code.CodeStruct("slice_param", param_, dynamic_param_);
  std::string input_data = GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  std::string output_data = GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  if (!support_parallel_) {
    code.CodeFunction("DoSliceNoParallel", input_data, output_data, "&slice_param", param_.data_type_size_);
  }
  context->AppendCode(code.str());
  return NNACL_OK;
}

int SliceDynamicFP16Coder::Init() {
  MS_CHECK_TRUE_MSG(memset_s(&param_, sizeof(param_), 0, sizeof(param_)) == EOK, RET_ERROR, "memset_s failed.");
  auto begin_tensor = input_tensors_[SECOND_INPUT];
  auto size_tensor = input_tensors_[THIRD_INPUT];
  data_shape_ = shape_info_container_->GetTemplateShape(input_tensor_);
  MS_CHECK_TRUE_MSG(data_shape_.size() == static_cast<size_t>(begin_tensor->ElementsNum()), RET_ERROR,
                    "The begin tensor is invalid.");
  MS_CHECK_TRUE_MSG(data_shape_.size() == static_cast<size_t>(size_tensor->ElementsNum()), RET_ERROR,
                    "The size tensor is invalid.");
  auto begin = reinterpret_cast<int32_t *>(begin_tensor->data());
  CHECK_NULL_RETURN(begin);
  auto size = reinterpret_cast<int32_t *>(size_tensor->data());
  CHECK_NULL_RETURN(size);
  param_.data_type_size_ = static_cast<int>(DataTypeSize(input_tensor_->data_type()));
  param_.param_length_ = static_cast<int>(data_shape_.size());
  if (param_.param_length_ > DIMENSION_8D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_8D;
    return RET_ERROR;
  }
  dynamic_param_.shape_ = "{";
  dynamic_param_.size_ = "{";
  dynamic_param_.end_ = "{";
  for (int i = 0; i < param_.param_length_; ++i) {
    dynamic_param_.shape_ += data_shape_[i] + ", ";
    param_.begin_[i] = begin[i];
    if (size[i] < 0) {
      std::string cur_size = data_shape_[i] + " - " + std::to_string(begin[i]);
      slice_size_.emplace_back(cur_size);
      dynamic_param_.size_ += cur_size + ", ";
    } else {
      slice_size_.emplace_back(std::to_string(size[i]));
      dynamic_param_.size_ += std::to_string(size[i]) + ", ";
    }
    std::string cur_end = std::to_string(param_.begin_[i]) + " + " + slice_size_[i];
    end_.emplace_back(cur_end);
    dynamic_param_.end_ += cur_end + ", ";
  }
  dynamic_param_.shape_ += "}";
  dynamic_param_.size_ += "}";
  dynamic_param_.end_ += "}";
  if (param_.param_length_ < DIMENSION_8D) {
    PadSliceParameterTo8D();
  }
  return RET_OK;
}

void SliceDynamicFP16Coder::PadSliceParameterTo8D() {
  std::vector<int32_t> begin(DIMENSION_8D, 0);
  std::vector<std::string> end(DIMENSION_8D, "");
  std::vector<std::string> slice_size(DIMENSION_8D, "");
  std::vector<std::string> data_shape(DIMENSION_8D, "");
  for (int32_t i = 0; i < param_.param_length_; ++i) {
    begin[i] = param_.begin_[i];
    end[i] = end_[i];
    slice_size[i] =
      slice_size_[i] + " < 0 ? " + data_shape[i] + " - " + std::to_string(begin[i]) + " : " + slice_size_[i];
    data_shape[i] = data_shape_[i];
  }
  data_shape_.resize(DIMENSION_8D);
  slice_size_.resize(DIMENSION_8D);
  end_.resize(DIMENSION_8D);
  int32_t real_index = param_.param_length_ - 1;
  for (int32_t i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param_.begin_[i] = begin[real_index];
      end_[i] = end[real_index];
      slice_size_[i] = slice_size[real_index];
      data_shape_[i] = data_shape[real_index--];
    } else {
      param_.begin_[i] = 0;
      end_[i] = "1";
      slice_size_[i] = "1";
      data_shape_[i] = "1";
    }
  }
  param_.param_length_ = DIMENSION_8D;
  dynamic_param_.shape_.clear();
  dynamic_param_.size_.clear();
  dynamic_param_.end_.clear();
  dynamic_param_.shape_ = "{";
  dynamic_param_.size_ = "{";
  dynamic_param_.end_ = "{";
  for (int i = 0; i < DIMENSION_8D; ++i) {
    dynamic_param_.end_ += end_[i] + ", ";
    dynamic_param_.size_ += slice_size_[i] + ", ";
    dynamic_param_.shape_ += data_shape_[i] + ", ";
  }
  dynamic_param_.shape_ += "}";
  dynamic_param_.size_ += "}";
  dynamic_param_.end_ += "}";
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SliceFusion,
                           CPUOpCoderCreator<SliceDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SliceFusion,
                           CPUOpCoderCreator<SliceDynamicFP16Coder>)
};  // namespace mindspore::lite::micro::nnacl
