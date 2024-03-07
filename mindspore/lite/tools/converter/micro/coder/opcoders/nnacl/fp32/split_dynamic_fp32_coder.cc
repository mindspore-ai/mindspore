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
#include "coder/opcoders/nnacl/fp32/split_dynamic_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"
#include "nnacl/op_base.h"

using mindspore::schema::PrimitiveType_Split;

namespace mindspore::lite::micro::nnacl {
int SplitDynamicFP32Coder::Prepare(CoderContext *const context) {
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  int in_shape_size = static_cast<int>(input_shape.size());
  CHECK_LESS_RETURN(in_shape_size, 1);
  CHECK_LESS_RETURN(SPLIT_STRIDES_SIZE - 1, in_shape_size);
  param_ = reinterpret_cast<SplitParameter *>(parameter_);
  CHECK_NULL_RETURN(param_);

  auto split_dim = param_->split_dim_;
  param_->split_dim_ = split_dim >= 0 ? split_dim : in_shape_size + split_dim;
  std::vector<std::string> strides(in_shape_size);
  strides[in_shape_size - 1] = "1";
  for (int i = static_cast<int>(in_shape_size) - C2NUM; i >= 0; i--) {
    strides[i] = strides[i + 1] + " * " + input_shape[i + 1];
  }
  dynamic_param_.strides_ = "{";
  for (int i = 0; i < in_shape_size; ++i) {
    dynamic_param_.strides_ += strides[i] + ", ";
  }
  dynamic_param_.strides_ += "}";
  CHECK_LESS_RETURN(in_shape_size, param_->split_dim_ + 1);
  if (input_shape.at(param_->split_dim_) == "0") {
    MS_LOG(ERROR) << "input_shape[" << param_->split_dim_ << "] must not be zero!";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(SPLIT_STRIDES_SIZE, param_->split_dim_ + 1);
  if (strides[param_->split_dim_] == "0") {
    MS_LOG(ERROR) << "strides[" << param_->split_dim_ << "] must not be zero!";
    return RET_ERROR;
  }
  dynamic_param_.split_count_ = strides[0] + " * " + input_shape[0] + " / (" + input_shape.at(param_->split_dim_) +
                                " * " + strides[param_->split_dim_] + ")";
  param_->n_dims_ = static_cast<int>(input_shape.size());
  CHECK_LESS_RETURN(param_->num_split_, 1);
  MS_CHECK_TRUE_MSG(param_->split_sizes_[0] != 0 && param_->split_sizes_[param_->num_split_ - 1] != -1,
                    lite::RET_PARAM_INVALID, "Currently, split not support split_size 0 or -1");
  return RET_OK;
}

int SplitDynamicFP32Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/base/split_base.h"}, {"split_base.c"});
  NNaclFp32Serializer code;
  code << "    void *output_ptrs[" << output_tensors_.size() << "] = {";
  for (int i = 0; i < param_->num_split_; i++) {
    code << GetTensorAddr(output_tensors_.at(i), output_tensors_.at(i)->IsConst(), dynamic_mem_manager_, allocator_)
         << ", ";
  }
  code << "};\n";
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  code << "    int input_dim[" << input_shape.size() << "] = {";
  for (auto &dim : input_shape) {
    code << dim << ", ";
  }
  code << "};\n";
  std::string input_data = GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  std::string num_unit = dynamic_param_.split_count_ + " * " + std::to_string(param_->num_split_);
  code.CodeStruct("split_param", *param_, dynamic_param_);
  code.CodeFunction("DoSplit", input_data, "output_ptrs", "input_dim", "0", num_unit, "&split_param",
                    lite::DataTypeSize(input_tensor_->data_type()));
  context->AppendCode(code.str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat32, PrimitiveType_Split, CPUOpCoderCreator<SplitDynamicFP32Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeInt32, PrimitiveType_Split, CPUOpCoderCreator<SplitDynamicFP32Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Split, CPUOpCoderCreator<SplitDynamicFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
