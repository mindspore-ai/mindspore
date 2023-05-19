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

#include "coder/opcoders/nnacl/fp16/transpose_fp16_coder.h"
#include <vector>
#include <unordered_set>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {
int TransposeFp16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  thread_num_ = 1;
  MS_CHECK_RET_CODE(Init(), "init failed");
  return RET_OK;
}

int TransposeFp16Coder::ResetStatus() {
  auto in_shape = input_tensors_[FIRST_INPUT]->shape();
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
    if (perm_tensor->data_type() != kNumberTypeInt32) {
      MS_LOG(ERROR) << "Unsupported type id: " << perm_tensor->data_type() << " of perm tensor.";
      return RET_ERROR;
    }
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

int TransposeFp16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/transpose_parameter.h",
            "nnacl/errorcode.h",
            "nnacl/fp16/transpose_fp16.h",
          },
          {
            "transpose_fp16.c",
          });

  NNaclFp32Serializer code;
  if (input_tensor_->data() != output_tensor_->data()) {
    code.CodeFunction("memcpy", output_tensor_, input_tensor_, input_tensor_->Size());
    context->AppendCode(code.str());
  }

  auto out_shape = output_tensor_->shape();
  dims_ = static_cast<int>(out_shape.size());
  code.CodeArray("output_shape", out_shape.data(), dims_, true);
  code.CodeStruct("trans_param", *param_);
  if (param_->num_axes_ > DIMENSION_6D) {
    code.CodeFunction("TransposeDimsFp16", input_tensor_, output_tensor_, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.num_axes_", kDefaultTaskId,
                      kDefaultThreadNum);
  } else {
    code.CodeFunction("DoTransposeFp16", input_tensor_, output_tensor_, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.data_num_",
                      "trans_param.num_axes_");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeFp16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeFp16Coder>)
}  // namespace mindspore::lite::micro::nnacl
