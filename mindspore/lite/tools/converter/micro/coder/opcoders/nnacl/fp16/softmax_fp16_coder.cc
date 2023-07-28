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

#include "coder/opcoders/nnacl/fp16/softmax_fp16_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_LogSoftmax;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::lite::micro::nnacl {
int SoftMaxFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  auto ret = SoftmaxBaseCoder::Init();
  MS_CHECK_RET_CODE(ret, "SoftmaxBaseCoder::Init() failed!");
  ret = SoftmaxBaseCoder::MallocTmpBuffer();
  MS_CHECK_RET_CODE(ret, "SoftmaxBaseCoder::MallocTmpBuffer() failed!");
  sum_data_ = static_cast<float16 *>(allocator_->Malloc(input_tensor_->data_type(), sum_data_size_, kWorkspace));
  return RET_OK;
}

int SoftMaxFP16Coder::DoCode(CoderContext *const context) {
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
  NNaclFp32Serializer code;
  std::string param_name = "softmax_parameter";
  code.CodeStruct(param_name, *softmax_param_);
  code.CodeStruct("input_shape", input_shape_, DIMENSION_5D);
  code.CodeFunction("memset", sum_data_, "0", sum_data_size_);
  auto primitive_type = softmax_param_->op_parameter_.type_;
  if (support_parallel_) {
    code << "    " << param_name << ".op_parameter_.thread_num_ = 1;\n";
  }
  if (primitive_type == schema::PrimitiveType_Softmax) {
    code.CodeFunction("SoftmaxFp16", input_tensor_, output_tensor_, sum_data_, "softmax_parameter.axis_", n_dim_,
                      "input_shape");
  } else {
    code.CodeFunction("LogSoftmaxFp16", input_tensor_, output_tensor_, sum_data_, "input_shape", n_dim_,
                      "softmax_parameter.axis_");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Softmax, CPUOpCoderCreator<SoftMaxFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Softmax, CPUOpCoderCreator<SoftMaxFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogSoftmax, CPUOpCoderCreator<SoftMaxFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogSoftmax, CPUOpCoderCreator<SoftMaxFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
