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

#include "coder/opcoders/nnacl/fp32/arithmetic_self_fp32_coder.h"
#include <string>
#include <map>
#include "nnacl/fp32/arithmetic_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro::nnacl {

int ArithmeticSelfFP32Coder::ReSize() {
  data_size_ = input_tensor_->ElementsNum();
  thread_sz_count_ = MSMIN(thread_num_, static_cast<int>(data_size_));
  MS_CHECK_TRUE(thread_sz_count_ > 0, "thread_sz_count_ <= 0");
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ArithmeticSelfFP32Coder::Prepare(CoderContext *const context) {
  if (parameter_ == nullptr) {
    return RET_ERROR;
  }
  std::map<int, std::function<void()>> type_setters = {
    {PrimitiveType_Abs, [this]() { arithmetic_self_run_ = "ElementAbs"; }},
    {PrimitiveType_Cos, [this]() { arithmetic_self_run_ = "ElementCos"; }},
    {PrimitiveType_Log, [this]() { arithmetic_self_run_ = "ElementLog"; }},
    {PrimitiveType_Square, [this]() { arithmetic_self_run_ = "ElementSquare"; }},
    {PrimitiveType_Sqrt, [this]() { arithmetic_self_run_ = "ElementSqrt"; }},
    {PrimitiveType_Rsqrt, [this]() { arithmetic_self_run_ = "ElementRsqrt"; }},
    {PrimitiveType_Sin, [this]() { arithmetic_self_run_ = "ElementSin"; }},
    {PrimitiveType_LogicalNot, [this]() { arithmetic_self_run_ = "ElementLogicalNot"; }},
    {PrimitiveType_Floor, [this]() { arithmetic_self_run_ = "ElementFloor"; }},
    {PrimitiveType_Ceil, [this]() { arithmetic_self_run_ = "ElementCeil"; }},
    {PrimitiveType_Round, [this]() { arithmetic_self_run_ = "ElementRound"; }},
    {PrimitiveType_Neg, [this]() { arithmetic_self_run_ = "ElementNegative"; }},
  };
  auto iter = type_setters.find(parameter_->type_);
  if (iter != type_setters.end()) {
    iter->second();
  } else {
    MS_LOG(ERROR) << "Error Operator type " << parameter_;
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(ReSize(), "ReSize failed");
  return RET_OK;
}

int ArithmeticSelfFP32Coder::DoCode(CoderContext *const context) {
  int size = MSMIN(thread_sz_stride_, static_cast<int>(data_size_ - kDefaultTaskId * thread_sz_stride_));

  MS_CHECK_TRUE(!arithmetic_self_run_.empty(), "arithmetic_run function is nullptr!");

  Collect(context, {"nnacl/arithmetic_common.h", "nnacl/fp32/arithmetic_self.h"}, {"nnacl/fp32/arithmetic_self.c"});
  NNaclFp32Serializer code;
  code.CodeFunction(arithmetic_self_run_, input_tensor_, output_tensor_, size);

  MS_LOG(DEBUG) << "ArithmeticSelfFP32Coder has been called";
  context->AppendCode(code.str());

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Abs, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Cos, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Log, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Square, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Sqrt, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Rsqrt, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Sin, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LogicalNot,
                   CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Floor, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Ceil, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Round, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Neg, CPUOpCoderCreator<ArithmeticSelfFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
