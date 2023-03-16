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
#include "coder/opcoders/nnacl/fp16/arithmetic_fp16_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/log.h"

namespace mindspore::lite::micro::nnacl {
void ArithmeticFP16Coder::InitFunTable() {
  fun_table_ = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, "ElementMulReluFp16", "", "", "", ""},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, "ElementMulRelu6Fp16", "", "", "", ""},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, "ElementMulFp16", "", "", "", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, "ElementAddReluFp16", "", "", "", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, "ElementAddRelu6Fp16", "", "", "", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, "ElementAddFp16", "", "", "", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, "ElementSubReluFp16", "", "", "", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, "ElementSubRelu6Fp16", "", "", "", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, "ElementSubFp16", "", "", "", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, "ElementDivReluFp16", "", "", "", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, "ElementDivRelu6Fp16", "", "", "", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, "ElementDivFp16", "", "", "", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, "ElementDivReluFp16", "", "", "", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, "ElementDivRelu6Fp16", "", "", "", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, "ElementDivFp16", "", "", "", ""},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, "ElementLogicalAndFp16", "", "", "", ""},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, "ElementLogicalOrFp16", "", "", "", ""},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, "ElementMaximumFp16", "", "", "", ""},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, "ElementMinimumFp16", "", "", "", ""},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, "ElementFloorModFp16", "", "", "", ""},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, "ElementFloorDivFp16", "", "", "", ""},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, "ElementSquaredDifferenceFp16", "", "", "",
     ""}};
}

int ArithmeticFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 ||
      input_tensors_.at(kWeightIndex)->data_type() != kNumberTypeFloat16 ||
      output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  return ArithmeticFP32Coder::Prepare(context);
}

int ArithmeticFP16Coder::ReSize(CoderContext *const context) {
  CalcMultiplesAndStrides(arithmetic_parameter_);
  if (arithmetic_parameter_->in_shape0_ != arithmetic_parameter_->in_shape1_) {
    MS_LOG(ERROR) << "The shape of input0 and input1 are not equal, and broadCast is not support.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ArithmeticFP16Coder::ExecuteCode(const std::string &input0, const std::string &input1, const std::string &output,
                                     int size, CoderContext *const context, NNaclFp32Serializer *const code) {
  if (arithmetic_func_str_.empty()) {
    return RET_ERROR;
  }
  for (size_t i = 0; i < fun_table_.size(); i++) {
    if (fun_table_[i].primitive_type_ == arithmetic_parameter_->op_parameter_.type_ &&
        fun_table_[i].activation_type_ == arithmetic_parameter_->activation_type_) {
      code->CodeFunction(fun_table_[i].func_, input0, input1, output, size);
      break;
    }
  }
  context->AppendCode(code->str());
  return RET_OK;
}

int ArithmeticFP16Coder::DoCode(CoderContext *const context) {
  int element_num = output_tensor_->ElementsNum();
  input0_ptr_str_ = allocator_->GetRuntimeAddr(input_tensor_, input_tensor_->IsConst());
  input1_ptr_str_ = allocator_->GetRuntimeAddr(filter_tensor_, filter_tensor_->IsConst());
  output_ptr_str_ = allocator_->GetRuntimeAddr(output_tensor_);
  NNaclFp32Serializer code;
  Collect(context,
          {
            "nnacl/fp16/arithmetic_fp16.h",
          },
          {
            "arithmetic_fp16.c",
            "arithmetic_base.c",
          });

  // all elements eltwise calculation
  ChooseArithmeticFunc(false);
  return ExecuteCode(input0_ptr_str_, input1_ptr_str_, output_ptr_str_, element_num, context, &code);
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_MulFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_SubFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_DivFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_RealDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_LogicalAnd, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_LogicalOr, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Maximum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Minimum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_FloorDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_FloorMod, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_SquaredDifference,
                   CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Equal, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_NotEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Less, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_LessEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Greater, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_GreaterEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Eltwise, CPUOpCoderCreator<ArithmeticFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
