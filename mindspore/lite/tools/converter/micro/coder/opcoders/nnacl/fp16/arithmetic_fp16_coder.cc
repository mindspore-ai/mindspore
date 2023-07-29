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
#include "coder/log.h"
#include "nnacl/broadcast_to_parameter.h"
#include "base/float16.h"

namespace mindspore::lite::micro::nnacl {
void ArithmeticFP16Coder::InitFunTable() {
  fun_table_ = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, "ElementMulReluFp16", "", "", "ElementOptMulReluFp16", ""},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, "ElementMulRelu6Fp16", "", "", "ElementOptMulRelu6Fp16",
     ""},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, "ElementMulFp16", "", "", "ElementOptMulFp16", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, "ElementAddReluFp16", "", "", "ElementOptAddReluFp16", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, "ElementAddRelu6Fp16", "", "", "ElementOptAddRelu6Fp16",
     ""},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, "ElementAddFp16", "", "", "ElementOptAddFp16", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, "ElementSubReluFp16", "", "", "ElementOptSubReluFp16", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, "ElementSubRelu6Fp16", "", "", "ElementOptSubRelu6Fp16",
     ""},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, "ElementSubFp16", "", "", "ElementOptSubFp16", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, "ElementDivReluFp16", "", "", "ElementOptDivReluFp16", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, "ElementDivRelu6Fp16", "", "", "ElementOptDivRelu6Fp16",
     ""},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, "ElementDivFp16", "", "", "ElementOptDivFp16", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, "ElementDivReluFp16", "", "", "ElementOptDivReluFp16", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, "ElementDivRelu6Fp16", "", "", "ElementOptDivRelu6Fp16", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, "ElementDivFp16", "", "", "ElementOptDivFp16", ""},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, "ElementLogicalAndFp16", "", "",
     "ElementOptLogicalAndFp16", ""},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, "ElementLogicalOrFp16", "", "",
     "ElementOptLogicalOrFp16", ""},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, "ElementMaximumFp16", "", "", "ElementOptMaximumFp16",
     ""},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, "ElementMinimumFp16", "", "", "ElementOptMinimumFp16",
     ""},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, "ElementFloorModFp16", "", "",
     "ElementOptFloorModFp16", ""},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, "ElementFloorDivFp16", "", "",
     "ElementOptFloorDivFp16", ""},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, "ElementSquaredDifferenceFp16", "", "",
     "ElementOptSquaredDifferenceFp16", ""}};
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
  if (input_tensor_->shape() != output_tensor_->shape() && filter_tensor_->shape() != output_tensor_->shape()) {
    broadcast_temp_ = allocator_->Malloc(kNumberTypeFloat16, output_tensor_->Size(), kWorkspace);
    MS_CHECK_TRUE_MSG(broadcast_temp_ != nullptr, RET_NULL_PTR, "malloc broadcast temp data failed");
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
      if (IsScalarClac()) {
        code->CodeFunction(fun_table_[i].opt_func_, input0, input1, output, size,
                           arithmetic_parameter_->in_elements_num0_ == 1);
        break;
      }
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
            "nnacl/base/broadcast_to.h",
          },
          {
            "arithmetic_fp16.c",
            "arithmetic_base.c",
            "broadcast_to.c",
          });
  if (IsScalarClac()) {
    ChooseArithmeticFunc(true);
    return ExecuteCode(input0_ptr_str_, input1_ptr_str_, output_ptr_str_, element_num, context, &code);
  }

  // all elements eltwise calculation
  ChooseArithmeticFunc(false);
  auto in0_shape = input_tensor_->shape();
  auto in1_shape = filter_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  BroadcastShapeInfo broadcast_info;
  auto ret = memset_s(&broadcast_info, sizeof(BroadcastShapeInfo), 0, sizeof(BroadcastShapeInfo));
  MS_CHECK_TRUE_MSG(ret == EOK, RET_ERROR, "memset failed");
  ret = memcpy_s(broadcast_info.output_shape_, MAX_SHAPE_SIZE * sizeof(int), out_shape.data(),
                 out_shape.size() * sizeof(int));
  MS_CHECK_TRUE_MSG(ret == EOK, RET_ERROR, "memcpy output-info failed");
  broadcast_info.output_shape_size_ = static_cast<int>(out_shape.size());
  if (in0_shape != out_shape) {
    ret = memcpy_s(broadcast_info.input_shape_, MAX_SHAPE_SIZE * sizeof(int), in0_shape.data(),
                   in0_shape.size() * sizeof(int));
    MS_CHECK_TRUE_MSG(ret == EOK, RET_ERROR, "memcpy in0-info failed");
    broadcast_info.input_shape_size_ = static_cast<int>(in0_shape.size());
    code.CodeStruct("in0_broadcast_info", broadcast_info);
    code.CodeFunction("BroadcastToSize16", input0_ptr_str_, "&in0_broadcast_info", output_ptr_str_);
    input0_ptr_str_ = output_ptr_str_;
  }
  if (in1_shape != out_shape) {
    ret = memcpy_s(broadcast_info.input_shape_, MAX_SHAPE_SIZE * sizeof(int), in1_shape.data(),
                   in1_shape.size() * sizeof(int));
    MS_CHECK_TRUE_MSG(ret == EOK, RET_ERROR, "memcpy in0-info failed");
    broadcast_info.input_shape_size_ = static_cast<int>(in1_shape.size());
    code.CodeStruct("in1_broadcast_info", broadcast_info);
    auto temp = output_ptr_str_;
    if (input0_ptr_str_ == output_ptr_str_) {
      temp = allocator_->GetRuntimeAddr(static_cast<float16 *>(broadcast_temp_));
    }
    code.CodeFunction("BroadcastToSize16", input1_ptr_str_, "&in1_broadcast_info", temp);
    input1_ptr_str_ = temp;
  }
  return ExecuteCode(input0_ptr_str_, input1_ptr_str_, output_ptr_str_, element_num, context, &code);
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_MulFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SubFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_DivFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_RealDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogicalAnd, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogicalOr, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Maximum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Minimum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_FloorDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_FloorMod, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SquaredDifference, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Equal, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_NotEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Less, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LessEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Greater, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_GreaterEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Eltwise, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MulFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SubFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_DivFusion, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_RealDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogicalAnd, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogicalOr, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Maximum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Minimum, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_FloorDiv, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_FloorMod, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SquaredDifference, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Equal, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_NotEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Less, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LessEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Greater, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_GreaterEqual, CPUOpCoderCreator<ArithmeticFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Eltwise, CPUOpCoderCreator<ArithmeticFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
