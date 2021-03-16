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
#include "coder/opcoders/nnacl/fp32/arithmetic_fp32_coder.h"
#include <string>
#include <map>
#include <type_traits>
#include "coder/opcoders/file_collector.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "coder/log.h"

namespace mindspore::lite::micro::nnacl {

int ArithmeticFP32Coder::Init(CoderContext *const context) {
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensor_->data_type() == kNumberTypeFloat32 || input_tensor_->data_type() == kNumberTypeFloat16) {
    data_type_ = kDataTypeFloat;
  } else {
    data_type_ = kDataTypeInt;
  }
  arithmetic_parameter_->in_elements_num0_ = input_tensor_->ElementsNum();
  arithmetic_parameter_->in_elements_num1_ = filter_tensor_->ElementsNum();
  arithmetic_parameter_->out_elements_num_ = output_tensor_->ElementsNum();
  for (size_t i = 0; i < input_tensor_->shape().size(); i++) {
    if (arithmetic_parameter_->in_shape0_[i] == -1) {
      MS_CHECK_RET_CODE(
        memcpy_s(arithmetic_parameter_->in_shape0_, DEFAULT_ARITHMETIC_NDIMS * sizeof(int),
                 static_cast<void *>(input_tensor_->shape().data()), input_tensor_->shape().size() * sizeof(int)),
        "memcpy_s in shape0 failed!");
    }
  }
  for (size_t i = 0; i < filter_tensor_->shape().size(); i++) {
    if (arithmetic_parameter_->in_shape1_[i] == -1) {
      MS_CHECK_RET_CODE(
        memcpy_s(arithmetic_parameter_->in_shape1_, DEFAULT_ARITHMETIC_NDIMS * sizeof(int),
                 static_cast<void *>(filter_tensor_->shape().data()), filter_tensor_->shape().size() * sizeof(int)),
        "memcpy_s in shape1 failed!");
    }
  }
  for (size_t i = 0; i < output_tensor_->shape().size(); i++) {
    if (arithmetic_parameter_->out_shape_[i] == -1) {
      MS_CHECK_RET_CODE(
        memcpy_s(arithmetic_parameter_->out_shape_, DEFAULT_ARITHMETIC_NDIMS * sizeof(int),
                 static_cast<void *>(output_tensor_->shape().data()), output_tensor_->shape().size() * sizeof(int)),
        "memcpy_s in out shape failed!");
    }
  }

  if (arithmetic_parameter_->in_elements_num0_ == 1 || arithmetic_parameter_->in_elements_num1_ == 1) {
    switch (arithmetic_parameter_->op_parameter_.type_) {
      case PrimitiveType_MulFusion:
        switch (arithmetic_parameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptMulRelu";
            arithmetic_opt_run_int_ = "ElementOptMulReluInt";
            break;
          case schema::ActivationType_RELU6:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptMulRelu6";
            arithmetic_opt_run_int_ = "ElementOptMulRelu6Int";
            break;
          default:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptMul";
            arithmetic_opt_run_int_ = "ElementOptMulInt";
            break;
        }
        break;
      case PrimitiveType_AddFusion:
        switch (arithmetic_parameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptAddRelu";
            arithmetic_opt_run_int_ = "ElementOptAddReluInt";
            break;
          case schema::ActivationType_RELU6:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptAddRelu6";
            arithmetic_opt_run_int_ = "ElementOptAddRelu6Int";
            break;
          default:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptAdd";
            arithmetic_opt_run_int_ = "ElementOptAddInt";
            break;
        }
        break;
      case PrimitiveType_SubFusion:
        switch (arithmetic_parameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptSubRelu";
            break;
          case schema::ActivationType_RELU6:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptSubRelu6";
            break;
          default:
            arithmetic_parameter_->broadcasting_ = false;
            arithmetic_opt_run_ = "ElementOptSub";
            break;
        }
        break;
      default:
        break;
    }
  }
  return RET_OK;
}

int ArithmeticFP32Coder::BroadcastRun(const std::string &input0, const std::string &input1, const std::string &output,
                                      int dim, int out_count, int out_thread_stride, NNaclFp32Serializer *const code) {
  if (dim > break_pos_) {
    if (data_type_ == kDataTypeInt) {
      *code << "\t\t" << arithmetic_run_int_ << "(((" << input0 << ") + " << out_thread_stride << "), ((" << input1
            << ") + " << out_thread_stride << "), ((" << output << ") + " << out_thread_stride << "), " << out_count
            << ");\n";

    } else {
      *code << "\t\t" << arithmetic_run_ << "(((" << input0 << ") + " << out_thread_stride << "), ((" << input1
            << ") + " << out_thread_stride << "), ((" << output << ") + " << out_thread_stride << "), " << out_count
            << ");\n";
    }
    return RET_OK;
  }
  for (int i = 0; i < arithmetic_parameter_->out_shape_[dim]; ++i) {
    int pos0_ = arithmetic_parameter_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = arithmetic_parameter_->in_shape1_[dim] == 1 ? 0 : i;
    int error_code = BroadcastRun(input0 + "+" + std::to_string(pos0_ * arithmetic_parameter_->in_strides0_[dim]),
                                  input1 + "+" + std::to_string(pos1_ * arithmetic_parameter_->in_strides1_[dim]),
                                  output + "+" + std::to_string(i * arithmetic_parameter_->out_strides_[dim]), dim + 1,
                                  out_count, out_thread_stride, code);
    if (error_code != RET_OK) {
      return error_code;
    }
  }
  return RET_OK;
}

int ArithmeticFP32Coder::Prepare(CoderContext *const context) {
  if (parameter_ == nullptr) {
    return RET_ERROR;
  }
  arithmetic_parameter_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  std::map<int, std::function<void()>> type_setters = {
    {PrimitiveType_MulFusion,
     [this]() {
       switch (arithmetic_parameter_->activation_type_) {
         case schema::ActivationType_RELU:
           arithmetic_run_ = "ElementMulRelu";
           arithmetic_run_int_ = "ElementMulReluInt";
           break;
         case schema::ActivationType_RELU6:
           arithmetic_run_ = "ElementMulRelu6";
           arithmetic_run_int_ = "ElementMulRelu6Int";
           break;
         default:
           arithmetic_run_ = "ElementMul";
           arithmetic_run_int_ = "ElementMulInt";
           break;
       }
     }},
    {PrimitiveType_AddFusion,
     [this]() {
       switch (arithmetic_parameter_->activation_type_) {
         case schema::ActivationType_RELU:
           arithmetic_run_ = "ElementAddRelu";
           arithmetic_run_int_ = "ElementAddReluInt";
           break;
         case schema::ActivationType_RELU6:
           arithmetic_run_ = "ElementAddRelu6";
           arithmetic_run_int_ = "ElementAddRelu6Int";
           break;
         default:
           arithmetic_run_ = "ElementAdd";
           arithmetic_run_int_ = "ElementAddInt";
           break;
       }
     }},
    {PrimitiveType_SubFusion,
     [this]() {
       switch (arithmetic_parameter_->activation_type_) {
         case schema::ActivationType_RELU:
           arithmetic_run_ = "ElementSubRelu";
           break;
         case schema::ActivationType_RELU6:
           arithmetic_run_ = "ElementSubRelu6";
           break;
         default:
           arithmetic_run_ = "ElementSub";
           break;
       }
     }},
    {PrimitiveType_DivFusion,
     [this]() {
       switch (arithmetic_parameter_->activation_type_) {
         case schema::ActivationType_RELU:
           arithmetic_run_ = "ElementDivRelu";
           break;
         case schema::ActivationType_RELU6:
           arithmetic_run_ = "ElementDivRelu6";
           break;
         default:
           arithmetic_run_ = "ElementDiv";
           break;
       }
     }},
    {PrimitiveType_LogicalAnd, [this]() { arithmetic_run_ = "ElementLogicalAnd"; }},
    {PrimitiveType_LogicalOr, [this]() { arithmetic_run_ = "ElementLogicalOr"; }},
    {PrimitiveType_Maximum, [this]() { arithmetic_run_ = "ElementMaximum"; }},
    {PrimitiveType_Minimum, [this]() { arithmetic_run_ = "ElementMinimum"; }},
    {PrimitiveType_FloorDiv, [this]() { arithmetic_run_ = "ElementFloorDiv"; }},
    {PrimitiveType_FloorMod, [this]() { arithmetic_run_ = "ElementFloorMod"; }},
    {PrimitiveType_Equal, [this]() { arithmetic_run_ = "ElementEqual"; }},
    {PrimitiveType_NotEqual, [this]() { arithmetic_run_ = "ElementNotEqual"; }},
    {PrimitiveType_Less, [this]() { arithmetic_run_ = "ElementLess"; }},
    {PrimitiveType_LessEqual, [this]() { arithmetic_run_ = "ElementLessEqual"; }},
    {PrimitiveType_Greater, [this]() { arithmetic_run_ = "ElementGreater"; }},
    {PrimitiveType_GreaterEqual, [this]() { arithmetic_run_ = "ElementGreaterEqual"; }},
    {PrimitiveType_SquaredDifference, [this]() { arithmetic_run_ = "ElementSquaredDifference"; }},
  };
  auto iter = type_setters.find(parameter_->type_);
  if (iter != type_setters.end()) {
    iter->second();
  } else {
    MS_LOG(ERROR) << "Error Operator type " << parameter_;
    arithmetic_run_ = "NULL";
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(Init(context), "do arothmetic code failed!");
  return RET_OK;
}

int ArithmeticFP32Coder::DoCode(CoderContext *const context) {
  int task_id = 0;
  if (arithmetic_parameter_->broadcasting_) {
    outside_ = 1;
    for (auto i = arithmetic_parameter_->ndim_ - 1; i >= 0; --i) {
      if (arithmetic_parameter_->in_shape0_[i] != arithmetic_parameter_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= arithmetic_parameter_->out_shape_[i];
    }
    ComputeStrides(arithmetic_parameter_->in_shape0_, arithmetic_parameter_->in_strides0_,
                   arithmetic_parameter_->ndim_);
    ComputeStrides(arithmetic_parameter_->in_shape1_, arithmetic_parameter_->in_strides1_,
                   arithmetic_parameter_->ndim_);
    ComputeStrides(arithmetic_parameter_->out_shape_, arithmetic_parameter_->out_strides_,
                   arithmetic_parameter_->ndim_);
  }

  int element_num = output_tensor_->ElementsNum();
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(element_num, thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  MS_CHECK_TRUE(!arithmetic_run_.empty(), "arithmetic_run function is nullptr!");
  NNaclFp32Serializer code;
  /**
   * for nnacl's operator combine all arithmetic to nnalc/arithmetic.c
   * this solution is not suitable for micro, for the size of package.
   * */
  if (arithmetic_opt_run_ == "ElementOptSub" || arithmetic_run_ == "ElementSub") {
    Collect(context, {"nnacl/fp32/sub_fp32.h"}, {"sub_fp32.c"});
  } else if (arithmetic_opt_run_ == "ElementOptAdd" || arithmetic_run_ == "ElementAdd") {
    Collect(context, {"nnacl/fp32/add_fp32.h"}, {"add_fp32.c", "arithmetic_fp32.c", "arithmetic_base.c"});
  } else if (arithmetic_opt_run_ == "ElementOptMul" || arithmetic_run_ == "ElementMul") {
    Collect(context, {"nnacl/fp32/mul_fp32.h"}, {"mul_fp32.c"});
  } else if (arithmetic_run_ == "ElementAddRelu") {
    Collect(context, {"nnacl/fp32/add_relu_fp32.h"}, {"add_relu_fp32.c"});
  } else {
    Collect(context, {"nnacl/arithmetic_common.h", "nnacl/fp32/arithmetic_fp32.h"},
            {"arithmetic_common.c", "arithmetic_fp32.c"});
  }

  if (arithmetic_parameter_->broadcasting_) {
    stride = UP_DIV(outside_, thread_num_);
    out_count_ = MSMIN(stride, outside_ - stride * task_id);
    out_thread_stride_ = stride * task_id;
    std::string input0_str = allocator_->GetRuntimeAddr(input_tensor_);
    std::string input1_str = allocator_->GetRuntimeAddr(filter_tensor_);
    std::string output_str = allocator_->GetRuntimeAddr(output_tensor_);
    MS_CHECK_RET_CODE(BroadcastRun(input0_str, input1_str, output_str, 0, out_count_, out_thread_stride_, &code),
                      "do broad cast code failed!");
  } else if (!arithmetic_opt_run_.empty()) {
    code.CodeStruct("arithmetic_parameter", *arithmetic_parameter_);
    if (arithmetic_parameter_->in_elements_num0_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        code.CodeFunction(arithmetic_opt_run_, input_tensor_, filter_tensor_, output_tensor_, count,
                          "&arithmetic_parameter");
      } else {
        code.CodeFunction(arithmetic_opt_run_int_, input_tensor_, filter_tensor_, output_tensor_, count,
                          "&arithmetic_parameter");
      }
    } else if (arithmetic_parameter_->in_elements_num1_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        code.CodeFunction(arithmetic_opt_run_, input_tensor_, filter_tensor_, output_tensor_, count,
                          "&arithmetic_parameter");
      } else {
        code.CodeFunction(arithmetic_opt_run_int_, input_tensor_, filter_tensor_, output_tensor_, count,
                          "&arithmetic_parameter");
      }
    } else {
      MS_LOG(ERROR) << "arithmetic opt code run: at least one of inputs is scalar";
      return RET_ERROR;
    }
  } else {
    if (data_type_ == kDataTypeFloat) {
      code.CodeFunction(arithmetic_run_, input_tensor_, filter_tensor_, output_tensor_, count);
    } else {
      code.CodeFunction(arithmetic_run_int_, input_tensor_, filter_tensor_, output_tensor_, count);
    }
  }
  MS_LOG(DEBUG) << "ArithmeticFP32Code has been called";
  context->AppendCode(code.str());

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_MulFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SubFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_DivFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LogicalAnd, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LogicalOr, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Maximum, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Minimum, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_FloorDiv, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_FloorMod, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SquaredDifference,
                   CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Equal, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_NotEqual, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Less, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LessEqual, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Greater, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_GreaterEqual, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Eltwise, CPUOpCoderCreator<ArithmeticFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
