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
#include "coder/opcoders/nnacl/fp16/arithmetic_dynamic_fp16_coder.h"
#include <map>
#include <algorithm>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/log.h"
#include "coder/utils/coder_utils.h"
#include "tools/common/string_util.h"

namespace mindspore::lite::micro::nnacl {
namespace {
std::string wrap_void(const std::string &a) { return "(void *)(" + a + ")"; }
}  // namespace

void ArithmeticDynamicFP16Coder::InitFunTable() {
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

int ArithmeticDynamicFP16Coder::Prepare(CoderContext *const context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Tensor data type is invalid");
  }
  MS_CHECK_TRUE_MSG(output_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Tensor data type is invalid");
  filter_tensor_ = input_tensors_.at(SECOND_INPUT);
  MS_CHECK_PTR(filter_tensor_);
  param_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  MS_CHECK_PTR(param_);
  auto primitive_type = param_->op_parameter_.type_;
  if (primitive_type == schema::PrimitiveType_Eltwise) {
    switch (param_->eltwise_mode_) {
      case schema::EltwiseMode_PROD:
        primitive_type = schema::PrimitiveType_MulFusion;
        break;
      case schema::EltwiseMode_SUM:
        primitive_type = schema::PrimitiveType_AddFusion;
        break;
      case schema::EltwiseMode_MAXIMUM:
        primitive_type = schema::PrimitiveType_Maximum;
        break;
      default:
        MS_LOG(ERROR) << "Eltwise mode not support, mode:" << param_->eltwise_mode_;
        return RET_ERROR;
    }
  }
  InitRunFunction(primitive_type);
  InitDynamicParams();
  ResetStatus();
  CalcMultiplesAndStrides();
  return RET_OK;
}

int ArithmeticDynamicFP16Coder::DoCode(CoderContext *const context) {
  input0_ptr_str_ = GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  input1_ptr_str_ = GetTensorAddr(filter_tensor_, filter_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  output_ptr_str_ = GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
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

  // all elements eltwise calculation
  arithmetic_func_str_ = wrap_void(arithmetic_run_);
  // do broadcast
  MS_CHECK_TRUE(DoBroadcast(&code) == RET_OK, "DoBroadcast failed");
  return ExecuteCode("(float16_t *)(" + input0_ptr_str_ + ")", "(float16_t *)(" + input1_ptr_str_ + ")",
                     "(float16_t *)(" + output_ptr_str_ + ")", dynamic_param_.out_elements_num_, context, &code);
}

void ArithmeticDynamicFP16Coder::InitDynamicParams() {
  auto in0_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::vector<std::string> in1_shape;
  if (filter_tensor_->IsConst()) {
    auto tensor_shape = filter_tensor_->shape();
    (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(in1_shape),
                         [](const auto &dim) { return std::to_string(dim); });
  } else {
    in1_shape = shape_info_container_->GetTemplateShape(filter_tensor_);
  }
  auto out_shape = shape_info_container_->GetTemplateShape(output_tensor_);
  dynamic_param_.in_shape0_ = "{";
  dynamic_param_.in_shape1_ = "{";
  dynamic_param_.out_shape_ = "{";
  for (auto shape : in0_shape) {
    dynamic_param_.in_shape0_ += shape + ", ";
  }
  for (auto shape : in1_shape) {
    dynamic_param_.in_shape1_ += shape + ", ";
  }
  for (auto shape : out_shape) {
    dynamic_param_.out_shape_ += shape + ", ";
  }
  dynamic_param_.in_shape0_ += "}";
  dynamic_param_.in_shape1_ += "}";
  dynamic_param_.out_shape_ += "}";
  dynamic_param_.in_elements_num0_ = AccumulateShape(in0_shape, 0, in0_shape.size());
  dynamic_param_.in_elements_num1_ = AccumulateShape(in1_shape, 0, in1_shape.size());
  dynamic_param_.out_elements_num_ = AccumulateShape(out_shape, 0, out_shape.size());
}

void ArithmeticDynamicFP16Coder::InitRunFunction(int primitive_type) {
  InitFunTable();
  for (size_t i = 0; i < fun_table_.size(); i++) {
    if (fun_table_[i].primitive_type_ == primitive_type && fun_table_[i].activation_type_ == param_->activation_type_) {
      arithmetic_run_ = fun_table_[i].func_;
      arithmetic_run_int_ = fun_table_[i].int_func_;
      arithmetic_run_bool_ = fun_table_[i].bool_func_;
      arithmetic_opt_run_ = fun_table_[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table_[i].opt_int_func_;
    }
  }
  arithmetic_func_type_ = kArithmeticFuncFloat;
}

void ArithmeticDynamicFP16Coder::ResetStatus() {
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::vector<std::string> filter_shape;
  if (filter_tensor_->IsConst()) {
    auto tensor_shape = filter_tensor_->shape();
    (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(filter_shape),
                         [](const auto &dim) { return std::to_string(dim); });
  } else {
    filter_shape = shape_info_container_->GetTemplateShape(filter_tensor_);
  }
  auto dim_num = input_shape.size() >= filter_shape.size() ? input_shape.size() : filter_shape.size();
  for (size_t i = 0; i < dim_num - input_shape.size(); ++i) {
    in0_shape_.emplace_back("1");
  }
  in0_shape_.insert(in0_shape_.end(), input_shape.begin(), input_shape.end());
  for (size_t i = 0; i < dim_num - filter_shape.size(); ++i) {
    in1_shape_.emplace_back("1");
  }
  in1_shape_.insert(in1_shape_.end(), filter_shape.begin(), filter_shape.end());
}

void ArithmeticDynamicFP16Coder::CalcMultiplesAndStrides() {
  out_shape_ = shape_info_container_->GetTemplateShape(output_tensor_);
  dynamic_param_.multiples0_ = "{";
  dynamic_param_.multiples1_ = "{";
  for (size_t i = 0; i < param_->ndim_; i++) {
    if (in0_shape_[i] != "0") {
      dynamic_param_.multiples0_ += out_shape_[i] + " / " + in0_shape_[i] + ", ";
    }
    if (in1_shape_[i] != "0") {
      dynamic_param_.multiples1_ += out_shape_[i] + " / " + in1_shape_[i] + ", ";
    }
  }
  dynamic_param_.multiples0_ += "}";
  dynamic_param_.multiples1_ += "}";

  // cal strides
  in0_strides_.resize(param_->ndim_);
  in1_strides_.resize(param_->ndim_);
  out_strides_.resize(param_->ndim_);
  ComputeStrides(in0_shape_, &in0_strides_);
  ComputeStrides(in1_shape_, &in1_strides_);
  ComputeStrides(out_shape_, &out_strides_);
  dynamic_param_.in_strides0_ = "{";
  dynamic_param_.in_strides1_ = "{";
  dynamic_param_.out_strides_ = "{";
  for (size_t i = 0; i < param_->ndim_; ++i) {
    dynamic_param_.in_strides0_ += in0_strides_[i] + ", ";
    dynamic_param_.in_strides1_ += in1_strides_[i] + ", ";
    dynamic_param_.out_strides_ += out_strides_[i] + ", ";
  }
  dynamic_param_.in_strides0_ += "}";
  dynamic_param_.in_strides1_ += "}";
  dynamic_param_.out_strides_ += "}";
}

void ArithmeticDynamicFP16Coder::ComputeStrides(const std::vector<std::string> &shape,
                                                std::vector<std::string> *strides) {
  std::string stride = "1";
  for (int i = param_->ndim_ - 1; i >= 0; i--) {
    (*strides)[i] = stride;
    stride += "*=" + shape[i];
  }
}

int ArithmeticDynamicFP16Coder::DoBroadcast(NNaclFp32Serializer *const code) {
  auto in0_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::vector<std::string> in1_shape;
  if (filter_tensor_->IsConst()) {
    auto tensor_shape = filter_tensor_->shape();
    (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(in1_shape),
                         [](const auto &dim) { return std::to_string(dim); });
  } else {
    in1_shape = shape_info_container_->GetTemplateShape(filter_tensor_);
  }
  auto out_shape = shape_info_container_->GetTemplateShape(output_tensor_);
  broadcast_info_.output_shape_size_ = static_cast<int>(out_shape_.size());
  if (in0_shape != out_shape) {
    broadcast_info_.input_shape_size_ = static_cast<int>(in0_shape.size());
    dynamic_shape_info_.input_shape_ = dynamic_param_.in_shape0_;
    dynamic_shape_info_.output_shape_ = dynamic_param_.out_shape_;
    code->CodeStruct("in0_broadcast_info", broadcast_info_, dynamic_shape_info_);
    code->CodeFunction("BroadcastToSize16", input0_ptr_str_, "&in0_broadcast_info", output_ptr_str_);
    input0_ptr_str_ = output_ptr_str_;
  }
  MS_CHECK_TRUE_RET(in1_shape != out_shape, RET_OK);
  broadcast_info_.input_shape_size_ = static_cast<int>(in1_shape.size());
  dynamic_shape_info_.input_shape_ = dynamic_param_.in_shape1_;
  dynamic_shape_info_.output_shape_ = dynamic_param_.out_shape_;
  code->CodeStruct("in1_broadcast_info", broadcast_info_, dynamic_shape_info_);
  auto temp = output_ptr_str_;
  if (input0_ptr_str_ == output_ptr_str_) {
    std::map<std::string, std::vector<int>> real_nums;
    size_t scene_num = 0;
    for (auto &dim_template : out_shape) {
      auto dim_nums = shape_info_container_->GetRealNums(dim_template);
      MS_CHECK_TRUE_MSG(!dim_nums.empty(), RET_ERROR, "Dynamic shape's num must be greater than 0.");
      real_nums[dim_template] = dim_nums;
      scene_num = std::max(scene_num, dim_nums.size());
    }
    for (size_t i = 0; i < scene_num; ++i) {
      int out_element_num = 1;
      for (size_t j = 0; j < out_shape.size(); ++j) {
        if (IsNumber(out_shape[j])) {
          out_element_num *= std::stoi(out_shape[j]);
        } else {
          out_element_num *= real_nums[out_shape[j]][i % real_nums[out_shape[j]].size()];
        }
      }
      int workspace = out_element_num * DataTypeSize(kNumberTypeFloat16);
      temp = dynamic_mem_manager_->AllocWorkSpace(workspace, i);
      MS_CHECK_TRUE_MSG(!temp.empty(), RET_ERROR, "Arithmetic cannot alloc workspace.");
    }
  }
  code->CodeFunction("BroadcastToSize16", input1_ptr_str_, "&in1_broadcast_info", temp);
  input1_ptr_str_ = temp;
  return RET_OK;
}

int ArithmeticDynamicFP16Coder::ExecuteCode(const std::string &input0, const std::string &input1,
                                            const std::string &output, const std::string size,
                                            CoderContext *const context, NNaclFp32Serializer *const code) {
  if (arithmetic_func_str_.empty()) {
    return RET_ERROR;
  }
  for (size_t i = 0; i < fun_table_.size(); i++) {
    if (fun_table_[i].primitive_type_ == param_->op_parameter_.type_ &&
        fun_table_[i].activation_type_ == param_->activation_type_) {
      code->CodeFunction(fun_table_[i].func_, input0, input1, output, size);
      break;
    }
  }
  context->AppendCode(code->str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_AddFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_MulFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SubFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_DivFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_RealDiv,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogicalAnd,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogicalOr,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Maximum,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Minimum,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_FloorDiv,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_FloorMod,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SquaredDifference,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Equal,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_NotEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Less,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LessEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Greater,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_GreaterEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Eltwise,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_AddFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MulFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SubFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_DivFusion,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_RealDiv,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogicalAnd,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogicalOr,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Maximum,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Minimum,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_FloorDiv,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_FloorMod,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SquaredDifference,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Equal,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_NotEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Less,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LessEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Greater,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_GreaterEqual,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Eltwise,
                           CPUOpCoderCreator<ArithmeticDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
