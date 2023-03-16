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
#include "coder/opcoders/file_collector.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "coder/opcoders/parallel.h"
#include "coder/log.h"

namespace mindspore::lite::micro::nnacl {
namespace {
std::string wrap_void(const std::string &a) { return "(void *)(" + a + ")"; }
std::string wrap_uint8(const std::string &a) { return "(uint8_t *)(" + a + ")"; }
std::string wrap_offset(const std::string &a, int offset) { return "(" + a + "+" + std::to_string(offset) + ")"; }
}  // namespace

void ArithmeticFP32Coder::InitFunTable() {
  fun_table_ = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, "ElementMulRelu", "ElementMulReluInt", "",
     "ElementOptMulRelu", "ElementOptMulReluInt"},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, "ElementMulRelu6", "ElementMulRelu6Int", "",
     "ElementOptMulRelu6", "ElementOptMulRelu6Int"},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, "ElementMul", "ElementMulInt", "", "ElementOptMul",
     "ElementOptMulInt"},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, "ElementAddRelu", "", "", "ElementOptAddRelu", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, "ElementAddRelu6", "", "", "ElementOptAddRelu6", ""},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, "ElementAdd", "ElementAddInt", "", "ElementOptAdd",
     "ElementOptAddInt"},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, "ElementSubRelu", "", "", "ElementOptSubRelu", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, "ElementSubRelu6", "", "", "ElementOptSubRelu6", ""},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, "ElementSub", "ElementSubInt", "", "ElementOptSub",
     "ElementOptSubInt"},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, "ElementDivRelu", "", "", "ElementOptDivRelu", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, "ElementDivRelu6", "", "", "ElementOptDivRelu6", ""},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, "ElementDiv", "", "", "ElementOptDiv",
     "ElementOptDivInt"},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, "ElementDivRelu", "", "", "ElementOptDivRelu", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, "ElementDivRelu6", "", "", "ElementOptDivRelu6", ""},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, "ElementDiv", "", "", "ElementOptDiv",
     "ElementOptDivInt"},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, "ElementLogicalAnd", "ElementLogicalAndInt",
     "ElementLogicalAndBool", "", ""},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, "ElementLogicalOr", "", "ElementLogicalOrBool", "",
     ""},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, "ElementMaximum", "ElementMaximumInt", "", "", ""},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, "ElementMinimum", "ElementMinimumInt", "", "", ""},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, "ElementFloorMod", "ElementFloorModInt", "", "", ""},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, "ElementFloorDiv", "ElementFloorDivInt", "", "", ""},
    {PrimitiveType_Mod, schema::ActivationType_NO_ACTIVATION, "ElementMod", "ElementModInt", "", "ElementOptMod",
     "ElementOptModInt"},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, "ElementSquaredDifference", "", "", "",
     ""}};
}

void ArithmeticFP32Coder::InitRunFunction(int primitive_type) {
  InitFunTable();
  for (size_t i = 0; i < fun_table_.size(); i++) {
    if (fun_table_[i].primitive_type_ == primitive_type &&
        fun_table_[i].activation_type_ == arithmetic_parameter_->activation_type_) {
      arithmetic_run_ = fun_table_[i].func_;
      arithmetic_run_int_ = fun_table_[i].int_func_;
      arithmetic_run_bool_ = fun_table_[i].bool_func_;
      arithmetic_opt_run_ = fun_table_[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table_[i].opt_int_func_;
    }
  }
  TypeId input_type_id = input_tensor_->data_type();
  data_type_len_ = lite::DataTypeSize(input_tensor_->data_type());
  if (input_type_id == kNumberTypeFloat32 || input_type_id == kNumberTypeFloat || input_type_id == kNumberTypeFloat16) {
    arithmetic_func_type_ = kArithmeticFuncFloat;
  } else if (input_type_id == kNumberTypeBool) {
    arithmetic_func_type_ = kArithmeticFuncBool;
  } else if (input_type_id == kNumberTypeInt || input_type_id == kNumberTypeInt32) {
    arithmetic_func_type_ = kArithmeticFuncInt;
  } else {
    arithmetic_func_type_ = kArithmeticFuncUnknow;
  }
}

int ArithmeticFP32Coder::ReSize(CoderContext *const context) {
  CalcMultiplesAndStrides(arithmetic_parameter_);
  if (arithmetic_parameter_->broadcasting_) {
    outside_ = 1;
    int resize_n_index = static_cast<int>(arithmetic_parameter_->ndim_) - 1;
    if (resize_n_index < 0 || resize_n_index >= static_cast<int>(max_dims_)) {
      return RET_ERROR;
    }
    for (auto i = resize_n_index; i >= 0; --i) {
      if (arithmetic_parameter_->in_shape0_[i] != arithmetic_parameter_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= arithmetic_parameter_->out_shape_[i];
    }
  }
  int ret = RET_OK;
  if (!IsScalarClac() && !IsBatchScalarCalc() && !IsBiasCalc()) {
    ret = ConstTensorBroadCast(context);
  }
  return ret;
}

int ArithmeticFP32Coder::CheckDataType() {
  auto in0_dataType = input_tensor_->data_type();
  auto in1_dataType = filter_tensor_->data_type();
  if (in0_dataType != in1_dataType) {
    MS_LOG(ERROR) << "The dataTypes of input tensor0 and input tensor1 should be the same.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticFP32Coder::ChooseArithmeticFunc(bool is_opt) {
  if (input_tensor_->data_type() == kNumberTypeFloat32 || input_tensor_->data_type() == kNumberTypeFloat ||
      input_tensor_->data_type() == kNumberTypeFloat16) {
    if (is_opt) {
      arithmetic_func_str_ = wrap_void(arithmetic_opt_run_);
    } else {
      arithmetic_func_str_ = wrap_void(arithmetic_run_);
    }
  } else if (input_tensor_->data_type() == kNumberTypeBool) {
    arithmetic_func_str_ = wrap_void(arithmetic_run_bool_);
  } else {
    if (is_opt) {
      arithmetic_func_str_ = wrap_void(arithmetic_opt_run_int_);
    } else {
      arithmetic_func_str_ = wrap_void(arithmetic_run_int_);
    }
  }
}

bool ArithmeticFP32Coder::IsScalarClac() {
  return (arithmetic_parameter_->in_elements_num0_ == 1 || arithmetic_parameter_->in_elements_num1_ == 1) &&
         (!arithmetic_opt_run_.empty());
}

bool ArithmeticFP32Coder::IsBatchScalarCalc() {
  if (arithmetic_opt_run_.empty()) {
    return false;
  }
  size_t break_axis = 0;
  MS_CHECK_TRUE_RET(arithmetic_parameter_->ndim_ <= max_dims_, false);
  for (size_t i = 0; i < arithmetic_parameter_->ndim_; i++) {
    if (arithmetic_parameter_->in_shape0_[i] != arithmetic_parameter_->in_shape1_[i]) {
      break_axis = i;
      break;
    }
  }
  if (break_axis < arithmetic_parameter_->ndim_) {
    for (size_t i = break_axis; i < arithmetic_parameter_->ndim_; i++) {
      if (arithmetic_parameter_->in_shape1_[i] != 1) {
        return false;
      }
    }
  }
  break_pos_ = break_axis;
  return true;
}

bool ArithmeticFP32Coder::IsBiasCalc() {
  int last_shape0 = arithmetic_parameter_->in_shape0_[arithmetic_parameter_->ndim_ - 1];
  int last_shape1 = arithmetic_parameter_->in_shape1_[arithmetic_parameter_->ndim_ - 1];
  if (arithmetic_parameter_->in_elements_num0_ > arithmetic_parameter_->in_elements_num1_) {
    return arithmetic_parameter_->in_elements_num1_ == last_shape1 && last_shape0 == last_shape1;
  } else if (arithmetic_parameter_->in_elements_num0_ < arithmetic_parameter_->in_elements_num1_) {
    return arithmetic_parameter_->in_elements_num0_ == last_shape0 && last_shape0 == last_shape1;
  }
  return false;
}

void ArithmeticFP32Coder::FreeConstTileBuff() {
  if (input0_broadcast_ && input0_ptr_ != nullptr) {
    input0_ptr_ = nullptr;
    input0_broadcast_ = false;
  }
  if (input1_broadcast_ && input1_ptr_ != nullptr) {
    input1_ptr_ = nullptr;
    input0_broadcast_ = false;
  }
}

int ArithmeticFP32Coder::ConstTensorBroadCast(CoderContext *const context) {
  // if const node need broadcast and all need-broadcast-node are const, broadcast in resize
  if (!arithmetic_parameter_->broadcasting_) {
    return RET_OK;
  }
  if (output_tensor_->Size() < 0) {
    return RET_OK;
  }
  // need broadcast both input
  if (arithmetic_parameter_->in_elements_num0_ != arithmetic_parameter_->out_elements_num_ &&
      arithmetic_parameter_->in_elements_num1_ != arithmetic_parameter_->out_elements_num_) {
    return RET_OK;
  }
  FreeConstTileBuff();
  NNaclFp32Serializer init_code;
  Collect(context, {"wrapper/fp32/arithmetic_fp32_wrapper.h", "nnacl/fp32/arithmetic_fp32.h"},
          {"arithmetic_fp32_wrapper.c", "arithmetic_fp32.c"});
  if (input_tensor_->IsConst() &&
      arithmetic_parameter_->in_elements_num0_ != arithmetic_parameter_->out_elements_num_) {
    input0_ptr_ = reinterpret_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat32, arithmetic_parameter_->out_elements_num_ * data_type_len_, kWorkspace));
    MS_CHECK_PTR(input0_ptr_);
    init_code.CodeArray("in_shape", arithmetic_parameter_->in_shape0_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("in_stride", arithmetic_parameter_->in_strides0_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("out_stride", arithmetic_parameter_->out_strides_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("multiple", arithmetic_parameter_->multiples0_, arithmetic_parameter_->ndim_, true);
    init_code.CodeFunction("TileConstTensor", input_tensor_, input0_ptr_, arithmetic_parameter_->ndim_, "in_shape",
                           "in_stride", "out_stride", "multiple");
    input0_broadcast_ = true;
    arithmetic_parameter_->in_elements_num0_ = arithmetic_parameter_->out_elements_num_;
    arithmetic_parameter_->broadcasting_ = false;
  }
  if (filter_tensor_->IsConst() &&
      arithmetic_parameter_->in_elements_num1_ != arithmetic_parameter_->out_elements_num_) {
    input1_ptr_ = reinterpret_cast<float *>(
      allocator_->Malloc(kNumberTypeFloat32, arithmetic_parameter_->out_elements_num_ * data_type_len_, kWorkspace));
    MS_CHECK_PTR(input1_ptr_);
    init_code.CodeArray("in_shape", arithmetic_parameter_->in_shape1_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("in_stride", arithmetic_parameter_->in_strides1_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("out_stride", arithmetic_parameter_->out_strides_, arithmetic_parameter_->ndim_, true);
    init_code.CodeArray("multiple", arithmetic_parameter_->multiples1_, arithmetic_parameter_->ndim_, true);
    init_code.CodeFunction("TileConstTensor", filter_tensor_, input1_ptr_, arithmetic_parameter_->ndim_, "in_shape",
                           "in_stride", "out_stride", "multiple");
    input1_broadcast_ = true;
    arithmetic_parameter_->in_elements_num1_ = arithmetic_parameter_->out_elements_num_;
    arithmetic_parameter_->broadcasting_ = false;
  }
  return RET_OK;
}

int ArithmeticFP32Coder::Prepare(CoderContext *const context) {
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  MS_CHECK_RET_CODE(CheckDataType(), "ArithmeticFP32Coder check datatype fail");
  MS_CHECK_PTR(parameter_);
  arithmetic_parameter_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  max_dims_ = sizeof(arithmetic_parameter_->in_shape0_) / sizeof(int);
  auto primitive_type = arithmetic_parameter_->op_parameter_.type_;
  if (primitive_type == schema::PrimitiveType_Eltwise) {
    switch (arithmetic_parameter_->eltwise_mode_) {
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
        MS_LOG(ERROR) << "Eltwise mode not support, mode:" << arithmetic_parameter_->eltwise_mode_;
        return RET_ERROR;
    }
  }
  InitRunFunction(primitive_type);
  MS_CHECK_RET_CODE(ReSize(context), "do arithmetic ReSize fail!");
  return RET_OK;
}

void ArithmeticFP32Coder::CollectFilesForFunc(CoderContext *const context) {
  //  collect wrapper files
  Collect(context,
          {
            "wrapper/fp32/arithmetic_fp32_wrapper.h",
          },
          {
            "arithmetic_fp32_wrapper.c",
          });
  // for nnacl's operator combine all arithmetic to nnalc/arithmetic.c
  // this solution is not suitable for micro, for the size of package.
  if (arithmetic_opt_run_ == "ElementOptSub" || arithmetic_run_ == "ElementSub") {
    Collect(context,
            {
              "nnacl/fp32/sub_fp32.h",
            },
            {
              "sub_fp32.c",
            });
  } else if (arithmetic_opt_run_ == "ElementOptAdd" || arithmetic_run_ == "ElementAdd") {
    Collect(context,
            {
              "nnacl/fp32/add_fp32.h",
            },
            {
              "add_fp32.c",
              "arithmetic_fp32.c",
              "arithmetic_base.c",
            });
  } else if (arithmetic_opt_run_ == "ElementOptMul" || arithmetic_run_ == "ElementMul") {
    Collect(context,
            {
              "nnacl/fp32/mul_fp32.h",
            },
            {
              "mul_fp32.c",
            });
  } else if (arithmetic_run_ == "ElementAddRelu") {
    Collect(context,
            {
              "nnacl/fp32/add_fp32.h",
            },
            {
              "add_fp32.c",
            });
  } else if (arithmetic_run_ == "ElementDivRelu6" || arithmetic_run_ == "ElementDivRelu" ||
             arithmetic_run_ == "ElementDiv") {
    Collect(context,
            {
              "nnacl/fp32/div_fp32.h",
            },
            {
              "div_fp32.c",
            });
  } else {
    Collect(context,
            {
              "nnacl/fp32/arithmetic_fp32.h",
            },
            {
              "arithmetic_fp32.c",
            });
  }
}

int ArithmeticFP32Coder::ExecuteCode(const std::string &input0, const std::string &input1, const std::string &output,
                                     int size, bool is_opt, CoderContext *const context,
                                     NNaclFp32Serializer *const code) {
  if (arithmetic_func_str_.empty()) {
    return RET_ERROR;
  }
  code->CodeStruct("arithmetic_parameter", *arithmetic_parameter_);
  if (support_parallel_) {
    code->CodeBaseStruct("ArithmeticFp32Args", kRunArgs, input0, input1, output, size, is_opt, arithmetic_func_type_,
                         arithmetic_func_str_, "&arithmetic_parameter", gThreadNum);
    code->CodeFunction(kParallelLaunch, "ArithmeticFp32Run", kRunArgsAddr, gThreadNum);
  } else {
    code->CodeFunction("ArithmeticExecute", input0, input1, output, size, is_opt, arithmetic_func_type_,
                       arithmetic_func_str_, "&arithmetic_parameter");
  }
  context->AppendCode(code->str());
  return RET_OK;
}

int ArithmeticFP32Coder::BatchScalarCalc(int task_id, CoderContext *const context, NNaclFp32Serializer *const code) {
  if (break_pos_ < 1) {
    return RET_ERROR;
  }
  if (support_parallel_) {
    thread_num_ = 1;
  }
  int batch = arithmetic_parameter_->out_elements_num_ / arithmetic_parameter_->out_strides_[break_pos_ - 1];
  int batch_per_thread = UP_DIV(batch, thread_num_);

  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, batch);
  int batch_size = end_batch - start_batch;

  int stride0 = arithmetic_parameter_->in_strides0_[break_pos_ - 1] * data_type_len_;
  int stride1 = arithmetic_parameter_->in_strides1_[break_pos_ - 1] * data_type_len_;
  int out_stride = arithmetic_parameter_->out_strides_[break_pos_ - 1] * data_type_len_;

  int offset0 = stride0 * start_batch;
  int offset1 = stride1 * start_batch;
  int out_offset = out_stride * start_batch;

  arithmetic_wrapper_info_ = {offset0, stride0, offset1, stride1, out_offset, out_stride, arithmetic_func_type_};
  code->CodeStruct("arithmetic_wrapper_info", arithmetic_wrapper_info_);
  std::string param_name = "arithmetic_parameter";
  code->CodeStruct(param_name, *arithmetic_parameter_);
  if (support_parallel_) {
    *code << "    " << param_name << ".op_parameter_.thread_num_ = 1;\n";
  }
  code->CodeFunction("BatchScalarCalc", wrap_uint8(input0_ptr_str_), wrap_uint8(input1_ptr_str_),
                     wrap_uint8(output_ptr_str_), batch_size, arithmetic_parameter_->out_strides_[break_pos_ - 1], true,
                     arithmetic_func_str_, "&arithmetic_wrapper_info", "&arithmetic_parameter");
  context->AppendCode(code->str());
  return RET_OK;
}

int ArithmeticFP32Coder::BiasCalc(int task_id, CoderContext *const context, NNaclFp32Serializer *const code) {
  MS_CHECK_TRUE_RET(arithmetic_parameter_->ndim_ - 1 >= 0 && arithmetic_parameter_->ndim_ - 1 < 10, RET_ERROR);
  if (support_parallel_) {
    thread_num_ = 1;
  }
  int last_shape = arithmetic_parameter_->out_shape_[arithmetic_parameter_->ndim_ - 1];
  int batch = arithmetic_parameter_->out_elements_num_ / last_shape;
  int batch_per_thread = UP_DIV(batch, thread_num_);

  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, batch);
  int batch_size = end_batch - start_batch;

  int stride = last_shape * data_type_len_;
  int offset = stride * start_batch;
  std::string param_name = "arithmetic_parameter";
  code->CodeStruct(param_name, *arithmetic_parameter_);
  if (support_parallel_) {
    *code << "    " << param_name << ".op_parameter_.thread_num_ = 1;\n";
  }
  if (arithmetic_parameter_->in_elements_num0_ > arithmetic_parameter_->in_elements_num1_) {
    arithmetic_wrapper_info_ = {offset, stride, 0, 0, offset, stride, arithmetic_func_type_};
    code->CodeStruct("arithmetic_wrapper_info", arithmetic_wrapper_info_);
    code->CodeFunction("BatchScalarCalc", wrap_uint8(input0_ptr_str_), wrap_uint8(input1_ptr_str_),
                       wrap_uint8(output_ptr_str_), batch_size, last_shape, false, arithmetic_func_str_,
                       "&arithmetic_wrapper_info", "&arithmetic_parameter");
  } else {
    arithmetic_wrapper_info_ = {0, 0, offset, stride, offset, stride, arithmetic_func_type_};
    code->CodeStruct("arithmetic_wrapper_info", arithmetic_wrapper_info_);
    code->CodeFunction("BatchScalarCalc", wrap_uint8(input0_ptr_str_), wrap_uint8(input1_ptr_str_),
                       wrap_uint8(output_ptr_str_), batch_size, last_shape, false, arithmetic_func_str_,
                       "&arithmetic_wrapper_info", "&arithmetic_parameter");
  }
  context->AppendCode(code->str());
  return RET_OK;
}

int ArithmeticFP32Coder::BroadcastRun(const std::string &input0, const std::string &input1, const std::string &output,
                                      int dim, int out_count, int out_thread_stride, CoderContext *const context,
                                      NNaclFp32Serializer *const code) {
  code->CodeStruct("arithmetic_parameter", *arithmetic_parameter_);
  code->CodeFunction("BroadcastRun", wrap_uint8(input0_ptr_str_), wrap_uint8(input1_ptr_str_),
                     wrap_uint8(output_ptr_str_), dim, out_count, out_thread_stride, break_pos_, data_type_len_,
                     arithmetic_func_type_, arithmetic_func_str_, "&arithmetic_parameter");
  context->AppendCode(code->str());
  return RET_OK;
}

int ArithmeticFP32Coder::DoCode(CoderContext *const context) {
  int element_num = output_tensor_->ElementsNum();
  int stride = UP_DIV(element_num, thread_num_);
  int count = element_num;
  if (count <= 0) {
    return RET_OK;
  }
  int offset = stride * kDefaultTaskId * data_type_len_;
  input0_ptr_str_ = allocator_->GetRuntimeAddr(input0_ptr_);
  input1_ptr_str_ = allocator_->GetRuntimeAddr(input1_ptr_);
  if (!input0_broadcast_) {
    input0_ptr_str_ = allocator_->GetRuntimeAddr(input_tensor_, true);
    input1_ptr_str_ = allocator_->GetRuntimeAddr(filter_tensor_);
  }
  if (!input1_broadcast_) {
    input0_ptr_str_ = allocator_->GetRuntimeAddr(input_tensor_);
    input1_ptr_str_ = allocator_->GetRuntimeAddr(filter_tensor_, true);
  }
  output_ptr_str_ = allocator_->GetRuntimeAddr(output_tensor_);
  NNaclFp32Serializer code;
  CollectFilesForFunc(context);
  if (IsScalarClac()) {
    ChooseArithmeticFunc(true);
    if (arithmetic_parameter_->in_elements_num0_ == 1) {
      return ExecuteCode(wrap_uint8(input0_ptr_str_), wrap_offset(wrap_uint8(input1_ptr_str_), offset),
                         wrap_offset(wrap_uint8(output_ptr_str_), offset), count, true, context, &code);
    } else if (arithmetic_parameter_->in_elements_num1_ == 1) {
      return ExecuteCode(wrap_offset(wrap_uint8(input0_ptr_str_), offset), wrap_void(input1_ptr_str_),
                         wrap_offset(wrap_uint8(output_ptr_str_), offset), count, true, context, &code);
    }
  }

  // run opt function, every batch one of input is scalar
  if (IsBatchScalarCalc()) {
    ChooseArithmeticFunc(true);
    return BatchScalarCalc(kDefaultTaskId, context, &code);
  }

  // each batch is eltwise calculation
  if (IsBiasCalc()) {
    ChooseArithmeticFunc(false);
    return BiasCalc(kDefaultTaskId, context, &code);
  }

  // need broadcast in runtime
  if (arithmetic_parameter_->broadcasting_) {
    ChooseArithmeticFunc(false);
    stride = UP_DIV(outside_, thread_num_);
    int out_count = MSMIN(stride, outside_ - stride * kDefaultTaskId);
    if (out_count <= 0) {
      return RET_OK;
    }
    return BroadcastRun(input0_ptr_str_, input1_ptr_str_, output_ptr_str_, 0, out_count, stride * kDefaultTaskId,
                        context, &code);
  }
  // all elements eltwise calculation
  ChooseArithmeticFunc(false);
  return ExecuteCode(wrap_offset(wrap_uint8(input0_ptr_str_), offset), wrap_offset(wrap_uint8(input1_ptr_str_), offset),
                     wrap_offset(wrap_uint8(output_ptr_str_), offset), count, false, context, &code);
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_MulFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AddFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SubFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_DivFusion, CPUOpCoderCreator<ArithmeticFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_RealDiv, CPUOpCoderCreator<ArithmeticFP32Coder>)

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
