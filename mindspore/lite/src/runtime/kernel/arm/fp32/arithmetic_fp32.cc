/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp32/arithmetic_fp32.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {
int ArithmeticCPUKernel::Init() {
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
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCPUKernel::ReSize() {
  CalcMultiplesAndStrides(param_);
  if (param_->broadcasting_) {
    outside_ = 1;
    for (auto i = param_->ndim_ - 1; i >= 0; --i) {
      if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= param_->out_shape_[i];
    }
  }
  data_type_len_ = lite::DataTypeSize(in_tensors_.at(0)->data_type());
  int ret = RET_OK;
  if (!IsScalarClac() && !IsBatchScalarCalc() && !IsBiasCalc()) {
    ret = ConstTensorBroadCast();
  }
  return ret;
}

int ArithmeticCPUKernel::CheckDataType() {
  auto in0_dataType = in_tensors_.at(0)->data_type();
  auto in1_dataType = in_tensors_.at(1)->data_type();
  if (in0_dataType != in1_dataType) {
    MS_LOG(ERROR) << "The dataTypes of input tensor0 and input tensor1 should be the same.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool ArithmeticCPUKernel::IsScalarClac() {  // 2 32 240 240, 1 1 1 1
  if ((param_->in_elements_num0_ == 1 || param_->in_elements_num1_ == 1) && (arithmetic_opt_run_ != nullptr)) {
    return true;
  } else {
    return false;
  }
}

bool ArithmeticCPUKernel::IsBatchScalarCalc() {  // 2 32 240 240,  2 32 1 1
  if (arithmetic_opt_run_ == nullptr) {
    return false;
  }
  size_t break_axis = 0;
  for (size_t i = 0; i < param_->ndim_; i++) {
    if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
      break_axis = i;
      break;
    }
  }
  if (break_axis < param_->ndim_) {
    for (size_t i = break_axis; i < param_->ndim_; i++) {
      if (param_->in_shape1_[i] != 1) {
        return false;
      }
    }
  }
  break_pos_ = break_axis;
  return true;
}

bool ArithmeticCPUKernel::IsBiasCalc() {  // 2 240 240 32,    1 1 1 32
  int last_shape0 = param_->in_shape0_[param_->ndim_ - 1];
  int last_shape1 = param_->in_shape1_[param_->ndim_ - 1];
  if (param_->in_elements_num0_ > param_->in_elements_num1_) {
    return param_->in_elements_num1_ == last_shape1 && last_shape0 == last_shape1;
  } else if (param_->in_elements_num0_ < param_->in_elements_num1_) {
    return param_->in_elements_num0_ == last_shape0 && last_shape0 == last_shape1;
  }
  return false;
}

int ArithmeticCPUKernel::ConstTensorBroadCast() {
  /* if const node need broadcast and all need-broadcast-node are const, broadcast in resize */
  if (!param_->broadcasting_) {
    return RET_OK;
  }
  if (out_tensors_[0]->Size() < 0) {
    return RET_OK;
  }
  /* [1, 1, 2] + [1, 2, 1] -> [1, 2, 2], need broadcast both input */
  if (param_->in_elements_num0_ != param_->out_elements_num_ &&
      param_->in_elements_num1_ != param_->out_elements_num_) {
    return RET_OK;
  }

  FreeConstTileBuff();
  if (in_tensors_[0]->IsConst() && param_->in_elements_num0_ != param_->out_elements_num_) {
    input0_ptr_ = malloc(param_->out_elements_num_ * data_type_len_);
    if (input0_ptr_ == nullptr) {
      return RET_ERROR;
    }
    TileConstTensor(in_tensors_[0]->data_c(), input0_ptr_, param_->ndim_, param_->in_shape0_, param_->in_strides0_,
                    param_->out_strides_, param_->multiples0_);
    input0_broadcast_ = true;
    param_->in_elements_num0_ = param_->out_elements_num_;
    param_->broadcasting_ = false;
  }
  if (in_tensors_[1]->IsConst() && param_->in_elements_num1_ != param_->out_elements_num_) {
    input1_ptr_ = malloc(param_->out_elements_num_ * data_type_len_);
    if (input1_ptr_ == nullptr) {
      FreeConstTileBuff();
      return RET_ERROR;
    }
    TileConstTensor(in_tensors_[1]->data_c(), input1_ptr_, param_->ndim_, param_->in_shape1_, param_->in_strides1_,
                    param_->out_strides_, param_->multiples1_);
    input1_broadcast_ = true;
    param_->in_elements_num1_ = param_->out_elements_num_;
    param_->broadcasting_ = false;
  }
  return RET_OK;
}

void ArithmeticCPUKernel::TileConstTensor(const void *in_data, void *out_data, size_t ndim, const int *in_shape,
                                          const int *in_strides, const int *out_strides, const int *multiple) {
  TileOneDimensionFp32(reinterpret_cast<const float *>(in_data), reinterpret_cast<float *>(out_data), 0, ndim, in_shape,
                       in_strides, out_strides, multiple);
}

void ArithmeticCPUKernel::FreeConstTileBuff() {
  if (input0_broadcast_ == true && input0_ptr_ != nullptr) {
    free(input0_ptr_);
    input0_ptr_ = nullptr;
    input0_broadcast_ = false;
  }
  if (input1_broadcast_ == true && input1_ptr_ != nullptr) {
    free(input1_ptr_);
    input1_ptr_ = nullptr;
    input0_broadcast_ = false;
  }
  return;
}

void ArithmeticCPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, ElementMulRelu, ElementMulReluInt, nullptr,
     ElementOptMulRelu, ElementOptMulReluInt},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, ElementMulRelu6, ElementMulRelu6Int, nullptr,
     ElementOptMulRelu6, ElementOptMulRelu6Int},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, ElementMul, ElementMulInt, nullptr, ElementOptMul,
     ElementOptMulInt},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, ElementAddRelu, nullptr, nullptr, ElementOptAddRelu,
     nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, ElementAddRelu6, nullptr, nullptr, ElementOptAddRelu6,
     nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, ElementAdd, ElementAddInt, nullptr, ElementOptAdd,
     ElementOptAddInt},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, ElementSubRelu, nullptr, nullptr, ElementOptSubRelu,
     nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, ElementSubRelu6, nullptr, nullptr, ElementOptSubRelu6,
     nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, ElementSub, ElementSubInt, nullptr, ElementOptSub,
     ElementOptSubInt},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu,
     nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAnd, ElementLogicalAndInt,
     ElementLogicalAndBool, nullptr, nullptr},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOr, nullptr, ElementLogicalOrBool,
     nullptr, nullptr},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, ElementMaximum, ElementMaximumInt, nullptr, nullptr,
     nullptr},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, ElementMinimum, ElementMinimumInt, nullptr, nullptr,
     nullptr},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, ElementFloorMod, ElementFloorModInt, nullptr,
     nullptr, nullptr},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, ElementFloorDiv, ElementFloorDivInt, nullptr,
     nullptr, nullptr},
    {PrimitiveType_Mod, schema::ActivationType_NO_ACTIVATION, ElementMod, ElementModInt, nullptr, ElementOptMod,
     ElementOptModInt},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, ElementSquaredDifference, nullptr, nullptr,
     nullptr, nullptr}};

  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_FUNC_INFO_FP32);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type && fun_table[i].activation_type_ == param_->activation_type_) {
      arithmetic_run_ = fun_table[i].func_;
      arithmetic_run_int_ = fun_table[i].int_func_;
      arithmetic_run_bool_ = fun_table[i].bool_func_;
      arithmetic_opt_run_ = fun_table[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table[i].opt_int_func_;
      return;
    }
  }
}

int ArithmeticCPUKernel::Execute(const void *input0, const void *input1, void *output, int size, bool is_opt) {
  int ret = RET_OK;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (is_opt) {
      CHECK_NULL_RETURN(arithmetic_opt_run_, RET_ERROR);
      ret = arithmetic_opt_run_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                                reinterpret_cast<float *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_, RET_ERROR);
      ret = arithmetic_run_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                            reinterpret_cast<float *>(output), size);
    }
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    CHECK_NULL_RETURN(arithmetic_run_bool_, RET_ERROR);
    ret = arithmetic_run_bool_(reinterpret_cast<const bool *>(input0), reinterpret_cast<const bool *>(input1),
                               reinterpret_cast<bool *>(output), size);
  } else {
    if (is_opt) {
      CHECK_NULL_RETURN(arithmetic_opt_run_int_, RET_ERROR);
      ret = arithmetic_opt_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                    reinterpret_cast<int *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_int_, RET_ERROR);
      ret = arithmetic_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                reinterpret_cast<int *>(output), size);
    }
  }
  return ret;
}

int ArithmeticCPUKernel::BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count,
                                      int out_thread_stride) {
  if (dim > break_pos_) {
    int offset = out_thread_stride * data_type_len_;
    return Execute(static_cast<uint8_t *>(input0) + offset, static_cast<uint8_t *>(input1) + offset,
                   static_cast<uint8_t *>(output) + offset, out_count, false);
  }
  int offset_size[] = {param_->in_strides0_[dim] * data_type_len_, param_->in_strides1_[dim] * data_type_len_,
                       param_->out_strides_[dim] * data_type_len_};
  for (int i = 0; i < param_->out_shape_[dim]; ++i) {
    int pos0_ = param_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = param_->in_shape1_[dim] == 1 ? 0 : i;
    int ret = BroadcastRun(static_cast<uint8_t *>(input0) + pos0_ * offset_size[0],
                           static_cast<uint8_t *>(input1) + pos1_ * offset_size[1],
                           static_cast<uint8_t *>(output) + i * offset_size[2], dim + 1, out_count, out_thread_stride);
    if (ret != RET_OK) {
      return ret;
    }
  }
  return RET_OK;
}

int ArithmeticCPUKernel::BatchScalarCalc(int task_id) {
  if (break_pos_ < 1) {
    return RET_ERROR;
  }
  int batch = param_->out_elements_num_ / param_->out_strides_[break_pos_ - 1];
  int batch_per_thread = UP_DIV(batch, context_->thread_num_);

  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, batch);
  int batch_size = end_batch - start_batch;

  int stride0 = param_->in_strides0_[break_pos_ - 1] * data_type_len_;
  int stride1 = param_->in_strides1_[break_pos_ - 1] * data_type_len_;
  int out_stride = param_->out_strides_[break_pos_ - 1] * data_type_len_;

  int offset0 = stride0 * start_batch;
  int offset1 = stride1 * start_batch;
  int out_offset = out_stride * start_batch;

  int ret = RET_OK;
  for (int i = 0; i < batch_size; i++) {
    ret = Execute(static_cast<uint8_t *>(input0_ptr_) + offset0, static_cast<uint8_t *>(input1_ptr_) + offset1,
                  static_cast<uint8_t *>(output_ptr_) + out_offset, param_->out_strides_[break_pos_ - 1], true);
    offset0 += stride0;
    offset1 += stride1;
    out_offset += out_stride;
  }
  return ret;
}

int ArithmeticCPUKernel::BiasCalc(int task_id) {
  int last_shape = param_->out_shape_[param_->ndim_ - 1];
  int batch = param_->out_elements_num_ / last_shape;
  int batch_per_thread = UP_DIV(batch, context_->thread_num_);

  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, batch);
  int batch_size = end_batch - start_batch;

  int stride = last_shape * data_type_len_;
  int offset = stride * start_batch;
  int ret = RET_OK;
  if (param_->in_elements_num0_ > param_->in_elements_num1_) {
    for (int i = 0; i < batch_size; i++) {
      ret = Execute(static_cast<uint8_t *>(input0_ptr_) + offset, static_cast<uint8_t *>(input1_ptr_),
                    static_cast<uint8_t *>(output_ptr_) + offset, last_shape, false);
      if (ret != RET_OK) {
        return ret;
      }
      offset += stride;
    }
  } else {
    for (int i = 0; i < batch_size; i++) {
      ret = Execute(static_cast<uint8_t *>(input0_ptr_), static_cast<uint8_t *>(input1_ptr_) + offset,
                    static_cast<uint8_t *>(output_ptr_) + offset, last_shape, false);
      if (ret != RET_OK) {
        return ret;
      }
      offset += stride;
    }
  }
  return ret;
}

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  auto element_num = out_tensors_[0]->ElementsNum();
  int stride = UP_DIV(element_num, context_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  int offset = stride * task_id * data_type_len_;
  /* run opt function, one of input is scalar */
  if (IsScalarClac()) {  // 2 32 240 240, 1 1 1 1
    if (param_->in_elements_num0_ == 1) {
      return Execute(input0_ptr_, static_cast<uint8_t *>(input1_ptr_) + offset,
                     static_cast<uint8_t *>(output_ptr_) + offset, count, true);
    } else if (param_->in_elements_num1_ == 1) {
      return Execute(static_cast<uint8_t *>(input0_ptr_) + offset, input1_ptr_,
                     static_cast<uint8_t *>(output_ptr_) + offset, count, true);
    }
  }
  /* run opt function, every batch one of input is scalar */
  if (IsBatchScalarCalc()) {  // 2 32 240 240,  2 32 1 1
    return BatchScalarCalc(task_id);
  }
  /* each batch is eltwise calculation */
  if (IsBiasCalc()) {  // 2 240 240 32,    1 1 1 32
    return BiasCalc(task_id);
  }
  /* need broadcast in runtime */
  if (param_->broadcasting_) {
    stride = UP_DIV(outside_, context_->thread_num_);
    int out_count = MSMIN(stride, outside_ - stride * task_id);
    if (out_count <= 0) {
      return RET_OK;
    }
    return BroadcastRun(input0_ptr_, input1_ptr_, output_ptr_, 0, out_count, stride * task_id);
  }
  /* all elements eltwise calculation */
  return Execute(static_cast<uint8_t *>(input0_ptr_) + offset, static_cast<uint8_t *>(input1_ptr_) + offset,
                 static_cast<uint8_t *>(output_ptr_) + offset, count, false);
}

int ArithmeticsRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<ArithmeticCPUKernel *>(cdata);
  auto ret = kernel->DoArithmetic(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int ArithmeticCPUKernel::Run() {
  if (CheckDataType() != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticCPUKernel check dataType failed.";
    return RET_ERROR;
  }
  if (!input0_broadcast_) {
    input0_ptr_ = in_tensors_[0]->data_c();
  }
  if (!input1_broadcast_) {
    input1_ptr_ = in_tensors_[1]->data_c();
  }
  output_ptr_ = out_tensors_[0]->data_c();
  return ParallelLaunch(this->context_->thread_pool_, ArithmeticsRun, this, context_->thread_num_);
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RealDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Eltwise, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
}  // namespace mindspore::kernel
