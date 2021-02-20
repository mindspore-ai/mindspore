/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/int8/add_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/ops/arithmetic.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {
ArithmeticCPUKernel::~ArithmeticCPUKernel() {
  FreeTmpPtr();
  return;
}

int ArithmeticCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCPUKernel::InitBroadCastCase() {
  /* if const node need broadcast
   * and all need-broadcast-node are const
   * broadcast in resize */

  if (!arithmeticParameter_->broadcasting_) {
    return RET_OK;
  }

  if (out_tensors_[0]->Size() < 0) {
    return RET_OK;
  }

  if (arithmeticParameter_->in_elements_num0_ != arithmeticParameter_->out_elements_num_ &&
      arithmeticParameter_->in_elements_num1_ != arithmeticParameter_->out_elements_num_) {
    /* [1, 1, 2] + [1, 2, 1] -> [1, 2, 2]
     * need broadcast both input */
    return RET_OK;
  }

  if ((arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) &&
      (arithmetic_opt_run_ != nullptr && arithmetic_opt_run_int_ != nullptr)) {
    /* run opt function
     * one of input is scalar */
    return RET_OK;
  }

  FreeTmpPtr();

  CalcMultiplesAndStrides(arithmeticParameter_);

  if (in_tensors_[0]->data_c() != nullptr &&
      arithmeticParameter_->in_elements_num1_ == arithmeticParameter_->out_elements_num_) {
    input0_ptr_ = malloc(arithmeticParameter_->out_elements_num_ * sizeof(float));
    if (input0_ptr_ == nullptr) {
      return RET_ERROR;
    }
    TileOneDimensionFp32(reinterpret_cast<float *>(in_tensors_[0]->data_c()), reinterpret_cast<float *>(input0_ptr_), 0,
                         arithmeticParameter_->ndim_, arithmeticParameter_->in_shape0_,
                         arithmeticParameter_->in_strides0_, arithmeticParameter_->out_strides_,
                         arithmeticParameter_->multiples0_);
    arithmeticParameter_->broadcasting_ = false;
    input0_broadcast_ = true;
  }
  if (in_tensors_[1]->data_c() != nullptr &&
      arithmeticParameter_->in_elements_num0_ == arithmeticParameter_->out_elements_num_) {
    input1_ptr_ = malloc(arithmeticParameter_->out_elements_num_ * sizeof(float));
    if (input1_ptr_ == nullptr) {
      FreeTmpPtr();
      return RET_ERROR;
    }
    TileOneDimensionFp32(reinterpret_cast<float *>(in_tensors_[1]->data_c()), reinterpret_cast<float *>(input1_ptr_), 0,
                         arithmeticParameter_->ndim_, arithmeticParameter_->in_shape1_,
                         arithmeticParameter_->in_strides1_, arithmeticParameter_->out_strides_,
                         arithmeticParameter_->multiples1_);
    arithmeticParameter_->broadcasting_ = false;
    input1_broadcast_ = true;
  }
  return RET_OK;
}

void ArithmeticCPUKernel::InitRunFunction() {
  ARITHMETIC_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_Mul, schema::ActivationType_RELU, ElementMulRelu, ElementMulReluInt, nullptr, ElementOptMulRelu,
     ElementOptMulReluInt},
    {PrimitiveType_Mul, schema::ActivationType_RELU6, ElementMulRelu6, ElementMulRelu6Int, nullptr, ElementOptMulRelu6,
     ElementOptMulRelu6Int},
    {PrimitiveType_Mul, schema::ActivationType_NO_ACTIVATION, ElementMul, ElementMulInt, nullptr, ElementOptMul,
     ElementOptMulInt},
    {PrimitiveType_Add, schema::ActivationType_RELU, ElementAddRelu, nullptr, nullptr, ElementOptAddRelu, nullptr},
    {PrimitiveType_Add, schema::ActivationType_RELU6, ElementAddRelu6, nullptr, nullptr, ElementOptAddRelu6, nullptr},
    {PrimitiveType_Add, schema::ActivationType_NO_ACTIVATION, ElementAdd, ElementAddInt, nullptr, ElementOptAdd,
     ElementOptAddInt},
    {PrimitiveType_Sub, schema::ActivationType_RELU, ElementSubRelu, nullptr, nullptr, ElementOptSubRelu, nullptr},
    {PrimitiveType_Sub, schema::ActivationType_RELU6, ElementSubRelu6, nullptr, nullptr, ElementOptSubRelu6, nullptr},
    {PrimitiveType_Sub, schema::ActivationType_NO_ACTIVATION, ElementSub, ElementSubInt, nullptr, ElementOptSub,
     ElementOptSubInt},
    {PrimitiveType_Div, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr},
    {PrimitiveType_Div, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6, nullptr},
    {PrimitiveType_Div, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAnd, ElementLogicalAndInt,
     ElementLogicalAndBool, nullptr, nullptr},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOr, nullptr, nullptr, nullptr,
     nullptr},
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
    if (fun_table[i].primitive_type_ == op_parameter_->type_ &&
        fun_table[i].activation_type_ == arithmeticParameter_->activation_type_) {
      arithmetic_run_ = fun_table[i].func_;
      arithmetic_run_int_ = fun_table[i].int_func_;
      arithmetic_run_bool_ = fun_table[i].bool_func_;
      arithmetic_opt_run_ = fun_table[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table[i].opt_int_func_;
      return;
    }
  }
}

void ArithmeticCPUKernel::InitParam() {
  auto arithmetic_lite_primitive = (lite::Arithmetic *)primitive_;
  arithmeticParameter_->broadcasting_ = arithmetic_lite_primitive->Broadcasting();
  arithmeticParameter_->ndim_ = arithmetic_lite_primitive->NDims();
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32 || in_tensors_[0]->data_type() == kNumberTypeFloat16) {
    data_type_ = kDataTypeFloat;
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    data_type_ = KDataTypeBool;
  } else {
    data_type_ = kDataTypeInt;
  }

  arithmeticParameter_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arithmeticParameter_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arithmeticParameter_->out_elements_num_ = out_tensors_[0]->ElementsNum();
  memcpy(arithmeticParameter_->in_shape0_, reinterpret_cast<const lite::Arithmetic *>(primitive_)->InShape0().data(),
         reinterpret_cast<const lite::Arithmetic *>(primitive_)->InShape0().size() * sizeof(int));
  memcpy(arithmeticParameter_->in_shape1_, reinterpret_cast<const lite::Arithmetic *>(primitive_)->InShape1().data(),
         reinterpret_cast<const lite::Arithmetic *>(primitive_)->InShape1().size() * sizeof(int));
  memcpy(arithmeticParameter_->out_shape_, reinterpret_cast<const lite::Arithmetic *>(primitive_)->OutputShape().data(),
         reinterpret_cast<const lite::Arithmetic *>(primitive_)->OutputShape().size() * sizeof(int));

  return;
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

int ArithmeticCPUKernel::ReSize() {
  if (CheckDataType() != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticCPUKernel resize failed.";
    return RET_ERROR;
  }
  InitParam();
  return InitBroadCastCase();
}

int ArithmeticCPUKernel::BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count,
                                      int out_thread_stride) {
  if (dim > break_pos_) {
    if (data_type_ == kDataTypeInt) {
      return arithmetic_run_int_(reinterpret_cast<int *>(input0) + out_thread_stride,
                                 reinterpret_cast<int *>(input1) + out_thread_stride,
                                 reinterpret_cast<int *>(output) + out_thread_stride, out_count);
    }
    return arithmetic_run_(reinterpret_cast<float *>(input0) + out_thread_stride,
                           reinterpret_cast<float *>(input1) + out_thread_stride,
                           reinterpret_cast<float *>(output) + out_thread_stride, out_count);
  }
  for (int i = 0; i < arithmeticParameter_->out_shape_[dim]; ++i) {
    int pos0_ = arithmeticParameter_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = arithmeticParameter_->in_shape1_[dim] == 1 ? 0 : i;
    int error_code;
    if (data_type_ == kDataTypeInt) {
      error_code = BroadcastRun(reinterpret_cast<int *>(input0) + pos0_ * arithmeticParameter_->in_strides0_[dim],
                                reinterpret_cast<int *>(input1) + pos1_ * arithmeticParameter_->in_strides1_[dim],
                                reinterpret_cast<int *>(output) + i * arithmeticParameter_->out_strides_[dim], dim + 1,
                                out_count, out_thread_stride);
    } else {
      error_code = BroadcastRun(reinterpret_cast<float *>(input0) + pos0_ * arithmeticParameter_->in_strides0_[dim],
                                reinterpret_cast<float *>(input1) + pos1_ * arithmeticParameter_->in_strides1_[dim],
                                reinterpret_cast<float *>(output) + i * arithmeticParameter_->out_strides_[dim],
                                dim + 1, out_count, out_thread_stride);
    }
    if (error_code != RET_OK) {
      return error_code;
    }
  }
  return RET_OK;
}

bool ArithmeticCPUKernel::CanBatchScalar() {  // 2 32 240 240,  2 32 1 1
  if (input0_broadcast_ || input1_broadcast_) {
    return false;
  }
  if (arithmeticParameter_->in_elements_num0_ == arithmeticParameter_->in_elements_num1_ ||
      arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) {
    return false;
  }
  size_t break_axis = 0;
  for (size_t i = 0; i < arithmeticParameter_->ndim_; i++) {
    if (arithmeticParameter_->in_shape0_[i] != arithmeticParameter_->in_shape1_[i]) {
      break_axis = i;
      break;
    }
  }
  if (break_axis < arithmeticParameter_->ndim_) {
    for (size_t i = break_axis; i < arithmeticParameter_->ndim_; i++) {
      if (arithmeticParameter_->in_shape1_[i] != 1) {
        return false;
      }
    }
  }
  break_pos_ = break_axis;
  return true;
}

int ArithmeticCPUKernel::BatchScalarCalc(int task_id) {
  int batch = arithmeticParameter_->out_elements_num_ / arithmeticParameter_->out_strides_[break_pos_ - 1];
  int batch_per_thread = UP_DIV(batch, thread_count_);

  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, batch);
  int batch_size = end_batch - start_batch;

  int stride0 = arithmeticParameter_->in_strides0_[break_pos_ - 1];
  int stride1 = arithmeticParameter_->in_strides1_[break_pos_ - 1];
  int out_stride = arithmeticParameter_->out_strides_[break_pos_ - 1];

  int offset0 = stride0 * start_batch;
  int offset1 = stride1 * start_batch;
  int out_offset = out_stride * start_batch;

  int ret = RET_OK;
  for (int i = 0; i < batch_size; i++) {
    if (data_type_ == kDataTypeFloat) {
      ret = arithmetic_opt_run_(
        reinterpret_cast<float *>(input0_ptr_) + offset0, reinterpret_cast<float *>(input1_ptr_) + offset1,
        reinterpret_cast<float *>(out_tensors_[0]->data_c()) + out_offset, out_stride, arithmeticParameter_);
    } else {
      ret = arithmetic_opt_run_int_(
        reinterpret_cast<int *>(input0_ptr_) + offset0, reinterpret_cast<int *>(input1_ptr_) + offset1,
        reinterpret_cast<int *>(out_tensors_[0]->data_c()) + out_offset, out_stride, arithmeticParameter_);
    }
    offset0 += stride0;
    offset1 += stride1;
    out_offset += out_stride;
  }
  return ret;
}

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(thread_count_ != 0);
  int stride = UP_DIV(element_num, thread_count_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  if (arithmetic_run_ == nullptr) {
    MS_LOG(ERROR) << "arithmetic_run function is nullptr!";
    return RET_ERROR;
  }
  if (CanBatchScalar()) {
    return BatchScalarCalc(task_id);
  }
  int error_code = RET_OK;
  if ((arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) &&
      (arithmetic_opt_run_ != nullptr && arithmetic_opt_run_int_ != nullptr)) {
    /* run opt function
     * one of input is scalar */
    if (arithmeticParameter_->in_elements_num0_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        error_code = arithmetic_opt_run_(
          reinterpret_cast<float *>(input0_ptr_), reinterpret_cast<float *>(input1_ptr_) + stride * task_id,
          reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count, arithmeticParameter_);
      } else {
        error_code = arithmetic_opt_run_int_(
          reinterpret_cast<int *>(input0_ptr_), reinterpret_cast<int *>(input1_ptr_) + stride * task_id,
          reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id, count, arithmeticParameter_);
      }
    } else if (arithmeticParameter_->in_elements_num1_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        error_code = arithmetic_opt_run_(
          reinterpret_cast<float *>(input0_ptr_) + stride * task_id, reinterpret_cast<float *>(input1_ptr_),
          reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count, arithmeticParameter_);
      } else {
        error_code = arithmetic_opt_run_int_(
          reinterpret_cast<int *>(input0_ptr_) + stride * task_id, reinterpret_cast<int *>(input1_ptr_),
          reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id, count, arithmeticParameter_);
      }
    }
    return error_code;
  }
  if (arithmeticParameter_->broadcasting_) {
    /* need broadcast in runtime */
    stride = UP_DIV(outside_, thread_count_);
    int out_count = MSMIN(stride, outside_ - stride * task_id);
    if (out_count <= 0) {
      return RET_OK;
    }
    int out_thread_stride = stride * task_id;
    if (data_type_ == kDataTypeFloat) {
      error_code = BroadcastRun(reinterpret_cast<float *>(input0_ptr_), reinterpret_cast<float *>(input1_ptr_),
                                reinterpret_cast<float *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    } else {
      error_code = BroadcastRun(reinterpret_cast<int *>(input0_ptr_), reinterpret_cast<int *>(input1_ptr_),
                                reinterpret_cast<int *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    }
    return error_code;
  }

  /* no broadcast in runtime */
  if (data_type_ == kDataTypeFloat) {
    error_code = arithmetic_run_(reinterpret_cast<float *>(input0_ptr_) + stride * task_id,
                                 reinterpret_cast<float *>(input1_ptr_) + stride * task_id,
                                 reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count);
  } else if (data_type_ == KDataTypeBool) {
    error_code = arithmetic_run_bool_(reinterpret_cast<bool *>(input0_ptr_) + stride * task_id,
                                      reinterpret_cast<bool *>(input1_ptr_) + stride * task_id,
                                      reinterpret_cast<bool *>(out_tensors_[0]->data_c()) + stride * task_id, count);
  } else {
    error_code = arithmetic_run_int_(reinterpret_cast<int *>(input0_ptr_) + stride * task_id,
                                     reinterpret_cast<int *>(input1_ptr_) + stride * task_id,
                                     reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id, count);
  }
  return error_code;
}

int ArithmeticsRun(void *cdata, int task_id) {
  auto arithmetic_kernel = reinterpret_cast<ArithmeticCPUKernel *>(cdata);
  auto error_code = arithmetic_kernel->DoArithmetic(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticCPUKernel::FreeTmpPtr() {
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

void ArithmeticCPUKernel::InitParamInRunTime() {
  /* after infershape */
  if (arithmeticParameter_->broadcasting_) {
    outside_ = 1;
    for (auto i = arithmeticParameter_->ndim_ - 1; i >= 0; --i) {
      if (arithmeticParameter_->in_shape0_[i] != arithmeticParameter_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= arithmeticParameter_->out_shape_[i];
    }
  }
  ComputeStrides(arithmeticParameter_->in_shape0_, arithmeticParameter_->in_strides0_, arithmeticParameter_->ndim_);
  ComputeStrides(arithmeticParameter_->in_shape1_, arithmeticParameter_->in_strides1_, arithmeticParameter_->ndim_);
  ComputeStrides(arithmeticParameter_->out_shape_, arithmeticParameter_->out_strides_, arithmeticParameter_->ndim_);

  if (!input0_broadcast_) {
    input0_ptr_ = in_tensors_[0]->data_c();
  }
  if (!input1_broadcast_) {
    input1_ptr_ = in_tensors_[1]->data_c();
  }
  return;
}

int ArithmeticCPUKernel::Run() {
  InitParamInRunTime();

  int error_code = ParallelLaunch(this->context_->thread_pool_, ArithmeticsRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mul, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mul, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Add, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Add, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sub, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Sub, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Div, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RealDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
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
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Div, LiteKernelCreator<ArithmeticCPUKernel>)
}  // namespace mindspore::kernel
