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

  if (arithmeticParameter_->broadcasting_ == false) {
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
  switch (op_parameter_->type_) {
    case PrimitiveType_Mul:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementMulRelu;
          arithmetic_run_int_ = ElementMulReluInt;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementMulRelu6;
          arithmetic_run_int_ = ElementMulRelu6Int;
          break;
        default:
          arithmetic_run_ = ElementMul;
          arithmetic_run_int_ = ElementMulInt;
          break;
      }
      break;
    case PrimitiveType_Add:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementAddRelu;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementAddRelu6;
          break;
        default:
          arithmetic_run_ = ElementAdd;
          arithmetic_run_int_ = ElementAddInt;
          break;
      }
      break;
    case PrimitiveType_Sub:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementSubRelu;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementSubRelu6;
          break;
        default:
          arithmetic_run_ = ElementSub;
          arithmetic_run_int_ = ElementSubInt;
          break;
      }
      break;
    case PrimitiveType_Div:
    case PrimitiveType_RealDiv:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementDivRelu;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementDivRelu6;
          break;
        default:
          arithmetic_run_ = ElementDiv;
          break;
      }
      break;
    case PrimitiveType_LogicalAnd:
      arithmetic_run_ = ElementLogicalAnd;
      arithmetic_run_int_ = ElementLogicalAndInt;
      arithmetic_run_bool_ = ElementLogicalAndBool;
      break;
    case PrimitiveType_LogicalOr:
      arithmetic_run_ = ElementLogicalOr;
      break;
    case PrimitiveType_Maximum:
      arithmetic_run_ = ElementMaximum;
      arithmetic_run_int_ = ElementMaximumInt;
      break;
    case PrimitiveType_Minimum:
      arithmetic_run_ = ElementMinimum;
      arithmetic_run_int_ = ElementMinimumInt;
      break;
    case PrimitiveType_FloorDiv:
      arithmetic_run_ = ElementFloorDiv;
      arithmetic_run_int_ = ElementFloorDivInt;
      break;
    case PrimitiveType_FloorMod:
      arithmetic_run_ = ElementFloorMod;
      arithmetic_run_int_ = ElementFloorModInt;
      break;
    case PrimitiveType_Mod:
      arithmetic_run_ = ElementMod;
      arithmetic_run_int_ = ElementModInt;
      break;
    case PrimitiveType_SquaredDifference:
      arithmetic_run_ = ElementSquaredDifference;
      break;
    case PrimitiveType_Equal:
    case PrimitiveType_Less:
    case PrimitiveType_Greater:
    case PrimitiveType_NotEqual:
    case PrimitiveType_LessEqual:
    case PrimitiveType_GreaterEqual:
      arithmetic_run_ = nullptr;
      arithmetic_run_int_ = nullptr;
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
      arithmetic_run_ = nullptr;
      break;
  }
  return;
}

void ArithmeticCPUKernel::InitOptRunFunction() {
  if (arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) {
    switch (arithmeticParameter_->op_parameter_.type_) {
      case PrimitiveType_Mul:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMulRelu;
            arithmetic_opt_run_int_ = ElementOptMulReluInt;
            break;
          case schema::ActivationType_RELU6:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMulRelu6;
            arithmetic_opt_run_int_ = ElementOptMulRelu6Int;
            break;
          default:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMul;
            arithmetic_opt_run_int_ = ElementOptMulInt;
            break;
        }
        break;
      case PrimitiveType_Add:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptAddRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptAddRelu6;
            break;
          default:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptAdd;
            arithmetic_opt_run_int_ = ElementOptAddInt;
            break;
        }
        break;
      case PrimitiveType_Sub:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptSubRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptSubRelu6;
            break;
          default:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptSub;
            arithmetic_opt_run_int_ = ElementOptSubInt;
            break;
        }
        break;
      case PrimitiveType_Div:
      case PrimitiveType_RealDiv:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptDivRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptDivRelu6;
            break;
          default:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptDiv;
            arithmetic_opt_run_int_ = ElementOptDivInt;
            break;
        }
        break;
      case PrimitiveType_Mod:
        arithmeticParameter_->broadcasting_ = false;
        arithmetic_opt_run_ = ElementOptMod;
        arithmetic_opt_run_int_ = ElementOptModInt;
        break;
      default:
        arithmetic_opt_run_ = nullptr;
        arithmetic_opt_run_int_ = nullptr;
        break;
    }
  } else {
    arithmetic_opt_run_ = nullptr;
    arithmetic_opt_run_int_ = nullptr;
  }
  return;
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

int ArithmeticCPUKernel::ReSize() {
  InitParam();
  InitOptRunFunction();
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

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(thread_count_ != 0);
  int stride = UP_DIV(element_num, thread_count_);
  int count = MSMIN(stride, element_num - stride * task_id);

  if (arithmetic_run_ == nullptr) {
    MS_LOG(ERROR) << "arithmetic_run function is nullptr!";
    return RET_ERROR;
  }

  int error_code;
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

  if (arithmetic_opt_run_ != nullptr) {
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
    } else {
      MS_LOG(ERROR) << "Arithmetic opt run: at least one of inputs is scalar";
      return RET_ERROR;
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
