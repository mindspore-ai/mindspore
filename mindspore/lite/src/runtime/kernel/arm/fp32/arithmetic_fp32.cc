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
#include "src/ops/populate/arithmetic_populate.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Eltwise;

namespace mindspore::kernel {

ArithmeticCPUKernel::~ArithmeticCPUKernel() {}

int ArithmeticCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCPUKernel::PreProcess() {
  if (!InferShapeDone()) {
    (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->set_infer_flag(true);
    auto ret = (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->InferShape(in_tensors_, out_tensors_);
    if (ret != 0) {
      (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->set_infer_flag(false);
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    if (op_parameter_ != nullptr) {
      free(op_parameter_);
      op_parameter_ = nullptr;
    }
    op_parameter_ = PopulateArithmetic(primitive_);
    if (op_parameter_ == nullptr) {
      MS_LOG(ERROR) << "Malloc parameter failed";
      return RET_ERROR;
    }
    arithmeticParameter_ = reinterpret_cast<ArithmeticParameter *>(op_parameter_);
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  auto outputs = this->out_tensors();
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    output->MallocData();
  }
  return RET_OK;
}

int ArithmeticCPUKernel::ReSize() {
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32 || in_tensors_[0]->data_type() == kNumberTypeFloat16) {
    data_type_ = kDataTypeFloat;
  } else {
    data_type_ = kDataTypeInt;
  }
  arithmeticParameter_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arithmeticParameter_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arithmeticParameter_->out_elements_num_ = out_tensors_[0]->ElementsNum();
  for (size_t i = 0; i < in_tensors_[0]->shape().size(); i++) {
    if (arithmeticParameter_->in_shape0_[i] == -1) {
      memcpy(arithmeticParameter_->in_shape0_, static_cast<void *>(in_tensors_[0]->shape().data()),
             in_tensors_[0]->shape().size() * sizeof(int));
      break;
    }
  }
  for (size_t i = 0; i < in_tensors_[1]->shape().size(); i++) {
    if (arithmeticParameter_->in_shape1_[i] == -1) {
      memcpy(arithmeticParameter_->in_shape1_, static_cast<void *>(in_tensors_[1]->shape().data()),
             in_tensors_[1]->shape().size() * sizeof(int));
      break;
    }
  }
  for (size_t i = 0; i < out_tensors_[0]->shape().size(); i++) {
    if (arithmeticParameter_->out_shape_[i] == -1) {
      memcpy(arithmeticParameter_->out_shape_, static_cast<void *>(out_tensors_[0]->shape().data()),
             out_tensors_[0]->shape().size() * sizeof(int));
      break;
    }
  }

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
            break;
        }
        break;
      case PrimitiveType_Equal:
      case PrimitiveType_Less:
      case PrimitiveType_Greater:
      case PrimitiveType_NotEqual:
      case PrimitiveType_LessEqual:
      case PrimitiveType_GreaterEqual:
        arithmetic_opt_run_ = nullptr;
        arithmetic_opt_run_int_ = nullptr;
        break;
      default:
        break;
    }
  } else {
    arithmetic_opt_run_ = nullptr;
    arithmetic_opt_run_int_ = nullptr;
  }
  return RET_OK;
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
  if (arithmeticParameter_->broadcasting_) {  // need broadcast
    stride = UP_DIV(outside_, thread_count_);
    int out_count = MSMIN(stride, outside_ - stride * task_id);
    int out_thread_stride = stride * task_id;
    if (data_type_ == kDataTypeFloat) {
      error_code = BroadcastRun(reinterpret_cast<float *>(in_tensors_[0]->data_c()),
                                reinterpret_cast<float *>(in_tensors_[1]->data_c()),
                                reinterpret_cast<float *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    } else {
      error_code = BroadcastRun(reinterpret_cast<int *>(in_tensors_[0]->data_c()),
                                reinterpret_cast<int *>(in_tensors_[1]->data_c()),
                                reinterpret_cast<int *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    }

  } else if (arithmetic_opt_run_ != nullptr) {  // no broadcast, one of input is scalar
    if (arithmeticParameter_->in_elements_num0_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        error_code = arithmetic_opt_run_(reinterpret_cast<float *>(in_tensors_[0]->data_c()),
                                         reinterpret_cast<float *>(in_tensors_[1]->data_c()) + stride * task_id,
                                         reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count,
                                         arithmeticParameter_);
      } else {
        error_code = arithmetic_opt_run_int_(reinterpret_cast<int *>(in_tensors_[0]->data_c()),
                                             reinterpret_cast<int *>(in_tensors_[1]->data_c()) + stride * task_id,
                                             reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id,
                                             count, arithmeticParameter_);
      }
    } else if (arithmeticParameter_->in_elements_num1_ == 1) {
      if (data_type_ == kDataTypeFloat) {
        error_code = arithmetic_opt_run_(reinterpret_cast<float *>(in_tensors_[0]->data_c()) + stride * task_id,
                                         reinterpret_cast<float *>(in_tensors_[1]->data_c()),
                                         reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count,
                                         arithmeticParameter_);
      } else {
        error_code = arithmetic_opt_run_int_(reinterpret_cast<int *>(in_tensors_[0]->data_c()) + stride * task_id,
                                             reinterpret_cast<int *>(in_tensors_[1]->data_c()),
                                             reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id,
                                             count, arithmeticParameter_);
      }
    } else {
      MS_LOG(ERROR) << "Arithmetic opt run: at least one of inputs is scalar";
      return RET_ERROR;
    }
  } else {  // no broadcast, neither is scalar, two same shape
    if (data_type_ == kDataTypeFloat) {
      error_code = arithmetic_run_(reinterpret_cast<float *>(in_tensors_[0]->data_c()) + stride * task_id,
                                   reinterpret_cast<float *>(in_tensors_[1]->data_c()) + stride * task_id,
                                   reinterpret_cast<float *>(out_tensors_[0]->data_c()) + stride * task_id, count);
    } else {
      error_code = arithmetic_run_int_(reinterpret_cast<int *>(in_tensors_[0]->data_c()) + stride * task_id,
                                       reinterpret_cast<int *>(in_tensors_[1]->data_c()) + stride * task_id,
                                       reinterpret_cast<int *>(out_tensors_[0]->data_c()) + stride * task_id, count);
    }
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
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

int ArithmeticCPUKernel::Run() {
  if (arithmeticParameter_->broadcasting_) {
    outside_ = 1;
    for (auto i = arithmeticParameter_->ndim_ - 1; i >= 0; --i) {
      if (arithmeticParameter_->in_shape0_[i] != arithmeticParameter_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= arithmeticParameter_->out_shape_[i];
    }
    ComputeStrides(arithmeticParameter_->in_shape0_, arithmeticParameter_->in_strides0_, arithmeticParameter_->ndim_);
    ComputeStrides(arithmeticParameter_->in_shape1_, arithmeticParameter_->in_strides1_, arithmeticParameter_->ndim_);
    ComputeStrides(arithmeticParameter_->out_shape_, arithmeticParameter_->out_strides_, arithmeticParameter_->ndim_);
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, ArithmeticsRun, this, thread_count_);

  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(parameter != nullptr);
  auto kernel = new (std::nothrow) ArithmeticCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mul, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Mul, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mul, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Add, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_Add, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Add, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sub, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Div, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RealDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Maximum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Minimum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorMod, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_FloorDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt, PrimitiveType_FloorMod, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorMod, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Eltwise, CpuArithmeticFp32KernelCreator)

}  // namespace mindspore::kernel
