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

#include "src/runtime/kernel/arm/fp32/arithmetic.h"
#include "src/runtime/kernel/arm/int8/add_int8.h"
#include "src/runtime/kernel/arm/int8/mul_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

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

int ArithmeticCPUKernel::ReSize() {
  arithmeticParameter_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arithmeticParameter_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arithmeticParameter_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  if (arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) {
    switch (arithmeticParameter_->op_parameter_.type_) {
      case PrimitiveType_Mul:
        switch (arithmeticParameter_->activation_type_) {
          case schema::ActivationType_RELU:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMulRelu;
            break;
          case schema::ActivationType_RELU6:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMulRelu6;
            break;
          default:
            arithmeticParameter_->broadcasting_ = false;
            arithmetic_opt_run_ = ElementOptMul;
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
      default:
        break;
    }
  }
  return RET_OK;
}

int ArithmeticCPUKernel::BroadcastRun(float *input0, float *input1, float *output, int dim, int out_count,
                                        int out_thread_stride) {
  if (dim > break_pos_) {
    return arithmetic_run_(input0 + out_thread_stride, input1 + out_thread_stride, output + out_thread_stride,
                           out_count);
  }
  for (int i = 0; i < arithmeticParameter_->out_shape_[dim]; ++i) {
    int pos0_ = arithmeticParameter_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = arithmeticParameter_->in_shape1_[dim] == 1 ? 0 : i;
    int error_code =
      BroadcastRun(input0 + pos0_ * arithmeticParameter_->in_strides0_[dim],
                     input1 + pos1_ * arithmeticParameter_->in_strides1_[dim],
                     output + i * arithmeticParameter_->out_strides_[dim], dim + 1, out_count, out_thread_stride);
    if (error_code != RET_OK) {
      return error_code;
    }
  }
  return RET_OK;
}

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  auto input0_data = reinterpret_cast<float *>(in_tensors_[0]->Data());
  auto input1_data1 = reinterpret_cast<float *>(in_tensors_[1]->Data());
  auto output_data = reinterpret_cast<float *>(out_tensors_[0]->Data());
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(thread_count_ != 0);
  int stride = UP_DIV(element_num, thread_count_);
  int count = MSMIN(stride, element_num - stride * task_id);

  if (arithmetic_run_ == nullptr) {
    MS_LOG(ERROR) << "arithmetic_run function is nullptr!";
    return RET_ERROR;
  }

  int error_code = RET_OK;
  if (arithmeticParameter_->broadcasting_) {
    stride = UP_DIV(outside_, thread_count_);
    out_count_ = MSMIN(stride, outside_ - stride * task_id);
    out_thread_stride_ = stride * task_id;
    error_code = BroadcastRun(input0_data, input1_data1, output_data, 0, out_count_, out_thread_stride_);
  } else if (arithmetic_opt_run_ != nullptr) {
    if (arithmeticParameter_->in_elements_num0_ == 1) {
      error_code = arithmetic_opt_run_(input0_data, input1_data1 + stride * task_id, output_data + stride * task_id,
                                       count, arithmeticParameter_);
    } else if (arithmeticParameter_->in_elements_num1_ == 1) {
      error_code = arithmetic_opt_run_(input0_data + stride * task_id, input1_data1, output_data + stride * task_id,
                                       count, arithmeticParameter_);
    } else {
      error_code = arithmetic_opt_run_(input0_data + stride * task_id, input1_data1 + stride * task_id,
                                       output_data + stride * task_id, count, arithmeticParameter_);
    }
  } else {
    error_code = arithmetic_run_(input0_data + stride * task_id, input1_data1 + stride * task_id,
                                 output_data + stride * task_id, count);
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
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
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

  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, ArithmeticsRun, this, thread_count_);

  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *parameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(parameter != nullptr);
  auto kernel = new (std::nothrow) ArithmeticCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, name: " << parameter->name_;
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
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Add, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Sub, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Div, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Maximum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Minimum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorMod, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Equal, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NotEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Less, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LessEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Greater, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Eltwise, CpuArithmeticFp32KernelCreator)

}  // namespace mindspore::kernel
