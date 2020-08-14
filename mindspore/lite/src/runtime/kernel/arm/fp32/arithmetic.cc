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
void ArithmeticCPUKernel::FreeTileData() {
  if (tile_data0_ != nullptr) {
    delete[](tile_data0_);
    tile_data0_ = nullptr;
  }
  if (tile_data1_ != nullptr) {
    delete[](tile_data1_);
    tile_data1_ = nullptr;
  }
}

ArithmeticCPUKernel::~ArithmeticCPUKernel() { FreeTileData(); }

int ArithmeticCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCPUKernel::ReSize() {
  FreeTileData();
  arithmeticParameter_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arithmeticParameter_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arithmeticParameter_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  if (arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) {
    if (arithmeticParameter_->activation_type_ == schema::ActivationType_NO_ACTIVATION) {
      switch (arithmeticParameter_->op_parameter_.type_) {
        case PrimitiveType_Mul:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptMul;
          break;
        case PrimitiveType_Add:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptAdd;
          break;
        case PrimitiveType_Sub:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptSub;
          break;
        default:
          break;
      }
    }
  }

  if (arithmeticParameter_->broadcasting_) {
    tile_data0_ = new float[arithmeticParameter_->out_elements_num_];
    tile_data1_ = new float[arithmeticParameter_->out_elements_num_];
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
    error_code = arithmetic_run_(tile_data0_ + stride * task_id, tile_data1_ + stride * task_id,
                                 output_data + stride * task_id, count);
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

int ArithmeticsRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
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
    auto input_data0 = reinterpret_cast<float *>(in_tensors_[0]->Data());
    auto input_data1 = reinterpret_cast<float *>(in_tensors_[1]->Data());
    TileDimensions(input_data0, input_data1, tile_data0_, tile_data1_, arithmeticParameter_);
  }
  int error_code = LiteBackendParallelLaunch(ArithmeticsRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *parameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
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
