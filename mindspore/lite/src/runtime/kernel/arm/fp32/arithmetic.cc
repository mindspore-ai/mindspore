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

ArithmeticCPUKernel::~ArithmeticCPUKernel() {
  if (tile_data0_ != nullptr) {
    free(tile_data0_);
    tile_data0_ = nullptr;
  }
  if (tile_data1_ != nullptr) {
    free(tile_data1_);
    tile_data1_ = nullptr;
  }
}
int ArithmeticCPUKernel::Init() {
  auto element_num = outputs_[0]->ElementsNum();

  tile_data0_ = new float[element_num];
  tile_data1_ = new float[element_num];

  return RET_OK;
}

int ArithmeticCPUKernel::ReSize() { return RET_OK; }

int ArithmeticCPUKernel::DoArithmetic(int task_id) {
  auto input0_data = reinterpret_cast<float *>(inputs_[0]->Data());
  auto input1_data1 = reinterpret_cast<float *>(inputs_[1]->Data());
  auto output_data = reinterpret_cast<float *>(outputs_[0]->Data());
  auto element_num = outputs_[0]->ElementsNum();
  if (arithmeticParameter_->broadcasting_) {
    if (arithmetic_broadcast_run_ == nullptr) {
      MS_LOG(ERROR) << "broadcasting_run function is nullptr!";
      return RET_ERROR;
    }

    MS_ASSERT(thread_count_ != 0);
    int stride = UP_DIV(element_num, thread_count_);
    int count = MSMIN(stride, element_num - stride * task_id);

    int error_code = arithmetic_run_(tile_data0_ + stride * task_id, tile_data1_ + stride * task_id,
                                     output_data + stride * task_id, count);

    if (error_code != RET_OK) {
      return RET_ERROR;
    }
  } else if (arithmetic_run_ != nullptr) {
    int error_code = arithmetic_run_(input0_data, input1_data1, output_data, element_num);
    if (error_code != RET_OK) {
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "arithmetic_run function is nullptr!";
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
  if (arithmeticParameter_->broadcasting_) {
    auto input_data0 = reinterpret_cast<float *>(inputs_[0]->Data());
    auto input_data1 = reinterpret_cast<float *>(inputs_[1]->Data());
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
                                                   const kernel::KernelKey &desc) {
  MS_ASSERT(parameter);
  MS_ASSERT(inputs.at(0));
  auto data_type = inputs.at(0)->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeFloat32:
      kernel = new (std::nothrow) ArithmeticCPUKernel(parameter, inputs, outputs, ctx);
      break;
    case kNumberTypeInt8:
      if (desc.type == schema::PrimitiveType_Add) {
        kernel = new (std::nothrow) QuantizedAddCPUKernel(parameter, inputs, outputs, ctx);
      } else if (desc.type == schema::PrimitiveType_Mul) {
        kernel = new (std::nothrow) MulInt8CPUKernel(parameter, inputs, outputs, ctx);
      } else {
      }
      break;
    default:
      break;
  }
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

REG_KERNEL(kCPU, PrimitiveType_Mul, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Add, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Sub, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Div, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_LogicalAnd, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_LogicalOr, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Maximum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Minimum, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_FloorDiv, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_FloorMod, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_SquaredDifference, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Equal, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_NotEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Less, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_LessEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Greater, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_GreaterEqual, CpuArithmeticFp32KernelCreator)
REG_KERNEL(kCPU, PrimitiveType_Eltwise, CpuArithmeticFp32KernelCreator)

}  // namespace mindspore::kernel

