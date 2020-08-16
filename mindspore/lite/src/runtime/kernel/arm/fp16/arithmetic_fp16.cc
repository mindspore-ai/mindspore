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

#include "src/runtime/kernel/arm/fp16/arithmetic_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/arithmetic_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::schema::PrimitiveType_Add;
using mindspore::schema::PrimitiveType_Mul;
using mindspore::schema::PrimitiveType_Sub;

namespace mindspore::kernel {
void ArithmeticFP16CPUKernel::FreeTileData() {
  if (tile_data0_ != nullptr) {
    free(tile_data0_);
    tile_data0_ = nullptr;
  }
  if (tile_data1_ != nullptr) {
    free(tile_data1_);
    tile_data1_ = nullptr;
  }
}

ArithmeticFP16CPUKernel::~ArithmeticFP16CPUKernel() { FreeTileData(); }

int ArithmeticFP16CPUKernel::Init() {
  switch (op_parameter_->type_) {
    case PrimitiveType_Mul:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementMulReluFp16;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementMulRelu6Fp16;
          break;
        default:
          arithmetic_run_ = ElementMulFp16;
          break;
      }
      break;
    case PrimitiveType_Add:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementAddReluFp16;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementAddRelu6Fp16;
          break;
        default:
          arithmetic_run_ = ElementAddFp16;
          break;
      }
      break;
    case PrimitiveType_Sub:
      switch (arithmeticParameter_->activation_type_) {
        case schema::ActivationType_RELU:
          arithmetic_run_ = ElementSubReluFp16;
          break;
        case schema::ActivationType_RELU6:
          arithmetic_run_ = ElementSubRelu6Fp16;
          break;
        default:
          arithmetic_run_ = ElementSubFp16;
          break;
      }
      break;
    default:
      MS_LOG(ERROR) << "Error Operator type " << op_parameter_->type_;
      arithmetic_run_ = nullptr;
      break;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticFP16CPUKernel::ReSize() {
  FreeTileData();
  arithmeticParameter_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arithmeticParameter_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arithmeticParameter_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  if (arithmeticParameter_->in_elements_num0_ == 1 || arithmeticParameter_->in_elements_num1_ == 1) {
    if (arithmeticParameter_->activation_type_ == schema::ActivationType_NO_ACTIVATION) {
      switch (arithmeticParameter_->op_parameter_.type_) {
        case PrimitiveType_Mul:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptMulFp16;
          break;
        case PrimitiveType_Add:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptAddFp16;
          break;
        case PrimitiveType_Sub:
          arithmeticParameter_->broadcasting_ = false;
          arithmetic_opt_run_ = ElementOptSubFp16;
          break;
        default:
          break;
      }
    }
  }

  if (arithmeticParameter_->broadcasting_) {
    auto tile_size = arithmeticParameter_->out_elements_num_ * sizeof(float16_t);
    tile_data0_ = reinterpret_cast<float16_t *>(malloc(tile_size));
    tile_data1_ = reinterpret_cast<float16_t *>(malloc(tile_size));
    if (tile_data0_ == nullptr || tile_data1_ == nullptr) {
      MS_LOG(ERROR) << "malloc tile data fail!";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int ArithmeticFP16CPUKernel::DoArithmetic(int task_id) {
  auto input0_data = reinterpret_cast<float16_t *>(in_tensors_[0]->Data());
  auto input1_data1 = reinterpret_cast<float16_t *>(in_tensors_[1]->Data());
  auto output_data = reinterpret_cast<float16_t *>(out_tensors_[0]->Data());
  auto element_num = out_tensors_[0]->ElementsNum();

  int stride = UP_DIV(element_num, context_->thread_num_);
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
  auto arithmetic_kernel = reinterpret_cast<ArithmeticFP16CPUKernel *>(cdata);
  auto error_code = arithmetic_kernel->DoArithmetic(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticFP16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }

  if (arithmeticParameter_->broadcasting_) {
    auto input_data0 = reinterpret_cast<float16_t *>(in_tensors_[0]->Data());
    auto input_data1 = reinterpret_cast<float16_t *>(in_tensors_[1]->Data());
    TileDimensionsFp16(input_data0, input_data1, tile_data0_, tile_data1_, arithmeticParameter_);
  }
  ret = LiteBackendParallelLaunch(ArithmeticsRun, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic function fail!ret: " << ret;
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuArithmeticFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *parameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "input parameter is null!";
    return nullptr;
  }
  auto kernel = new (std::nothrow) ArithmeticFP16CPUKernel(parameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Mul, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Add, CpuArithmeticFp16KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Sub, CpuArithmeticFp16KernelCreator)
}  // namespace mindspore::kernel
