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
#include "src/runtime/kernel/arm/fp32/cast.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32/cast.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
int CastRun(int thread_id, LiteParallelGroupEnv *penv, void *cdata) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastCPUKernel *>(cdata)->DoCast(thread_id);
}
}  // namespace

int CastCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CastCPUKernel::ReSize() {
  data_num_ = in_tensors_[0]->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  op_parameter_->thread_num_ = MSMIN(op_parameter_->thread_num_, data_num_);
  stride_ = UP_DIV(data_num_, op_parameter_->thread_num_);
  return RET_OK;
}

int CastCPUKernel::DoCast(int thread_id) {
  auto input = in_tensors_.at(0);
  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto output = out_tensors_.at(0);
  auto output_data = output->Data();
  auto input_data_type = input->data_type();
  auto output_data_type = output->data_type();
  if (output_data_type != kNumberTypeFloat32) {
    if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
      Float32ToInt32(reinterpret_cast<float *>(input->Data()) + offset,
                     reinterpret_cast<int32_t *>(output_data) + offset, data_num);
    } else {
      MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " to " << output_data_type;
      return RET_ERROR;
    }
  } else {
    switch (input_data_type) {
      case kNumberTypeUInt8:
        Uint8ToFloat32(reinterpret_cast<uint8_t *>(input->Data()) + offset,
                       reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      case kNumberTypeInt32:
        Int32ToFloat32(reinterpret_cast<int32_t *>(input->Data()) + offset,
                       reinterpret_cast<float *>(output_data) + offset, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported input data type " << input_data_type;
        return RET_ERROR;
    }
  }
  return RET_OK;
}

int CastCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  if (data_num_ == 0) {
    return RET_OK;
  }
  return LiteBackendParallelLaunch(CastRun, this, op_parameter_->thread_num_);
}

kernel::LiteKernel *CpuCastFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    return nullptr;
  }
  if (ctx->thread_num_ == 0) {
    MS_LOG(ERROR) << "context thread num is 0!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) CastCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new CastCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Cast, CpuCastFp32KernelCreator)
}  // namespace mindspore::kernel
