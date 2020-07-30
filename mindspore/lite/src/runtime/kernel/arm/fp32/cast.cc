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
#include "src/runtime/kernel/arm/opclib/fp32/cast.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
const std::vector<int> kSupportInputDataType = {kNumberTypeUInt8, kNumberTypeInt32};
int CastRun(int thread_id, LiteParallelGroupEnv *penv, void *cdata) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }

  return reinterpret_cast<CastCPUKernel *>(cdata)->DoCast(thread_id);
}
}  // namespace

int CastCPUKernel::Init() {
  data_num_ = inputs_[0]->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  thread_num_ = MSMIN(thread_num_, data_num_);
  stride_ = UP_DIV(data_num_, thread_num_);
  return RET_OK;
}

int CastCPUKernel::DoCast(int thread_id) {
  auto input = inputs_.at(0);
  int data_num = MSMIN(stride_, data_num_ - thread_id * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }

  auto offset = thread_id * stride_;
  auto output_data = reinterpret_cast<float *>(outputs_.at(0)->Data());
  switch (input->data_type()) {
    case kNumberTypeUInt8:
      Uint8ToFloat32(reinterpret_cast<uint8_t *>(input->Data()) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeInt32:
      Int32ToFloat32(reinterpret_cast<int32_t *>(input->Data()) + offset, output_data + offset, data_num);
      break;
    default:
      MS_LOG(ERROR) << "Unsupport input data type " << input->data_type();
      return RET_ERROR;
  }
  return RET_OK;
}

int CastCPUKernel::Run() {
  if (data_num_ == 0) {
    return RET_OK;
  }
  return LiteBackendParallelLaunch(CastRun, this, thread_num_);
}

kernel::LiteKernel *CpuCastFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    return nullptr;
  }
  if (ctx->threadNum == 0) {
    MS_LOG(ERROR) << "context thread num is 0!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) CastCPUKernel(opParameter, inputs, outputs, ctx);
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

REG_KERNEL(kCPU, PrimitiveType_Cast, CpuCastFp32KernelCreator)
}  // namespace mindspore::kernel
