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

#include "src/runtime/kernel/arm/fp32/local_response_norm.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LocalResponseNormalization;

namespace mindspore::kernel {

int LocalResponseNormCPUKernel::Init() { return RET_OK; }

int LocalResponseNormCPUKernel::ReSize() { return RET_OK; }

int LocalResponseNormCPUKernel::DoLocalResponseNorm(int task_id) {
  auto input_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto input_ptr = reinterpret_cast<float *>(input_tensor->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensor->Data());

  auto in_shape = input_tensor->shape();
  MS_ASSERT(in_shape.size() == 4);

  int batch = in_shape[0];
  int height = in_shape[1];
  int width = in_shape[2];
  int channel = in_shape[3];

  int outer_size = batch * width * height;
  int stride = UP_DIV(outer_size, thread_count_);
  int count = MSMIN(stride, outer_size - stride * task_id);

  input_ptr += stride * task_id * channel;
  output_ptr += stride * task_id * channel;

  auto error_code = LocalResponseNorm(input_ptr, count, channel, output_ptr,
                                      reinterpret_cast<LocalResponseNormParameter *>(op_parameter_));
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoLocalResponseNorm error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LocalResponseNormRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto lrn = reinterpret_cast<LocalResponseNormCPUKernel *>(cdata);
  auto error_code = lrn->DoLocalResponseNorm(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LocalResponseNormRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LocalResponseNormCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  int error_code = LiteBackendParallelLaunch(LocalResponseNormRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LocalResponseNorm function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuLocalResponseNormFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                          const std::vector<lite::tensor::Tensor *> &outputs,
                                                          OpParameter *opParameter, const lite::Context *ctx,
                                                          const kernel::KernelKey &desc,
                                                          const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_LocalResponseNormalization);

  auto *kernel = new (std::nothrow) LocalResponseNormCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new LocalResponseNormCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LocalResponseNormalization, CpuLocalResponseNormFp32KernelCreator)
}  // namespace mindspore::kernel
