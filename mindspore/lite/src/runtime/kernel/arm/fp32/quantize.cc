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
#include "src/runtime/kernel/arm/fp32/quantize.h"
#include <vector>
#include "src/runtime/kernel/arm/opclib/fp32/quantize.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_OnnxInt8Quantize;

namespace mindspore::kernel {
namespace {
constexpr int kQuantizeInputNum = 1;
constexpr int kQuantizeOutputNum = 1;
}  // namespace

int QuantizeCPUKernel::Init() {
  if (inputs_.size() != 1) {
    MS_LOG(ERROR) << "inputs number should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "outputs number should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  auto in_tensor = inputs_.front();
  num_unit_ = static_cast<int>(in_tensor->DataSize());
  thread_n_num_ = MSMIN(thread_num_, num_unit_);
  thread_n_stride_ = UP_DIV(num_unit_, thread_n_num_);

  return RET_OK;
}

int QuantizeCPUKernel::ReSize() { return RET_OK; }

int QuantizeCPUKernel::Quantize(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto quant_arg = inputs_.front()->GetQuantParams().front();
  int ret = QuantizeToInt8(input_ptr_ + thread_offset, output_ptr_ + thread_offset, quant_arg.scale,
                           quant_arg.zeroPoint, num_unit_thread);

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Quantize error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantizeRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<QuantizeCPUKernel *>(cdata);
  auto ret = g_kernel->Quantize(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "QuantizeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantizeCPUKernel::Run() {
  input_ptr_ = reinterpret_cast<float *>(inputs_[0]->Data());
  output_ptr_ = reinterpret_cast<int8_t *>(outputs_[0]->Data());
  int ret = LiteBackendParallelLaunch(QuantizeRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

kernel::LiteKernel *CpuQuantizeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx,
                                                 const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) QuantizeCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new QuantizeCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_OnnxInt8Quantize, CpuQuantizeFp32KernelCreator)
}  // namespace mindspore::kernel
