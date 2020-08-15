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
#include "src/runtime/kernel/arm/fp16/pooling_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/fp16/pooling_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore::kernel {
int PoolingFp16CPUKernel::InitBuffer() {
  int in_batch = pooling_param_->input_batch_;
  int in_h = pooling_param_->input_h_;
  int in_w = pooling_param_->input_w_;
  int in_channel = pooling_param_->input_channel_;
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(in_batch * in_h * in_w * in_channel * sizeof(float16_t)));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }

  int out_batch = pooling_param_->output_batch_;
  int out_h = pooling_param_->output_h_;
  int out_w = pooling_param_->output_w_;
  int out_channel = pooling_param_->output_channel_;
  fp16_output_ = reinterpret_cast<float16_t *>(malloc(out_batch * out_h * out_w * out_channel * sizeof(float16_t)));
  if (fp16_output_ == nullptr) {
    MS_LOG(ERROR) << "fp16_out malloc failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::Init() {
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return ret;
  }

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Buffer failed.";
    return ret;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::ReSize() {
  auto ret = Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Pooling resize init failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::RunImpl(int task_id) {
  if (pooling_param_->max_pooling_) {
    MaxPoolingFp16(fp16_input_, fp16_output_, pooling_param_, task_id);
  } else {
    AvgPoolingFp16(fp16_input_, fp16_output_, pooling_param_, task_id);
  }
  return RET_OK;
}

int PoolingFp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto pooling = reinterpret_cast<PoolingFp16CPUKernel *>(cdata);
  auto error_code = pooling->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Pooling Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PoolingFp16CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto ele_num = in_tensors_.front()->ElementsNum();
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->Data());
  Float32ToFloat16(input_ptr, fp16_input_, ele_num);

  int error_code = LiteBackendParallelLaunch(PoolingFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  auto out_ele_num = out_tensors_.front()->ElementsNum();
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  Float16ToFloat32(fp16_output_, output_ptr, out_ele_num);
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const Context *ctx,
                                                const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Pooling);
  auto *kernel = new (std::nothrow) PoolingFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PoolingCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Pooling, CpuPoolingFp16KernelCreator)
}  // namespace mindspore::kernel
