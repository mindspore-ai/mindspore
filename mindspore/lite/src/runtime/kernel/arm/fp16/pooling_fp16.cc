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
#include "nnacl/fp16/pooling_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Pooling;

namespace mindspore::kernel {
int PoolingFp16CPUKernel::Init() {
  auto ret = PoolingBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase Init failed.";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PoolingFp16CPUKernel::ReSize() {
  auto ret = PoolingBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PoolingBase ReSize fai1!ret: " << ret;
    return ret;
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

static int PoolingFp16Impl(void *cdata, int task_id) {
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

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto in_data_type_ = input_tensor->data_type();
  MS_ASSERT(in_data_type_ == kNumberTypeFloat32 || in_data_type_ == kNumberTypeFloat16);
  fp16_input_ = ConvertInputFp32toFp16(input_tensor, context_);

  auto out_tensor = out_tensors_.at(kOutputIndex);
  auto out_data_type_ = out_tensor->data_type();
  MS_ASSERT(out_data_type_ == kNumberTypeFloat32 || out_data_type_ == kNumberTypeFloat16);
  fp16_output_ = MallocOutputFp16(out_tensor, context_);

  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, PoolingFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "pooling error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  if (in_data_type_ == kNumberTypeFloat32) {
    context_->allocator->Free(fp16_input_);
  }
  if (out_data_type_ == kNumberTypeFloat32) {
    auto out_ele_num = out_tensor->ElementsNum();
    auto output_addr = reinterpret_cast<float *>(out_tensor->Data());
    Float16ToFloat32(fp16_output_, output_addr, out_ele_num);
    context_->allocator->Free(fp16_output_);
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPoolingFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const Context *ctx,
                                                const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
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
