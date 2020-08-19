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
#include "src/runtime/kernel/arm/int8/gather_int8.h"
#include <vector>
#include "nnacl/gather_parameter.h"
#include "nnacl/int8/gather_int8.h"
#include "nnacl/quantization/quantize.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {

int GatherInt8CPUKernel::Init() {
  axis_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;
  batchDims_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->batchDims_;
  auto in_quant_args = in_tensors_.at(0)->GetQuantParams();
  auto ind_quant_args = in_tensors_.at(1)->GetQuantParams();
  auto out_quant_args = out_tensors_.at(0)->GetQuantParams();
  param_.alpha_ = in_quant_args.front().scale / out_quant_args.front().scale;
  param_.zp_in_ = in_quant_args.front().zeroPoint;
  param_.zp_out_ = out_quant_args.front().zeroPoint;

  auto indices_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->Data());
  if (indices_ != nullptr) {
    free(indices_);
    indices_ = nullptr;
  }
  int count = in_tensors_.at(1)->ElementsNum();
  indices_ = reinterpret_cast<int *>(malloc(count * sizeof(int)));
  if (indices_ == nullptr) {
    MS_LOG(ERROR) << "Gather Malloc indices_ error!";
    return RET_ERROR;
  }
  (void)memset(indices_, 0, count * sizeof(int));
  for (int i = 0; i < count; ++i) {
    indices_[i] =
      static_cast<int>(round((indices_ptr[i] - ind_quant_args.front().zeroPoint) * ind_quant_args.front().scale));
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherInt8CPUKernel::ReSize() { return RET_OK; }

int GatherInt8CPUKernel::DoGather(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto input_ptr = reinterpret_cast<int8_t *>(input_tensor->Data());
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensor->Data());

  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  int indices_element_size = indices_tensor->ElementsNum();

  const int limit = in_shape[axis_];
  for (int i = 0; i < indices_element_size; ++i) {
    if (indices_[i] >= limit) {
      MS_LOG(ERROR) << " indice data: " << indices_[i] << " is not in [ 0, " << limit - 1 << " ]";
      return RET_ERROR;
    }
  }

  int outer_size = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size *= in_shape[i];
  }

  int inner_size = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= in_shape[i];
  }

  int stride = UP_DIV(outer_size, thread_count_);
  int count = MSMIN(stride, outer_size - stride * task_id);
  auto thread_stride = stride * task_id;

  int error_code;
  input_ptr += thread_stride * limit;
  output_ptr += thread_stride * indices_element_size;
  error_code = GatherInt8(input_ptr, output_ptr, count, inner_size, limit, indices_, indices_element_size, param_);

  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherInt8Run(void *cdata, int task_id) {
  auto gather_kernel = reinterpret_cast<GatherInt8CPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherInt8CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, GatherInt8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuGatherInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Gather);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) GatherInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Gather, CpuGatherInt8KernelCreator)
}  // namespace mindspore::kernel
