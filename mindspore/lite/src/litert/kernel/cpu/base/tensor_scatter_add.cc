/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/tensor_scatter_add.h"
#include <cstring>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorScatterAdd;

namespace mindspore::kernel {
namespace {
int TensorScatterAddRun(void *cdata, int task_id, float, float) {
  auto kernel = static_cast<TensorScatterAddCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->TensorScatterAdd(task_id);
}
}  // namespace

int TensorScatterAddCPUKernel::TensorScatterAdd(int task_id) {
  auto data_type = in_tensors_[kScatterUpdateInputIndex]->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "TensorScatterAdd only support int32 and float32 input tensor, but got " << data_type;
    return RET_ERROR;
  }
  int type = data_type == kNumberTypeFloat32 ? 0 : 1;
  auto ret = ScatterNDAdd(in_tensors_[kScatterUpdateIndex]->data(), out_tensors_[kOutputIndex]->data(),
                          output_unit_offsets_.data(), param_, type, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNDAdd failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorScatterAddCPUKernel::Run() {
  auto in_tensor = in_tensors().front();
  auto out_tensor = out_tensors().front();
  (void)memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd resize failed.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(ms_context_, TensorScatterAddRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorScatterAdd, LiteKernelCreator<TensorScatterAddCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorScatterAdd, LiteKernelCreator<TensorScatterAddCPUKernel>)
}  // namespace mindspore::kernel
