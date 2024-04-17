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

#include "src/litert/kernel/cpu/base/custom_tensor_scatter.h"
#include <cstring>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/base/scatter_nd_binary.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
int TensorScatterRun(void *cdata, int task_id, float, float) {
  auto kernel = static_cast<CustomTensorScatterCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->TensorScatterDispatch(task_id);
}
}  // namespace

int CustomTensorScatterCPUKernel::TensorScatterDispatch(int task_id) {
  auto data_type = in_tensors_[kScatterUpdateInputIndex]->data_type();
  if (data_type != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "TensorScatterMax only support float32 input tensor, but got " << data_type;
    return RET_ERROR;
  }
  // multi thread have some problems to solve
  param_->op_parameter.thread_num_ = 1;
  auto ret = ScatterNDMax(in_tensors_[kScatterUpdateIndex]->data(), out_tensors_[kOutputIndex]->data(),
                          output_unit_offsets_.data(), param_, 0, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNDMax failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int CustomTensorScatterCPUKernel::Run() {
  auto in_tensor = in_tensors().front();
  auto out_tensor = out_tensors().front();
  (void)memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd resize failed.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(ms_context_, TensorScatterRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "TensorScatterAdd error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CustomTensorScatterMax,
           LiteKernelCreator<CustomTensorScatterCPUKernel>)
}  // namespace mindspore::kernel
