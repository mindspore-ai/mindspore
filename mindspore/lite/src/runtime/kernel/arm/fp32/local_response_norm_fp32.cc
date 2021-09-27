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

#include "src/runtime/kernel/arm/fp32/local_response_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LRN;

namespace mindspore::kernel {
int LocalResponseNormCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int LocalResponseNormCPUKernel::ReSize() { return RET_OK; }

int LocalResponseNormCPUKernel::DoLocalResponseNorm(int task_id) {
  auto input_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  auto input_ptr = reinterpret_cast<float *>(input_tensor->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensor->MutableData());

  auto in_shape = input_tensor->shape();
  MS_CHECK_TRUE_RET(in_shape.size() == C4NUM, RET_ERROR);

  int batch = in_shape.at(0);
  int height = in_shape.at(1);
  int width = in_shape.at(2);
  int channel = in_shape.at(3);

  int outer_size = batch * width * height;
  MS_CHECK_TRUE_RET(thread_count_ != 0, RET_ERROR);
  int stride = UP_DIV(outer_size, thread_count_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, RET_ERROR);
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

int LocalResponseNormRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto lrn = reinterpret_cast<LocalResponseNormCPUKernel *>(cdata);
  auto error_code = lrn->DoLocalResponseNorm(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LocalResponseNormRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LocalResponseNormCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, LocalResponseNormRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "LocalResponseNorm function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LRN, LiteKernelCreator<LocalResponseNormCPUKernel>)
}  // namespace mindspore::kernel
