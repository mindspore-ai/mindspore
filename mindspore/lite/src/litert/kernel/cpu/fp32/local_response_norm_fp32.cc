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

#include "src/litert/kernel/cpu/fp32/local_response_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LRN;

namespace mindspore::kernel {
int LocalResponseNormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int LocalResponseNormCPUKernel::ReSize() { return RET_OK; }

bool LocalResponseNormCPUKernel::CheckParamsValid() const {
  MS_CHECK_GT(lrn_param_->depth_radius_, 0, false);
  return true;
}

int LocalResponseNormCPUKernel::DoLocalResponseNorm(int task_id) const {
  auto input_tensor = in_tensors_.front();
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(input_tensor);
  CHECK_NULL_RETURN(out_tensor);
  auto input_ptr = reinterpret_cast<float *>(input_tensor->MutableData());
  auto output_ptr = reinterpret_cast<float *>(out_tensor->MutableData());
  CHECK_NULL_RETURN(input_ptr);
  CHECK_NULL_RETURN(output_ptr);

  auto in_shape = input_tensor->shape();
  MS_CHECK_TRUE_RET(in_shape.size() == C4NUM, RET_ERROR);

  int batch = in_shape.at(kNHWC_N);
  int height = in_shape.at(kNHWC_H);
  int width = in_shape.at(kNHWC_W);
  int channel = in_shape.at(kNHWC_C);

  MS_CHECK_INT_MUL_NOT_OVERFLOW(batch, width, RET_ERROR);
  int size_bw = batch * width;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(size_bw, height, RET_ERROR);
  int outer_size = size_bw * height;
  MS_CHECK_TRUE_RET(thread_count_ != 0, RET_ERROR);
  int stride = UP_DIV(outer_size, thread_count_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, RET_ERROR);
  int start = stride * task_id;
  int count = MSMIN(stride, outer_size - start);

  MS_CHECK_INT_MUL_NOT_OVERFLOW(start, channel, RET_ERROR);
  input_ptr += start * channel;
  output_ptr += start * channel;

  auto error_code = LocalResponseNorm(input_ptr, count, channel, output_ptr,
                                      reinterpret_cast<LocalResponseNormParameter *>(op_parameter_));
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoLocalResponseNorm error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int LocalResponseNormRun(void *cdata, int task_id, float, float) {
  auto lrn = reinterpret_cast<const LocalResponseNormCPUKernel *>(cdata);
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
