/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp16/prelu_fp16.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/prelu_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::kernel {
int PReluFp16CPUKernel::DoExcute(int task_id) const {
  int thread_num = param_->op_parameter_.thread_num_;
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread_num is 0!";
    return RET_ERROR;
  }
  int num = param_->channelShared ? param_->input_num_ : param_->input_num_ / param_->channel_num_;
  int step = UP_DIV(num, thread_num);
  int start = task_id * step;
  int end = MSMIN(start + step, num);

  if (param_->channelShared) {
    PReluShareChannelFp16(static_cast<float16_t *>(input_data_), static_cast<float16_t *>(output_data_),
                          static_cast<float16_t *>(slope_data_)[0], start, end);
  } else {
    PReluFp16(static_cast<float16_t *>(input_data_), static_cast<float16_t *>(output_data_),
              static_cast<float16_t *>(slope_data_), start, end, param_->channel_num_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_PReLUFusion, LiteKernelCreator<PReluFp16CPUKernel>)
}  // namespace mindspore::kernel
