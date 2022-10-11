/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/power_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::kernel {
int PowerCPUKernel::Prepare() {
  MS_CHECK_TRUE_MSG(in_tensors_.size() == C2NUM, RET_ERROR, "Only support Power op with 2 inputs.");
  auto base_data_type = in_tensors_.at(0)->data_type();
  MS_CHECK_TRUE_MSG((base_data_type == kNumberTypeFloat32 || base_data_type == kNumberTypeFloat16 ||
                     base_data_type == kNumberTypeFloat),
                    RET_ERROR, "unsupported datatype of base for Power op.");
  auto exp_data_type = in_tensors_.at(1)->data_type();
  MS_CHECK_TRUE_MSG((exp_data_type == kNumberTypeFloat32 || exp_data_type == kNumberTypeFloat ||
                     exp_data_type == kNumberTypeInt32 || exp_data_type == kNumberTypeInt),
                    RET_ERROR, "unsupported datatype of exponent for Power op.");
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int PowerCPUKernel::ReSize() { return RET_OK; }

int PowerImpl(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<const PowerCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerImpl error: " << ret;
    return ret;
  }
  return RET_OK;
}

int PowerCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, PowerImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerCPUKernel error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerCPUKernel::RunImpl(int task_id) const {
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIndex));
  auto x_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(x_addr);
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(output_addr);
  auto size = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);
  if (len <= 0) {
    return RET_OK;
  }
  float *exp_addr = reinterpret_cast<float *>(in_tensors_[1]->data());
  CHECK_NULL_RETURN(exp_addr);
  bool broadcast = in_tensors_[0]->shape() == in_tensors_[1]->shape() ? false : true;

  float *cur_exp = nullptr;
  if (broadcast) {
    cur_exp = exp_addr;
  } else {
    cur_exp = exp_addr + stride * task_id;
  }
  CHECK_NULL_RETURN(cur_exp);
  auto error_code =
    Power(x_addr + stride * task_id, cur_exp, output_addr + stride * task_id, len, scale_, shift_, broadcast);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "PowerCPUKernel RunImpl error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PowFusion, LiteKernelCreator<PowerCPUKernel>)
}  // namespace mindspore::kernel
