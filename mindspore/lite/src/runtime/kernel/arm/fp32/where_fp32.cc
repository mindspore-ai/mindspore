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
#include "src/runtime/kernel/arm/fp32/where_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/where.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Where;

namespace mindspore::kernel {
int WhereCPUKernel::Init() {
  where_param_->op_parameter_.thread_num_ = thread_count_;
  return RET_OK;
}

int WhereCPUKernel::DoExcute(int task_id) {
  MS_ASSERT(input_data);
  MS_ASSERT(input_data1);
  MS_ASSERT(input_data2);
  MS_ASSERT(output_data);
  MS_ASSERT(where_param_);
  Where(input_data, input_data1, input_data2, output_data, where_param_, task_id);
  return RET_OK;
}

int WhereRun(void *cdata, int task_id) {
  auto wheredata = reinterpret_cast<WhereCPUKernel *>(cdata);
  auto ret = wheredata->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WhereRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
int WhereCPUKernel::Run() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  auto input1 = in_tensors_.at(1);
  MS_ASSERT(input1);
  auto input2 = in_tensors_.at(2);
  MS_ASSERT(input2);
  int num = input->ElementsNum();
  int num1_ = input1->ElementsNum();
  int num2_ = input2->ElementsNum();

  input_data = reinterpret_cast<bool *>(input->MutableData());
  input_data1 = reinterpret_cast<float *>(input1->MutableData());
  input_data2 = reinterpret_cast<float *>(input2->MutableData());
  output_data = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  int num_max = num > num1_ ? num : (num1_ > num2_ ? num1_ : num2_);
  where_param_->num_ = num;
  where_param_->num1_ = num1_;
  where_param_->num2_ = num2_;
  where_param_->number_ = num_max;

  if (((num != 1) && (num != num_max)) || ((num1_ != 1) && (num1_ != num_max)) ||
      ((num2_ != 1) && (num2_ != num_max))) {
    MS_LOG(ERROR) << "The length of three inputs are not equal to 1 or length of output, which is unacceptable";
    return RET_ERROR;
  }
  if (num_max <= 0) {
    MS_LOG(ERROR) << "Error, inputs' length are zero !!!";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, WhereRun, this, where_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WhereDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Where, LiteKernelCreator<WhereCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Where, LiteKernelCreator<WhereCPUKernel>)
}  // namespace mindspore::kernel
