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

#include "src/runtime/kernel/arm/int8/prelu_int8.h"
#include <limits>
#include "src/runtime/kernel/arm/nnacl/int8/prelu_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Prelu;

namespace mindspore::kernel {
int PreluInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  PreluBaseCPUKernel::Init();
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  quant_prelu_parm_->quant_arg.in_args_.scale_ = in_quant_args.front().scale;
  quant_prelu_parm_->quant_arg.in_args_.zp_ = in_quant_args.front().zeroPoint;
  auto input_dim = input_tensor->shape().size();
  MS_ASSERT(input_dim <= CROP_OFFSET_MAX_SIZE);
  quant_prelu_parm_->input_dim_ = input_dim;
  quant_prelu_parm_->element_num = in_tensors_[0]->Size();
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  quant_prelu_parm_->quant_arg.out_args_.scale_ = out_quant_args.front().scale;
  quant_prelu_parm_->quant_arg.out_args_.zp_ = out_quant_args.front().zeroPoint;
  quant_prelu_parm_->in_shape_ = input_tensor->shape().data();
  quant_prelu_parm_->out_shape_ = out_tensor->shape().data();
  quant_prelu_parm_->quant_arg.output_activation_max_ = std::numeric_limits<int8_t>::max();
  quant_prelu_parm_->quant_arg.output_activation_min_ = std::numeric_limits<int8_t>::min();
  return RET_OK;
}

int PreluInt8CPUKernel::ReSize() { return 0; }

int PreluInt8CPUKernel::Run() {
  auto ret = LiteBackendParallelLaunch(PreluInt8Run, this, quant_prelu_parm_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunPreluParam failed. errorcode: ";
  }
  return RET_OK;
}
int PreluInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto prelu = reinterpret_cast<PreluInt8CPUKernel *>(cdata);
  prelu->DoExecute(task_id);
  return RET_OK;
}

int PreluInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->Data());
  prelu(input_data, output_data, quant_prelu_parm_, task_id);
  return RET_OK;
}

}  // namespace mindspore::kernel
