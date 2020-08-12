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

#include "src/runtime/kernel/arm/int8/concat_int8.h"
#include <limits>
#include "src/runtime/kernel/arm/nnacl/int8/concat_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int ConcatInt8CPUKernel::Init() {
  ConcatBaseCPUKernel::Init();
  auto input_num = in_tensors_.size();
  concat_param_->quant_arg_.in_args_ =
    reinterpret_cast<QuantArg *>(ctx_->allocator->Malloc(sizeof(QuantArg) * input_num));
  if (concat_param_->quant_arg_.in_args_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->in_quant_args_.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = in_tensors_.at(i);
    auto quant_args = input_tensor->GetQuantParams();
    concat_param_->quant_arg_.in_args_[i].scale_ = quant_args.front().scale;
    concat_param_->quant_arg_.in_args_[i].zp_ = quant_args.front().zeroPoint;
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto quant_args = output_tensor->GetQuantParams();
  concat_param_->quant_arg_.out_args_.scale_ = quant_args.front().scale;
  concat_param_->quant_arg_.out_args_.zp_ = quant_args.front().zeroPoint;

  concat_param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  concat_param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConcatInt8CPUKernel::ReSize() {
  auto ret = ConcatBaseCPUKernel::ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  if (concat_param_->input_shapes_ != nullptr) {
    ctx_->allocator->Free(concat_param_->input_shapes_);
  }
  auto input_num = in_tensors_.size();
  concat_param_->input_num_ = input_num;
  concat_param_->input_shapes_ = reinterpret_cast<const int **>(ctx_->allocator->Malloc(sizeof(int *) * input_num));
  for (size_t i = 0; i < input_num; i++) {
    concat_param_->input_shapes_[i] = reinterpret_cast<const int *>(in_tensors_.at(i)->shape().data());
  }

  before_axis_size = 1;
  for (int i = 0; i < axis_; i++) {
    before_axis_size *= out_tensors_.at(kOutputIndex)->DimensionSize(i);
  }

  int64_t after_axis_size = 1;
  auto output_tensor = out_tensors_.at(kOutputIndex);
  int output_dim = output_tensor->shape().size();
  concat_param_->output_shapes_ = output_tensor->shape().data();
  for (size_t i = axis_ + 1; i < output_dim; i++) {
    after_axis_size *= concat_param_->output_shapes_[i];
  }
  concat_param_->after_axis_size = after_axis_size;
  return RET_OK;
}

int ConcatInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }

  auto input_num = concat_param_->input_num_;
  count_unit_ = thread_count_ > 1 ? UP_DIV(before_axis_size, thread_count_) : before_axis_size;
  concat_param_->count_unit_ = count_unit_;
  input_data_ = reinterpret_cast<int8_t **>(ctx_->allocator->Malloc(sizeof(int8_t *) * input_num));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: inputs_array.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < input_num; i++) {
    input_data_[i] = static_cast<int8_t *>(in_tensors_.at(i)->Data());
  }
  output_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());

  ret = LiteBackendParallelLaunch(ConcatInt8Run, this, thread_count_);

  ctx_->allocator->Free(input_data_);
  ctx_->allocator->Free(concat_param_->input_shapes_);
  ctx_->allocator->Free(concat_param_->quant_arg_.in_args_);

  return ret;
}

int ConcatInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto concat = reinterpret_cast<ConcatInt8CPUKernel *>(cdata);
  concat->DoExecute(task_id);
  return lite::RET_OK;
}

int ConcatInt8CPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(before_axis_size - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  Int8Concat(input_data_, output_data_, concat_param_, axis_, real_dst_count, task_id);
  return lite::RET_OK;
}
}  // namespace mindspore::kernel
