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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {
int ConcatInt8CPUKernel::Init() {
  concat_param_->input_shapes_ = nullptr;
  auto input_num = in_tensors_.size();
  input_data_ = reinterpret_cast<int8_t **>(malloc(sizeof(int8_t *) * input_num));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: inputs_array.";
    return RET_ERROR;
  }
  concat_param_->quant_arg_.in_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg) * input_num));
  if (concat_param_->quant_arg_.in_args_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->in_quant_args_.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = in_tensors_.at(i);
    auto quant_args = input_tensor->quant_params();
    concat_param_->quant_arg_.in_args_[i].scale_ = quant_args.front().scale;
    concat_param_->quant_arg_.in_args_[i].zp_ = quant_args.front().zeroPoint;
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto quant_args = output_tensor->quant_params();
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
  concat_param_->axis_ =
    concat_param_->axis_ >= 0 ? concat_param_->axis_ : in_tensors_.front()->shape().size() + concat_param_->axis_;

  auto input_num = in_tensors_.size();
  concat_param_->input_num_ = input_num;
  concat_param_->input_shapes_ = reinterpret_cast<int **>(malloc(sizeof(int *) * input_num));
  if (concat_param_->input_shapes_ == nullptr) {
    MS_LOG(ERROR) << "malloc concat_param_->input_shapes_ failed.";
    return RET_ERROR;
  }
  for (size_t i = 0; i < input_num; i++) {
    auto in_shape = in_tensors_.at(i)->shape();
    concat_param_->input_shapes_[i] = reinterpret_cast<int *>(malloc(in_shape.size() * sizeof(int)));
    if (concat_param_->input_shapes_[i] == nullptr) {
      MS_LOG(ERROR) << "malloc concat_param_->input_shapes_[" << i << "]"
                    << " failed.";
      return RET_ERROR;
    }
    memcpy(reinterpret_cast<void *>(concat_param_->input_shapes_[i]), in_shape.data(), sizeof(int) * in_shape.size());
  }

  before_axis_size = 1;
  for (int i = 0; i < concat_param_->axis_; i++) {
    before_axis_size *= out_tensors_.at(kOutputIndex)->DimensionSize(i);
  }

  int64_t after_axis_size = 1;
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto out_shape = output_tensor->shape();
  size_t output_dim = out_shape.size();
  concat_param_->output_shapes_ = reinterpret_cast<int *>(malloc(output_dim * sizeof(int)));
  if (concat_param_->output_shapes_ == nullptr) {
    MS_LOG(ERROR) << "malloc concat_param_->output_shapes_ failed.";
    return RET_ERROR;
  }
  memcpy(reinterpret_cast<void *>(concat_param_->output_shapes_), output_tensor->shape().data(),
         sizeof(int) * output_dim);

  for (size_t i = concat_param_->axis_ + 1; i < output_dim; i++) {
    after_axis_size *= concat_param_->output_shapes_[i];
  }
  concat_param_->after_axis_size = after_axis_size;
  return RET_OK;
}

int ConcatInt8CPUKernel::Run() {
  auto input_num = concat_param_->input_num_;
  count_unit_ =
    op_parameter_->thread_num_ > 1 ? UP_DIV(before_axis_size, op_parameter_->thread_num_) : before_axis_size;
  concat_param_->count_unit_ = count_unit_;

  for (int i = 0; i < input_num; i++) {
    input_data_[i] = static_cast<int8_t *>(in_tensors_.at(i)->MutableData());
  }
  output_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());

  auto ret = ParallelLaunch(this->context_->thread_pool_, ConcatInt8Run, this, op_parameter_->thread_num_);

  return ret;
}

int ConcatInt8Run(void *cdata, int task_id) {
  auto concat = reinterpret_cast<ConcatInt8CPUKernel *>(cdata);
  concat->DoExecute(task_id);
  return lite::RET_OK;
}

int ConcatInt8CPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(before_axis_size - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  Int8Concat(input_data_, output_data_, concat_param_, concat_param_->axis_, real_dst_count, task_id);
  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Concat, LiteKernelCreator<ConcatInt8CPUKernel>)
}  // namespace mindspore::kernel
