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

#include "src/litert/kernel/cpu/int8/mul_int8.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MulFusion;

namespace mindspore::kernel {
MulInt8CPUKernel::~MulInt8CPUKernel() {
  if (quant_args_ != nullptr) {
    free(quant_args_);
    quant_args_ = nullptr;
  }
}

int MulInt8CPUKernel::Prepare() {
  lite::Tensor *input0 = in_tensors_.at(0);
  lite::Tensor *input1 = in_tensors_.at(1);
  lite::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input0);
  MS_ASSERT(input1);
  MS_ASSERT(output);

  quant_args_ = reinterpret_cast<MulQuantArg *>(malloc(sizeof(MulQuantArg)));
  if (quant_args_ == nullptr) {
    MS_LOG(ERROR) << "Malloc MulQuantArg for Mul int8 op failed!";
    return RET_ERROR;
  }
  const auto &input0_params = input0->quant_params();
  const auto &input1_params = input1->quant_params();
  const auto &output_params = output->quant_params();
  MS_CHECK_TRUE_MSG(!input0_params.empty(), RET_ERROR, "Input 0 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!input1_params.empty(), RET_ERROR, "Input 1 quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!output_params.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_args_->in_quant_args_[0].scale_ = static_cast<float>(input0_params.front().scale);
  quant_args_->in_quant_args_[0].zp_ = input0_params.front().zeroPoint * -1;
  quant_args_->in_quant_args_[1].scale_ = static_cast<float>(input1_params.front().scale);
  quant_args_->in_quant_args_[1].zp_ = input1_params.front().zeroPoint * -1;
  quant_args_->out_quant_arg_.scale_ = static_cast<float>(output_params.front().scale);
  quant_args_->out_quant_arg_.zp_ = output_params.front().zeroPoint;
  quant_args_->output_activation_max_ = std::numeric_limits<int8_t>::max();
  quant_args_->output_activation_min_ = std::numeric_limits<int8_t>::min();

  const double real_multiplier = (quant_args_->in_quant_args_[0].scale_ * quant_args_->in_quant_args_[1].scale_) /
                                 quant_args_->out_quant_arg_.scale_;

  int right_shift = 0;
  QuantizeMultiplierSmallerThanOne(real_multiplier, &quant_args_->output_multiplier_, &right_shift);

  quant_args_->shift_left_ = right_shift < 0 ? -right_shift : 0;
  quant_args_->shift_right_ = right_shift > 0 ? right_shift : 0;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void MulInt8CPUKernel::CheckSameShapeSize(std::vector<int> in_tensor0_shape, std::vector<int> in_tensor1_shape) {
  bool condition1 = in_tensor0_shape[kNHWC_N] == in_tensor1_shape[kNHWC_N];
  bool condition2 = in_tensor0_shape[kNHWC_H] == 1;
  bool condition3 = in_tensor0_shape[kNHWC_W] == 1;
  bool condition4 = in_tensor0_shape[kNHWC_C] == in_tensor1_shape[kNHWC_C];
  bool condition5 = in_tensor1_shape[kNHWC_H] == 1;
  bool condition6 = in_tensor1_shape[kNHWC_W] == 1;
  if (condition1 && condition2 && condition3 && condition4) {
    fast_hw_broadcast_ = true;
  } else if (condition1 && condition4 && condition5 && condition6) {
    fast_hw_broadcast_ = true;
    input1_hw_broadcast_ = true;
  }
}

void MulInt8CPUKernel::CheckIfFastImpl() {
  auto in_tensor0 = in_tensors_.at(0);
  auto in_tensor1 = in_tensors_.at(1);
  MS_CHECK_TRUE_RET_VOID(in_tensors_.at(0)->ElementsNum() > 0);
  MS_CHECK_TRUE_RET_VOID(in_tensors_.at(1)->ElementsNum() > 0);
  if (in_tensor0->ElementsNum() != in_tensor1->ElementsNum()) {
    if (in_tensor0->shape().size() == COMM_SHAPE_SIZE && in_tensor1->shape().size() == COMM_SHAPE_SIZE) {
      CheckSameShapeSize(in_tensor0->shape(), in_tensor1->shape());
    } else if (in_tensor0->shape().size() == 1 && in_tensor1->shape().size() == COMM_SHAPE_SIZE) {
      if (in_tensor0->ElementsNum() == in_tensor1->shape()[kNHWC_C]) {
        fast_hw_broadcast_ = true;
      }
    } else if (in_tensor0->shape().size() == COMM_SHAPE_SIZE && in_tensor1->shape().size() == 1) {
      if (in_tensor1->ElementsNum() == in_tensor0->shape()[kNHWC_C]) {
        fast_hw_broadcast_ = true;
        input1_hw_broadcast_ = true;
      }
    }
  }
}

int MulInt8CPUKernel::ReSize() {
  size_t input0_size = in_tensors_.at(0)->shape().size();
  size_t input1_size = in_tensors_.at(1)->shape().size();
  size_t output_size = out_tensors_.at(0)->shape().size();
  tile_para->ndim_ = output_size;

  if (input0_size == input1_size) {
    for (size_t i = 0; i < output_size; i++) {
      tile_para->in_shape0_[i] = in_tensors_.at(0)->DimensionSize(i);
      tile_para->in_shape1_[i] = in_tensors_.at(1)->DimensionSize(i);
      tile_para->out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
  } else if (input0_size < input1_size) {
    auto fill_dim_num = input1_size - input0_size;
    int j = 0;
    for (size_t i = 0; i < output_size; i++) {
      if (i < fill_dim_num) {
        tile_para->in_shape0_[i] = 1;
      } else {
        tile_para->in_shape0_[i] = in_tensors_.at(0)->DimensionSize(j++);
      }
      tile_para->in_shape1_[i] = in_tensors_.at(1)->DimensionSize(i);
      tile_para->out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
  } else {
    auto fill_dim_num = input0_size - input1_size;
    int j = 0;
    for (size_t i = 0; i < output_size; i++) {
      tile_para->in_shape0_[i] = in_tensors_.at(0)->DimensionSize(i);
      if (i < fill_dim_num) {
        tile_para->in_shape1_[i] = 1;
      } else {
        tile_para->in_shape1_[i] = in_tensors_.at(1)->DimensionSize(j++);
      }
      tile_para->out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
  }
  return RET_OK;
}

int MulInt8CPUKernel::Run() {
  input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input0_data_);
  input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  MS_ASSERT(input1_data_);
  output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_data_);

  CheckIfFastImpl();
  // can implement fast broadcast mul
  if (fast_hw_broadcast_) {
    elements_num_ = out_tensors_.front()->Batch() * out_tensors_.front()->Height() * out_tensors_.front()->Width();
    count_unit_ = thread_count_ > 1 ? UP_DIV(elements_num_, thread_count_) : elements_num_;
    return ParallelLaunch(this->ms_context_, FastHWBroadcastMulInt8Run, this, thread_count_);
  }

  elements_num_ = out_tensors_.at(0)->ElementsNum();
  MS_CHECK_GT(elements_num_, 0, RET_ERROR);
  count_unit_ = thread_count_ > 1 ? UP_DIV(elements_num_, thread_count_) : elements_num_;
  int ret = RET_ERROR;
  if (in_tensors_.at(0)->ElementsNum() != in_tensors_.at(1)->ElementsNum()) {
    MS_CHECK_GT(in_tensors_.at(0)->ElementsNum(), 0, RET_ERROR);
    MS_CHECK_GT(in_tensors_.at(1)->ElementsNum(), 0, RET_ERROR);
    MS_CHECK_GT(out_tensors_.at(0)->Size(), 0, RET_ERROR);
    input0_data_ = static_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (input0_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc input0_data_  failed.";
      return RET_ERROR;
    }
    input1_data_ = static_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (input1_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc input1_data_  failed.";
      ctx_->allocator->Free(input0_data_);
      return RET_ERROR;
    }
    TileDimensionsInt8(static_cast<int8_t *>(in_tensors_.at(0)->MutableData()),
                       static_cast<int8_t *>(in_tensors_.at(1)->MutableData()), input0_data_, input1_data_, tile_para);
    ret = ParallelLaunch(this->ms_context_, MulInt8Run, this, thread_count_);
    ctx_->allocator->Free(input0_data_);
    ctx_->allocator->Free(input1_data_);
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, MulInt8Run, this, thread_count_);
  return ret;
}

int FastHWBroadcastMulInt8Run(void *cdata, int task_id, float, float) {
  auto mul = reinterpret_cast<MulInt8CPUKernel *>(cdata);
  mul->FastDoExecute(task_id);
  return lite::RET_OK;
}

int MulInt8Run(void *cdata, int task_id, float, float) {
  auto mul = reinterpret_cast<MulInt8CPUKernel *>(cdata);
  mul->DoExecute(task_id);
  return lite::RET_OK;
}

void MulInt8CPUKernel::FastDoExecute(int task_id) {
  int depth = out_tensors_.front()->Channel();
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return;
  }
  int8_t *cur_input0_data = input0_data_;
  int8_t *cur_input1_data = input1_data_ + task_id * count_unit_ * depth;
  int8_t *cur_output_data = output_data_ + task_id * count_unit_ * depth;
  if (input1_hw_broadcast_) {
    cur_input0_data = input1_data_;
    cur_input1_data = input0_data_ + task_id * count_unit_ * depth;
  }
  FastMul(cur_input0_data, cur_input1_data, cur_output_data, depth, real_dst_count, input1_hw_broadcast_, quant_args_);
}

void MulInt8CPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return;
  }
  int8_t *cur_input0_data = input0_data_ + task_id * count_unit_;
  int8_t *cur_input1_data = input1_data_ + task_id * count_unit_;
  int8_t *cur_output_data = output_data_ + task_id * count_unit_;

  Mul(cur_input0_data, cur_input1_data, cur_output_data, real_dst_count, quant_args_);
  return;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MulFusion, LiteKernelCreator<MulInt8CPUKernel>)
}  // namespace mindspore::kernel
