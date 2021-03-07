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

#include "src/runtime/kernel/arm/int8/add_int8.h"
#include "nnacl/int8/quantize.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore::kernel {
int QuantizedAddCPUKernel::Init() {
  auto *input0 = in_tensors_.at(0);
  auto *input1 = in_tensors_.at(1);
  auto *output = out_tensors_.at(0);

  para_.in0_args_.zp_ = input0->quant_params().front().zeroPoint * -1;
  para_.in1_args_.zp_ = input1->quant_params().front().zeroPoint * -1;
  para_.out_zp_ = output->quant_params().front().zeroPoint;

  const double in0_scale = input0->quant_params().front().scale;
  const double in1_scale = input1->quant_params().front().scale;
  const double out_scale = output->quant_params().front().scale;

  para_.left_shift_ = 20;
  const double twice_max_input_scale = 2 * std::max(in0_scale, in1_scale);
  const double in0_multiplier = in0_scale / twice_max_input_scale;
  const double in1_multiplier = in1_scale / twice_max_input_scale;
  const double out_multiplier = twice_max_input_scale / ((1 << para_.left_shift_) * out_scale);

  QuantizeMultiplierSmallerThanOne(in0_multiplier, &para_.in0_args_.multiplier_, &para_.in0_args_.left_shift_);
  QuantizeMultiplierSmallerThanOne(in1_multiplier, &para_.in1_args_.multiplier_, &para_.in1_args_.left_shift_);
  QuantizeMultiplierSmallerThanOne(out_multiplier, &para_.out_multiplier_, &para_.out_left_shift_);

  para_.in0_args_.right_shift_ = -para_.in0_args_.left_shift_ > 0 ? 0 : para_.in0_args_.left_shift_;
  para_.in1_args_.right_shift_ = -para_.in1_args_.left_shift_ > 0 ? 0 : para_.in1_args_.left_shift_;
  para_.out_right_shift_ = -para_.out_left_shift_ > 0 ? 0 : para_.out_left_shift_;

  para_.in0_args_.left_shift_ = -para_.in0_args_.left_shift_ > 0 ? -para_.in0_args_.left_shift_ : 0;
  para_.in1_args_.left_shift_ = -para_.in1_args_.left_shift_ > 0 ? -para_.in1_args_.left_shift_ : 0;
  para_.out_left_shift_ = -para_.out_left_shift_ > 0 ? -para_.out_left_shift_ : 0;

  auto act = arith_para_->activation_type_;
  CalculateActivationRangeQuantized(act == ActType_Relu, act == ActType_Relu6, para_.out_zp_,
                                    static_cast<float>(out_scale), &para_.min_, &para_.max_);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantizedAddCPUKernel::ReSize() {
  auto *input0 = in_tensors_.at(0);
  auto *input1 = in_tensors_.at(1);
  auto *output = out_tensors_.at(0);
  support_opt_add_ = (input0->ElementsNum() == 1) || (input1->ElementsNum() == 1);
  if (support_opt_add_) {
    arith_para_->broadcasting_ = false;
  }

  elements_num_ = output->ElementsNum();
  thread_count_ = MSMIN(elements_num_, op_parameter_->thread_num_);

  arith_para_->in_elements_num0_ = in_tensors_[0]->ElementsNum();
  arith_para_->in_elements_num1_ = in_tensors_[1]->ElementsNum();
  arith_para_->out_elements_num_ = out_tensors_[0]->ElementsNum();

  for (size_t i = 0; i < in_tensors_[0]->shape().size(); i++) {
    if (arith_para_->in_shape0_[i] == -1) {
      memcpy(arith_para_->in_shape0_, input0->shape().data(), input0->shape().size() * sizeof(int));
      break;
    }
  }
  for (size_t i = 0; i < in_tensors_[1]->shape().size(); i++) {
    if (arith_para_->in_shape1_[i] == -1) {
      memcpy(arith_para_->in_shape1_, input1->shape().data(), input1->shape().size() * sizeof(int));
      break;
    }
  }
  for (size_t i = 0; i < out_tensors_[0]->shape().size(); i++) {
    if (arith_para_->out_shape_[i] == -1) {
      memcpy(arith_para_->out_shape_, output->shape().data(), output->shape().size() * sizeof(int));
      break;
    }
  }

  if (arith_para_->broadcasting_) {
    size_t break_pos_ = 0;
    for (auto i = arith_para_->ndim_ - 1; i >= 0; --i) {
      if (arith_para_->in_shape0_[i] != arith_para_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
    }
    in_size_ = 1;
    out_size_ = 1;
    for (size_t i = 0; i < arith_para_->ndim_; i++) {
      if (i > break_pos_) {
        in_size_ *= arith_para_->out_shape_[i];
      } else {
        out_size_ *= arith_para_->out_shape_[i];
      }
    }

    ComputeStrides(arith_para_->in_shape0_, arith_para_->in_strides0_, arith_para_->ndim_);
    ComputeStrides(arith_para_->in_shape1_, arith_para_->in_strides1_, arith_para_->ndim_);
    ComputeStrides(arith_para_->out_shape_, arith_para_->out_strides_, arith_para_->ndim_);
  }
  return RET_OK;
}

int AddInt8Run(void *cdata, int task_id) {
  auto add = reinterpret_cast<QuantizedAddCPUKernel *>(cdata);
  add->DoExecute(task_id);
  return RET_OK;
}

void QuantizedAddCPUKernel::BroadcastRun(int task_id) {
  int stride = UP_DIV(out_size_, thread_count_);
  int real_out_count = MSMIN(stride, out_size_ - stride * task_id);
  if (real_out_count <= 0) {
    return;
  }
  int8_t *cur_in0 = nullptr;
  int8_t *cur_in1 = nullptr;
  int8_t *cur_out = nullptr;
  for (int i = 0; i < real_out_count; i++) {
    if (arith_para_->in_elements_num0_ == arith_para_->out_elements_num_) {
      cur_in0 = input0_data_ + task_id * stride * in_size_ + i * in_size_;
      cur_in1 = input1_data_;
      cur_out = output_data_ + task_id * stride * in_size_ + i * in_size_;
    } else {
      cur_in0 = input0_data_;
      cur_in1 = input1_data_ + task_id * stride * in_size_ + i * in_size_;
      cur_out = output_data_ + task_id * stride * in_size_ + i * in_size_;
    }
#ifdef ENABLE_AVX
    AddInt8_AVX2(cur_in0, cur_in1, cur_out, in_size_, &para_);
#else
    AddInt8(cur_in0, cur_in1, cur_out, in_size_, &para_);
#endif
  }
  return;
}

int QuantizedAddCPUKernel::DoExecute(int task_id) {
  /* need broadcast */
  if (arith_para_->broadcasting_) {
    BroadcastRun(task_id);
    return RET_OK;
  }

  /* no need broadcast */
  int stride = UP_DIV(elements_num_, thread_count_);
  int rest_count = elements_num_ - task_id * stride;
  int real_count = MSMIN(stride, rest_count);
  if (real_count <= 0) {
    return RET_OK;
  }
  int8_t *cur_in0 = input0_data_ + stride * task_id;
  int8_t *cur_in1 = input1_data_ + stride * task_id;
  int8_t *cur_out = output_data_ + stride * task_id;
  if (support_opt_add_) {
    int8_t *ptr_in = arith_para_->in_elements_num0_ == 1 ? cur_in1 : cur_in0;
    int8_t element_in = arith_para_->in_elements_num0_ == 1 ? input0_data_[0] : input1_data_[0];
    AddQuantQrgs *ptr_args = arith_para_->in_elements_num0_ == 1 ? &para_.in1_args_ : &para_.in0_args_;
    AddQuantQrgs *ele_args = arith_para_->in_elements_num0_ == 1 ? &para_.in0_args_ : &para_.in1_args_;
#ifdef ENABLE_AVX
    AddOptInt8_AVX2(ptr_in, element_in, cur_out, rest_count, &para_, ptr_args, ele_args);
#else
    AddOptInt8(ptr_in, element_in, cur_out, rest_count, &para_, ptr_args, ele_args);
#endif
  } else {
#ifdef ENABLE_AVX
    AddInt8_AVX2(cur_in0, cur_in1, cur_out, rest_count, &para_);
#else
    AddInt8(cur_in0, cur_in1, cur_out, rest_count, &para_);
#endif
  }

  return RET_OK;
}

int QuantizedAddCPUKernel::Run() {
  input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->data_c());
  input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->data_c());
  output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->data_c());

  ParallelLaunch(this->context_->thread_pool_, AddInt8Run, this, thread_count_);

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_AddFusion, LiteKernelCreator<QuantizedAddCPUKernel>)
}  // namespace mindspore::kernel
