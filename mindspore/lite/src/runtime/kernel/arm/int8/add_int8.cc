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
#include <limits>
#include <algorithm>
#include "nnacl/arithmetic_common.h"
#include "nnacl/quantization/quantize.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Add;

namespace mindspore::kernel {
int QuantizedAddCPUKernel::Init() {
  auto *input0 = in_tensors_.at(0);
  auto *input1 = in_tensors_.at(1);
  auto *output = out_tensors_.at(0);
  auto act = arith_para_->activation_type_;

  para_.in0_zp_ = input0->GetQuantParams().front().zeroPoint * -1;
  para_.in1_zp_ = input1->GetQuantParams().front().zeroPoint * -1;
  para_.out_zp_ = output->GetQuantParams().front().zeroPoint;

  const double in0_scale = input0->GetQuantParams().front().scale;
  const double in1_scale = input1->GetQuantParams().front().scale;
  const double out_scale = output->GetQuantParams().front().scale;

  para_.left_shift_ = 20;
  const double twice_max_input_scale = 2 * std::max(in0_scale, in1_scale);
  const double in0_multiplier = in0_scale / twice_max_input_scale;
  const double in1_multiplier = in1_scale / twice_max_input_scale;
  const double out_multiplier = twice_max_input_scale / ((1 << para_.left_shift_) * out_scale);

  QuantizeMultiplierSmallerThanOne(in0_multiplier, &para_.in0_multiplier_, &para_.in0_left_shift_);
  QuantizeMultiplierSmallerThanOne(in1_multiplier, &para_.in1_multiplier_, &para_.in1_left_shift_);
  QuantizeMultiplierSmallerThanOne(out_multiplier, &para_.out_multiplier_, &para_.out_left_shift_);

  para_.in0_right_shift_ = -para_.in0_left_shift_ > 0 ? 0 : para_.in0_left_shift_;
  para_.in1_right_shift_ = -para_.in1_left_shift_ > 0 ? 0 : para_.in1_left_shift_;
  para_.out_right_shift_ = -para_.out_left_shift_ > 0 ? 0 : para_.out_left_shift_;

  para_.in0_left_shift_ = -para_.in0_left_shift_ > 0 ? -para_.in0_left_shift_ : 0;
  para_.in1_left_shift_ = -para_.in1_left_shift_ > 0 ? -para_.in1_left_shift_ : 0;
  para_.out_left_shift_ = -para_.out_left_shift_ > 0 ? -para_.out_left_shift_ : 0;

  CalculateActivationRangeQuantized(act == ActType_Relu, act == ActType_Relu6, 0, 1, &para_.min_, &para_.max_);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int QuantizedAddCPUKernel::ReSize() {
  elements_num_ = out_tensors_.at(0)->ElementsNum();
  arith_para_->broadcasting_ = in_tensors_.at(0)->ElementsNum() != in_tensors_.at(1)->ElementsNum();

  thread_count_ = MSMIN(elements_num_, op_parameter_->thread_num_);
  thread_stride_ = UP_DIV(elements_num_, thread_count_);
  return RET_OK;
}

int AddInt8Run(void *cdata, int task_id) {
  auto add = reinterpret_cast<QuantizedAddCPUKernel *>(cdata);
  add->DoExecute(task_id);
  return RET_OK;
}

int QuantizedAddCPUKernel::DoExecute(int task_id) {
  int rest_count = elements_num_ - task_id * thread_stride_;
  int real_count = MSMIN(thread_stride_, rest_count);
  if (real_count <= 0) {
    return RET_OK;
  }

  int8_t *cur_input0_data = input0_data_ + task_id * thread_stride_;
  int8_t *cur_input1_data = input1_data_ + task_id * thread_stride_;
  int8_t *cur_output_data = output_data_ + task_id * thread_stride_;

  AddInt8(cur_input0_data, cur_input1_data, cur_output_data, real_count, &para_);
  return RET_OK;
}

int QuantizedAddCPUKernel::Run() {
  int8_t *src_in0 = static_cast<int8_t *>(in_tensors_.at(0)->data_c());
  int8_t *src_in1 = static_cast<int8_t *>(in_tensors_.at(1)->data_c());
  output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->data_c());

  if (arith_para_->broadcasting_) {
    input0_data_ = static_cast<int8_t *>(context_->allocator->Malloc(elements_num_ * sizeof(int8_t)));
    if (input0_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc input0_data_  failed.";
      return RET_ERROR;
    }
    input1_data_ = static_cast<int8_t *>(context_->allocator->Malloc(elements_num_ * sizeof(int8_t)));
    if (input1_data_ == nullptr) {
      context_->allocator->Free(input0_data_);
      input0_data_ = nullptr;
      MS_LOG(ERROR) << "malloc input1_data_  failed.";
      return RET_ERROR;
    }

    TileDimensionsInt8(src_in0, src_in1, input0_data_, input1_data_, arith_para_);
    auto ret = ParallelLaunch(context_->thread_pool_, AddInt8Run, this, thread_count_);

    context_->allocator->Free(input0_data_);
    context_->allocator->Free(input1_data_);
    input0_data_ = nullptr;
    input1_data_ = nullptr;
    return ret;
  }

  input0_data_ = src_in0;
  input1_data_ = src_in1;
  auto ret = ParallelLaunch(this->context_->thread_pool_, AddInt8Run, this, thread_count_);
  return ret;
}

kernel::LiteKernel *CpuAddInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::InnerContext *ctx, const KernelKey &desc,
                                            const mindspore::lite::PrimitiveC *primitive) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "ctx is nullptr";
    free(parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Add);
  auto *kernel = new (std::nothrow) QuantizedAddCPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Add, CpuAddInt8KernelCreator)
}  // namespace mindspore::kernel
