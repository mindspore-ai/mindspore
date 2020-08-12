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

#include "src/runtime/kernel/arm/int8/div_int8.h"
#include <limits>
#include <algorithm>
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Div;

namespace mindspore::kernel {

int DivInt8CPUKernel::Init() {
  lite::tensor::Tensor *input0 = in_tensors_.at(0);
  lite::tensor::Tensor *input1 = in_tensors_.at(1);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input0);
  MS_ASSERT(input1);
  MS_ASSERT(output);

  broadcast_ = input0->ElementsNum() != input1->ElementsNum();

  param_.in0_args_.scale_ = input0->GetQuantParams().front().scale;
  param_.in0_args_.zp_ = -input0->GetQuantParams().front().zeroPoint;
  param_.in1_args_.scale_ = input1->GetQuantParams().front().scale;
  param_.in1_args_.zp_ = -input1->GetQuantParams().front().zeroPoint;
  param_.out_args_.scale_ = output->GetQuantParams().front().scale;
  param_.out_args_.zp_ = output->GetQuantParams().front().zeroPoint;

  const double real_multiplier = param_.in0_args_.scale_ / (param_.in1_args_.scale_ * param_.out_args_.scale_);

  QuantizeMultiplier(real_multiplier, &param_.output_multiplier_, &param_.output_shift_);

  param_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  param_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DivInt8CPUKernel::ReSize() {
  if (broadcast_) {
    if (tile0_data_ != nullptr) {
      if (context_ != nullptr && context_->allocator != nullptr) {
        context_->allocator->Free(tile0_data_);
      } else {
        free(tile0_data_);
      }
    }
    if (tile1_data_ != nullptr) {
      if (context_ != nullptr && context_->allocator != nullptr) {
        context_->allocator->Free(tile1_data_);
      } else {
        free(tile1_data_);
      }
    }

    if (context_ != nullptr && context_->allocator != nullptr) {
      tile0_data_ = static_cast<int8_t *>(context_->allocator->Malloc(out_tensors_.at(0)->Size()));
      tile1_data_ = static_cast<int8_t *>(context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    } else {
      tile0_data_ = static_cast<int8_t *>(malloc(sizeof(int8_t) * out_tensors_.at(0)->Size()));
      tile1_data_ = static_cast<int8_t *>(malloc(sizeof(int8_t) * out_tensors_.at(0)->Size()));
    }

    if (tile0_data_ == nullptr || tile1_data_ == nullptr) {
      if (tile0_data_ != nullptr) {
        if (context_ != nullptr && context_->allocator != nullptr) {
          context_->allocator->Free(tile0_data_);
        } else {
          free(tile0_data_);
        }
      }
      if (tile1_data_ != nullptr) {
        if (context_ != nullptr && context_->allocator != nullptr) {
          context_->allocator->Free(tile1_data_);
        } else {
          free(tile1_data_);
        }
      }
      MS_LOG(ERROR) << "malloc memroy fail!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int DivInt8CPUKernel::DoExecute(int task_id) {
  auto input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->Data());
  auto input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->Data());
  auto output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->Data());
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(op_parameter_->thread_num_ != 0);
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);

  auto ret = RET_OK;
  if (broadcast_) {
    ret = DivInt8(tile0_data_ + task_id * count, tile1_data_ + task_id * count, output_data_ + task_id * count, count,
                  &param_);
  } else {
    ret = DivInt8(input0_data_ + task_id * count, input1_data_ + task_id * count, output_data_ + task_id * count, count,
                  &param_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Divint8 function error error_code[" << ret << "]";
  }
  return ret;
}

int DivInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto div_kernel = reinterpret_cast<DivInt8CPUKernel *>(cdata);
  auto ret = div_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DivInt8 DoExecute error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int DivInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }

  if (broadcast_) {
    ArithmeticParameter tile_para = {0};
    tile_para.ndim_ = out_tensors_.at(0)->shape().size();
    for (size_t i = 0; i < tile_para.ndim_; i++) {
      tile_para.in_shape0_[i] = in_tensors_.at(0)->DimensionSize(i);
      tile_para.in_shape1_[i] = in_tensors_.at(1)->DimensionSize(i);
      tile_para.out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
    TileDimensionsUint8(static_cast<uint8_t *>(in_tensors_.at(0)->Data()),
                        static_cast<uint8_t *>(in_tensors_.at(1)->Data()), reinterpret_cast<uint8_t *>(tile0_data_),
                        reinterpret_cast<uint8_t *>(tile1_data_), &tile_para);
  }
  ret = LiteBackendParallelLaunch(DivInt8Run, this, op_parameter_->thread_num_);

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DivInt8Run function error error_code[" << ret << "]";
  }
  return ret;
}

kernel::LiteKernel *CpuDivInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                            const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::Context *ctx, const KernelKey &desc,
                                            const lite::Primitive *primitive) {
  if (parameter == nullptr || ctx == nullptr) {
    MS_LOG(ERROR) << "parameter or ctx is nullptr";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Div);
  auto *kernel = new (std::nothrow) DivInt8CPUKernel(parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Div, CpuDivInt8KernelCreator)
}  // namespace mindspore::kernel
