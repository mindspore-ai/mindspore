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

#include "src/runtime/kernel/arm/int8/mul_int8.h"
#include <limits>
#include <algorithm>
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"
#include "src/runtime/kernel/arm/nnacl/int8/mul_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Mul;

namespace mindspore::kernel {
int MulInt8CPUKernel::Init() {
  lite::tensor::Tensor *input0 = in_tensors_.at(0);
  lite::tensor::Tensor *input1 = in_tensors_.at(1);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input0);
  MS_ASSERT(input1);
  MS_ASSERT(output);

  para_.mul_quant_arg_.in_quant_args_[0].scale_ = input0->GetQuantParams().front().scale;
  para_.mul_quant_arg_.in_quant_args_[0].zp_ = input0->GetQuantParams().front().zeroPoint * -1;
  para_.mul_quant_arg_.in_quant_args_[1].scale_ = input1->GetQuantParams().front().scale;
  para_.mul_quant_arg_.in_quant_args_[1].zp_ = input1->GetQuantParams().front().zeroPoint * -1;
  para_.mul_quant_arg_.out_quant_arg_.scale_ = output->GetQuantParams().front().scale;
  para_.mul_quant_arg_.out_quant_arg_.zp_ = output->GetQuantParams().front().zeroPoint;
  para_.mul_quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  para_.mul_quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();

  const double real_multiplier =
    (para_.mul_quant_arg_.in_quant_args_[0].scale_ * para_.mul_quant_arg_.in_quant_args_[1].scale_) /
    para_.mul_quant_arg_.out_quant_arg_.scale_;

  int right_shift = 0;
  QuantizeMultiplierSmallerThanOne(real_multiplier, &para_.mul_quant_arg_.output_multiplier_, &right_shift);

  para_.mul_quant_arg_.shift_left_ = right_shift < 0 ? -right_shift : 0;
  para_.mul_quant_arg_.shift_right_ = right_shift > 0 ? right_shift : 0;

  return RET_OK;
}

int MulInt8CPUKernel::ReSize() { return RET_OK; }

int MulInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->Data());
  input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->Data());
  output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->Data());

  elements_num_ = in_tensors_.at(0)->ElementsNum();
  count_unit_ = thread_count_ > 1 ? UP_DIV(elements_num_, thread_count_) : elements_num_;
  if (in_tensors_.at(0)->ElementsNum() != in_tensors_.at(1)->ElementsNum()) {
    input0_data_ = static_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));
    input1_data_ = static_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));

    ArithmeticParameter tile_para = {0};
    tile_para.ndim_ = out_tensors_.at(0)->shape().size();
    for (size_t i = 0; i < tile_para.ndim_; i++) {
      tile_para.in_shape0_[i] = in_tensors_.at(0)->DimensionSize(i);
      tile_para.in_shape1_[i] = in_tensors_.at(1)->DimensionSize(i);
      tile_para.out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
    TileDimensionsInt8(static_cast<int8_t *>(in_tensors_.at(0)->Data()),
                       static_cast<int8_t *>(in_tensors_.at(1)->Data()), input0_data_, input1_data_, &tile_para);
    ret = LiteBackendParallelLaunch(MulInt8Run, this, thread_count_);
    ctx_->allocator->Free(input0_data_);
    ctx_->allocator->Free(input1_data_);
    return ret;
  }

  ret = LiteBackendParallelLaunch(MulInt8Run, this, thread_count_);
  return ret;
}

int MulInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto mul = reinterpret_cast<MulInt8CPUKernel *>(cdata);
  mul->DoExecute(task_id);
  return lite::RET_OK;
}

int MulInt8CPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  int8_t *cur_input0_data = input0_data_ + task_id * count_unit_;
  int8_t *cur_input1_data = input1_data_ + task_id * count_unit_;
  int8_t *cur_output_data = output_data_ + task_id * count_unit_;

  Mul(cur_input0_data, cur_input1_data, cur_output_data, real_dst_count, para_.mul_quant_arg_);
  return lite::RET_OK;
}

kernel::LiteKernel *CpuMulInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                            const std::vector<lite::tensor::Tensor *> &outputs,
                                            OpParameter *opParameter, const lite::Context *ctx, const KernelKey &desc,
                                            const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Mul);
  auto *kernel = new (std::nothrow) MulInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Mul, CpuMulInt8KernelCreator)
}  // namespace mindspore::kernel
