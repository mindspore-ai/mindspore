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
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"
#include "src/runtime/kernel/arm/nnacl/quantization/quantize.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Add;

namespace mindspore::kernel {
int QuantizedAddCPUKernel::Init() {
  lite::tensor::Tensor *input0 = in_tensors_.at(0);
  lite::tensor::Tensor *input1 = in_tensors_.at(1);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input0);
  MS_ASSERT(input1);
  MS_ASSERT(output);

  para_.input0_scale_ = input0->GetQuantParams().front().scale;
  para_.input0_offset_ = input0->GetQuantParams().front().zeroPoint * -1;
  para_.input1_scale_ = input1->GetQuantParams().front().scale;
  para_.input1_offset_ = input1->GetQuantParams().front().zeroPoint * -1;
  para_.output_scale_ = output->GetQuantParams().front().scale;
  para_.output_offset_ = output->GetQuantParams().front().zeroPoint;

  const int left_shift = 20;  // 1 << 20, 2/20
  const double twice_max_input_scale = 2 * std::max(para_.input0_scale_, para_.input1_scale_);
  const double real_input0_multiplier = para_.input0_scale_ / twice_max_input_scale;
  const double real_input1_multiplier = para_.input1_scale_ / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * para_.output_scale_);

  QuantizeMultiplierSmallerThanOne(real_input0_multiplier, &para_.input0_multiplier_, &para_.input0_shift_);
  QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &para_.input1_multiplier_, &para_.input1_shift_);
  QuantizeMultiplierSmallerThanOne(real_output_multiplier, &para_.output_multiplier_, &para_.output_shift_);

  para_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  para_.output_activation_max_ = std::numeric_limits<int8_t>::max();

  int left_shift0 = -para_.input0_shift_ > 0 ? -para_.input0_shift_ : 0;
  para_.right_shift0_ = -para_.input0_shift_ > 0 ? 0 : para_.input0_shift_;

  int left_shift1 = -para_.input1_shift_ > 0 ? -para_.input1_shift_ : 0;
  para_.right_shift1_ = -para_.input1_shift_ > 0 ? 0 : para_.input1_shift_;

  para_.left_shift_out_ = -para_.output_shift_ > 0 ? -para_.output_shift_ : 0;
  para_.right_shift_out_ = -para_.output_shift_ > 0 ? 0 : para_.output_shift_;

  para_.left_shift_result0_ = (1 << left_shift) * ((1 << left_shift0));
  para_.left_shift_result1_ = (1 << left_shift) * ((1 << left_shift1));

  MS_ASSERT(left_shift + left_shift0 == left_shift);
  MS_ASSERT(left_shift + left_shift1 == left_shift);
  return 0;
}

int QuantizedAddCPUKernel::ReSize() { return 0; }

int QuantizedAddCPUKernel::Run() {
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
    TileDimensionsUint8(static_cast<uint8_t *>(in_tensors_.at(0)->Data()),
                        static_cast<uint8_t *>(in_tensors_.at(1)->Data()), reinterpret_cast<uint8_t *>(input0_data_),
                        reinterpret_cast<uint8_t *>(input1_data_), &tile_para);
    ret = LiteBackendParallelLaunch(AddInt8Run, this, thread_count_);
    ctx_->allocator->Free(input0_data_);
    ctx_->allocator->Free(input1_data_);
    return ret;
  }

  ret = LiteBackendParallelLaunch(AddInt8Run, this, thread_count_);
  return ret;
}

int AddInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto add = reinterpret_cast<QuantizedAddCPUKernel *>(cdata);
  add->DoExecute(task_id);
  return lite::RET_OK;
}

int QuantizedAddCPUKernel::DoExecute(int tId) {
  int64_t real_dst_count = MSMIN(elements_num_ - tId * count_unit_, count_unit_);
  int8_t *cur_input0_data = input0_data_ + tId * count_unit_;
  int8_t *cur_input1_data = input1_data_ + tId * count_unit_;
  int8_t *cur_output_data = output_data_ + tId * count_unit_;

  AddInt8(cur_input0_data, cur_input1_data, cur_output_data, real_dst_count, &para_);
  return lite::RET_OK;
}

kernel::LiteKernel *CpuAddInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                            const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::Context *ctx, const KernelKey &desc,
                                            const lite::Primitive *primitive) {
  if (parameter == nullptr || ctx == nullptr) {
    MS_LOG(ERROR) << "parameter or ctx is nullptr";
    return nullptr;
  }
  MS_ASSERT(desc.type == PrimitiveType_Add);
  auto *kernel = new (std::nothrow) QuantizedAddCPUKernel(parameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Add, CpuAddInt8KernelCreator)
}  // namespace mindspore::kernel
