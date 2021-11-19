/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/base/random_standard_normal.h"
#include <random>
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_RandomStandardNormal;

namespace mindspore::kernel {
int RandomStandardNormalCPUKernel::Prepare() { return RET_OK; }

int RandomStandardNormalCPUKernel::ReSize() { return RET_OK; }

int RandomStandardNormalCPUKernel::Run() {
  size_t random_seed = 0;
  if (param_->seed2_ != 0) {
    random_seed = static_cast<size_t>(param_->seed2_);
  } else if (param_->seed_ != 0) {
    random_seed = static_cast<size_t>(param_->seed_);
  } else {
    random_seed = static_cast<size_t>(clock());
  }
  std::default_random_engine engine{static_cast<unsigned int>(random_seed)};
  std::normal_distribution<float> nums(0, 1.0);
  CHECK_NULL_RETURN(out_tensors_.front());
  auto all_data_nums = out_tensors_[kOutputIndex]->ElementsNum();
  MS_CHECK_GT(all_data_nums, 0, RET_ERROR);
  auto out_data = out_tensors_[kOutputIndex]->data();
  MS_ASSERT(out_data != nullptr);
  if (out_tensors_[kOutputIndex]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    auto output = static_cast<float16_t *>(out_data);
    std::generate_n(output, all_data_nums, [&]() { return nums(engine); });
#endif
  } else {
    auto output = static_cast<float *>(out_data);
    std::generate_n(output, all_data_nums, [&]() { return nums(engine); });
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_RandomStandardNormal, LiteKernelCreator<RandomStandardNormalCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_RandomStandardNormal,
           LiteKernelCreator<RandomStandardNormalCPUKernel>)
#endif
}  // namespace mindspore::kernel
