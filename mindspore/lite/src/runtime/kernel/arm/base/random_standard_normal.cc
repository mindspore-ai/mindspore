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
#include "src/tensorlist.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_RandomStandardNormal;

namespace mindspore::kernel {

int RandomStandardNormalCPUKernel::Init() { return RET_OK; }

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
  std::normal_distribution<double> nums(0, 1.0);
  auto all_data_nums = out_tensors_[0]->ElementsNum();
  auto out_data = out_tensors_[0]->data_c();
  MS_ASSERT(out_data != nullptr);
  auto output = reinterpret_cast<float *>(out_data);
  for (int i = 0; i < all_data_nums; ++i) {
    output[i] = nums(engine);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_RandomStandardNormal, LiteKernelCreator<RandomStandardNormalCPUKernel>)
}  // namespace mindspore::kernel
