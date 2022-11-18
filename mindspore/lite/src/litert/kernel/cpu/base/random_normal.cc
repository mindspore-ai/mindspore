/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/random_normal.h"
#include <random>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_RandomNormal;
using mindspore::schema::PrimitiveType_RandomStandardNormal;

namespace mindspore::kernel {
int RandomNormalCPUKernel::Prepare() {
  CHECK_NULL_RETURN(param_);
  return RET_OK;
}

int RandomNormalCPUKernel::ReSize() { return RET_OK; }

int RandomNormalCPUKernel::Run() {
  float random_seed = 0;
  if (param_->seed_ != 0) {
    random_seed = param_->seed_;
  } else {
    random_seed = static_cast<float>(clock());
  }
  std::default_random_engine engine{static_cast<unsigned int>(random_seed)};
  std::normal_distribution<float> nums(param_->mean_, param_->scale_);
  CHECK_NULL_RETURN(out_tensors_.front());
  auto all_data_nums = out_tensors_[kOutputIndex]->ElementsNum();
  MS_CHECK_GT(all_data_nums, 0, RET_ERROR);
  auto out_data = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(out_data);
  if (out_tensors_[kOutputIndex]->data_type() == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    auto output = static_cast<float16_t *>(out_data);
    std::generate_n(output, all_data_nums, [&]() { return nums(engine); });
#else
    MS_LOG(ERROR) << "ENABLE_FP16 is off, RandomNormal can not support kNumberTypeFloat16 datatype.";
    return lite::RET_NOT_SUPPORT;
#endif
  } else if (out_tensors_[kOutputIndex]->data_type() == kNumberTypeFloat32) {
    auto output = static_cast<float *>(out_data);
    std::generate_n(output, all_data_nums, [&]() { return nums(engine); });
  } else {
    MS_LOG(ERROR) << "RandomNormal op can not support out_tensors datatype is "
                  << out_tensors_[kOutputIndex]->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RandomNormal, LiteKernelCreator<RandomNormalCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RandomStandardNormal, LiteKernelCreator<RandomNormalCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_RandomNormal, LiteKernelCreator<RandomNormalCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_RandomStandardNormal, LiteKernelCreator<RandomNormalCPUKernel>)
#endif
}  // namespace mindspore::kernel
