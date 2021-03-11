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
#include "src/runtime/kernel/arm/fp32/crop_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
namespace {
int CropLaunch(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_NULL_PTR;
  }
  auto kernel = reinterpret_cast<CropCPUKernel *>(cdata);
  return kernel->CropParallelRun(task_id);
}
}  // namespace

int CropCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CropCPUKernel::CropParallelRun(int thread_id) {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  float *input_data = reinterpret_cast<float *>(input->data_c());
  float *output_data = reinterpret_cast<float *>(output->data_c());
  Crop4D(input_data, output_data, input->shape().data(), output->shape().data(), crop_para_, thread_id);
  return RET_OK;
}

int CropCPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  if (output->shape()[1] < crop_para_->thread_count_) {
    float *input_data = reinterpret_cast<float *>(input->data_c());
    float *output_data = reinterpret_cast<float *>(output->data_c());
    Crop4DNoParallel(input_data, output_data, input->shape().data(), output->shape().data(), crop_para_);
    return RET_OK;
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, CropLaunch, this, crop_para_->thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Crop launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Crop, LiteKernelCreator<CropCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Crop, LiteKernelCreator<CropCPUKernel>)
}  // namespace mindspore::kernel
