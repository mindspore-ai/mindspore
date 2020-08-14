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
#include "src/runtime/kernel/arm/fp32/crop.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32/crop.h"
#include "src/runtime/kernel/arm/nnacl/crop_parameter.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
namespace {
int CropLaunch(int thread_id, LiteParallelGroupEnv *penv, void *cdata) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_NULL_PTR;
  }
  auto kernel = reinterpret_cast<CropCPUKernel *>(cdata);
  return kernel->CropParallelRun(thread_id);
}
}  // namespace

int CropCPUKernel::Init() { return RET_OK; }

int CropCPUKernel::CropParallelRun(int thread_id) {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  float *input_data = reinterpret_cast<float *>(input->Data());
  float *output_data = reinterpret_cast<float *>(output->Data());
  auto param = reinterpret_cast<CropParameter *>(op_parameter_);
  Crop4D(input_data, output_data, input->shape().data(), output->shape().data(), param, thread_id);
  return RET_OK;
}

int CropCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  auto param = reinterpret_cast<CropParameter *>(op_parameter_);
  if (output->shape()[1] < param->op_parameter_.thread_num_) {
    float *input_data = reinterpret_cast<float *>(input->Data());
    float *output_data = reinterpret_cast<float *>(output->Data());
    Crop4DNoParallel(input_data, output_data, input->shape().data(), output->shape().data(), param);
    return RET_OK;
  }

  auto ret = LiteBackendParallelLaunch(CropLaunch, this, param->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Crop launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
