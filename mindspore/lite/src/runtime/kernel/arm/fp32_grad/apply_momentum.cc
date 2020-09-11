
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

#include "src/runtime/kernel/arm/fp32_grad/apply_momentum.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp32/nchw2nhwc.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ApplyMomentum;

namespace mindspore::kernel {

int ApplyMomentumCPUKernel::ReSize() { return RET_OK; }

int ApplyMomentumCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  auto weight = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  auto accumulate = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  float learning_rate = reinterpret_cast<float *>(in_tensors_[2]->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_[3]->MutableData());
  float moment = reinterpret_cast<float *>(in_tensors_[4]->MutableData())[0];
  size_t elem_num = in_tensors_[0]->ElementsNum();

  // align format
  if (in_tensors_[3]->shape().size() == 4 && in_tensors_[3]->GetFormat() == schema::Format_NCHW &&
      in_tensors_[0]->GetFormat() == schema::Format_KHWC) {
    PackNCHWToNHWCFp32(gradient, workspace, in_tensors_[0]->Batch(), in_tensors_[0]->Height() * in_tensors_[0]->Width(),
                       in_tensors_[0]->Channel());
  } else {
    memcpy(workspace, gradient, in_tensors_[3]->ElementsNum() * sizeof(float));
  }

  for (size_t i = 0; i < elem_num; ++i) {
    accumulate[i] = accumulate[i] * moment + workspace[i];  // * (1.0 - moment);
    weight[i] -= accumulate[i] * learning_rate;
  }
  return RET_OK;
}

int ApplyMomentumCPUKernel::Init() {
  // Only for test with uninitialized Data
  size_t elem_num = in_tensors_[0]->ElementsNum();
  auto accumulate = reinterpret_cast<float *>(in_tensors_[1]->MutableData());
  for (size_t i = 0; i < elem_num; i++) accumulate[i] = 0.0;

  workspace = new float[elem_num];
  return 0;
}
#if 0
OpParameter *PopulateApplyMomentumParameter(const lite::Primitive *primitive) {
  OpParameter *param = new (std::nothrow) OpParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "new Param for OptMomentum failed.";
    return nullptr;
  }
  param->type_ = primitive->Type();
  return param;
}
#endif

kernel::LiteKernel *CpuApplyMomentumFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::Context *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_ApplyMomentum);
  auto *kernel = new (std::nothrow) ApplyMomentumCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  MS_ASSERT(kernel != nullptr);

  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ApplyMomentum, CpuApplyMomentumFp32KernelCreator)
}  // namespace mindspore::kernel
