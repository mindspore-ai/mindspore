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

#include <vector>
#include "src/runtime/kernel/arm/fp32/l2_norm.h"
#include "include/errorcode.h"
#include "nnacl/l2_norm.h"


using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_L2Norm;

namespace mindspore::kernel {
int L2NormCPUKernel::Init() {
  l2_norm_param_->data_num_ = in_tensors_.at(kInputIndex)->DataSize();
  auto shape = in_tensors_.at(kInputIndex)->shape();
  l2_norm_param_->shape_ = reinterpret_cast<int *>(malloc(shape.size() * sizeof(int)));
  l2_norm_param_->shape_num_ = shape.size();
  for (size_t i = 0; i < shape.size(); i++) {
    l2_norm_param_->shape_[i] = shape[i];
  }
  return RET_OK;
}

kernel::LiteKernel *
CpuL2NormFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                           const std::vector<lite::tensor::Tensor *> &outputs,
                           OpParameter *param, const lite::Context *ctx,
                           const kernel::KernelKey &desc,
                           const mindspore::lite::PrimitiveC *primitive) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_L2Norm);
  auto *kernel = new (std::nothrow)
      L2NormCPUKernel(param, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new L2NormCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << param->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(
                         static_cast<schema::PrimitiveType>(param->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

int L2NormCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_ptr =
      reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->Data());
  auto output_ptr =
      reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  ret = L2NormFp32(input_ptr, output_ptr, l2_norm_param_);
  if (ret != 0) {
    MS_LOG_ERROR << "unsupported axis setting, more work will be done";
    return ret;
  }
  return RET_OK;
}

L2NormCPUKernel::~L2NormCPUKernel() {
  if (l2_norm_param_->shape_ != nullptr) {
    free(l2_norm_param_->shape_);
  }
  if (l2_norm_param_->axis_ != nullptr) {
    free(l2_norm_param_->axis_);
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_L2Norm,
           CpuL2NormFp32KernelCreator)
}  // namespace mindspore::kernel
