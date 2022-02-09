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

#include "src/control_flow/identity_kernel.h"
#include "src/tensor.h"
#include "src/inner_kernel.h"
#include "src/common/tensor_util.h"
#include "src/common/prim_inner.h"

namespace mindspore::kernel {
int IdentityKernel::Run() {
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    auto src_tensor = in_tensors()[i];
    auto dst_tensor = out_tensors()[i];
    if (NeedCastData(dst_tensor, src_tensor)) {
      CastTensorData(dst_tensor, src_tensor, support_fp16_);
      continue;
    }
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      SetTensorData(dst_tensor, src_tensor);
    } else {
      MoveTensorData(dst_tensor, src_tensor);
    }
  }
  return lite::RET_OK;
}

int IdentityKernel::PreProcess() {
  if (in_tensors().size() != out_tensors().size()) {
    MS_LOG(ERROR) << "output kernel in_tensors size is not same as out_tensors size.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    auto src_tensor = in_tensors()[i];
    auto dst_tensor = out_tensors()[i];
    if (src_tensor->data_type() == kObjectTypeTensorType) {
      SetTensorListShape(dst_tensor, src_tensor);
    } else {
      SetTensorShape(dst_tensor, src_tensor);
    }
  }
  return lite::RET_OK;
}

int IdentityKernel::PostProcess() { return lite::RET_OK; }

int IdentityKernel::ReSize() { return PreProcess(); }

LiteKernel *IdentityKernel::Create(std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
                                   const lite::InnerContext *ctx) {
  auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));
  param->type_ = PrimType::PrimType_Inner_Identity;
  auto inner_kernel = new IdentityKernel(param, in_tensors, out_tensors, ctx);
  MS_CHECK_TRUE_MSG(inner_kernel != nullptr, nullptr, "new inner kernel failed.");
  std::shared_ptr<kernel::Kernel> shared_kernel(inner_kernel);
  auto *lite_kernel = new LiteKernel(shared_kernel);
  lite_kernel->set_context(ctx);
  return lite_kernel;
}
}  // namespace mindspore::kernel
