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

#include "src/control_flow/kernel/identity_kernel.h"
#include "src/tensor.h"
#include "src/litert/lite_kernel.h"
#include "src/common/tensor_util.h"
#include "src/common/prim_inner.h"

namespace mindspore::kernel {
int IdentityKernel::Run() {
  auto ret = lite::RET_OK;
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    auto src_tensor = in_tensors()[i];
    auto dst_tensor = out_tensors()[i];
    if (NeedCastData(dst_tensor, src_tensor)) {
      ret = CastTensorData(dst_tensor, src_tensor, support_fp16_);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "identity cast failed.");
      continue;
    }
    if (src_tensor->allocator() == nullptr || src_tensor->IsGraphInput()) {
      ret = SetTensorData(dst_tensor, src_tensor);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "identity set tensor data failed.");
    } else {
      ret = MoveTensorData(dst_tensor, src_tensor);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "identity move tensor data failed.");
    }
  }
  return ret;
}

int IdentityKernel::PreProcess() {
  auto ret = InferShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "infer shape failed.";
    return ret;
  }
  ret = ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "resize failed.";
    return ret;
  }
  return RET_OK;
}

int IdentityKernel::InferShape() {
  if (in_tensors().size() != out_tensors().size()) {
    MS_LOG(ERROR) << "output kernel in_tensors size is not same as out_tensors size.";
    return lite::RET_ERROR;
  }
  need_resize_.resize(in_tensors().size());
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    auto src_tensor = in_tensors()[i];
    auto dst_tensor = out_tensors()[i];
    need_resize_[i] = !IsSameShape(src_tensor, dst_tensor);
    auto ret = SetTensorShape(dst_tensor, src_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "set output shape failed.";
      return ret;
    }
  }
  return RET_OK;
}

int IdentityKernel::PostProcess() { return lite::RET_OK; }

int IdentityKernel::ReSize() {
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    if (need_resize_[i]) {
      auto ret = lite::MallocTensorData(out_tensors_[i]);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "malloc dst tensor data failed.");
    }
  }
  return RET_OK;
}

KernelExec *IdentityKernel::Create(std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
                                   const lite::InnerContext *ctx) {
  auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = PrimType::PrimType_Inner_Identity;
  auto lite_kernel = new IdentityKernel(param, in_tensors, out_tensors, ctx);
  MS_CHECK_TRUE_MSG(lite_kernel != nullptr, nullptr, "new inner kernel failed.");
  std::shared_ptr<kernel::Kernel> shared_kernel(lite_kernel);
  auto *kernel_exec = new KernelExec(shared_kernel);
  kernel_exec->set_context(ctx);
  return kernel_exec;
}
}  // namespace mindspore::kernel
