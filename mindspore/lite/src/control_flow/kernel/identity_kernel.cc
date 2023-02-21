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
#include "src/runtime/lite_kernel.h"
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
  if (in_tensors().size() != out_tensors().size()) {
    MS_LOG(ERROR) << "output kernel in_tensors size is not same as out_tensors size.";
    return lite::RET_ERROR;
  }
  auto ret = lite::RET_OK;
  for (size_t i = 0; i < in_tensors().size(); ++i) {
    auto src_tensor = in_tensors()[i];
    auto dst_tensor = out_tensors()[i];
    bool need_resize = false;
    if (!IsSameShape(src_tensor, dst_tensor)) {
      need_resize = true;
    }
    ret = SetTensorShape(dst_tensor, src_tensor);
    MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "set input shape failed.");
    if (need_resize) {
      ret = lite::MallocTensorData(dst_tensor);
      MS_CHECK_FALSE_MSG(ret != RET_OK, ret, "malloc dst tensor data failed.");
    }
  }
  return ret;
}

int IdentityKernel::PostProcess() { return lite::RET_OK; }

int IdentityKernel::ReSize() { return PreProcess(); }

KernelExec *IdentityKernel::Create(std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
                                   const lite::InnerContext *ctx) {
  auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));
  param->type_ = PrimType::PrimType_Inner_Identity;
  auto lite_kernel = new IdentityKernel(param, in_tensors, out_tensors, ctx);
  MS_CHECK_TRUE_MSG(lite_kernel != nullptr, nullptr, "new inner kernel failed.");
  std::shared_ptr<kernel::Kernel> shared_kernel(lite_kernel);
  auto *kernel_exec = new KernelExec(shared_kernel);
  kernel_exec->set_context(ctx);
  return kernel_exec;
}
}  // namespace mindspore::kernel
