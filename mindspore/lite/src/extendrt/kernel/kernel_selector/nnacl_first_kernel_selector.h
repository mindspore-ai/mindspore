/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_NNACL_FIRST_KERNEL_SELECTOR_H
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_NNACL_FIRST_KERNEL_SELECTOR_H

#include <vector>
#include <map>
#include <memory>
#include <string>
#include "src/extendrt/kernel/kernel_selector/kernel_selector.h"
#include "src/extendrt/kernel/nnacl/nnacl_lib.h"
#include "src/extendrt/kernel/default/default_kernel_lib.h"
#include "src/extendrt/kernel/kernel_spec_infos.h"
#include "src/extendrt/graph_compiler/compile_option.h"

namespace mindspore::kernel {
class NNACLFirstKernelSelector : public KernelSelector {
 public:
  explicit NNACLFirstKernelSelector(const std::shared_ptr<lite::CompileOption> &compile_option)
      : KernelSelector(compile_option) {}
  ~NNACLFirstKernelSelector() override = default;

  InferKernel *CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                            const std::vector<InferTensor *> &outputs, const InferContext *ctx) override {
    auto nnacl_lib = KernelLibRegister::Instance().GetKernelLib(kNNACLLibName);
    if (nnacl_lib == nullptr) {
      MS_LOG(ERROR) << "Can not find NNACL kernellib.";
      return nullptr;
    }
    auto match_ks = spec;
    match_ks.format = DEFAULT_FORMAT;
    if (nnacl_lib->Support(match_ks.op_type, match_ks.attr, match_ks.backend)) {
      auto kernel = nnacl_lib->CreateKernelExec(match_ks, inputs, outputs, ctx);
      if (kernel == nullptr) {
        MS_LOG(ERROR) << "Create kernel from " << nnacl_lib->Name() << " failed, op_type: " << match_ks.op_type
                      << ", kernel attr: " << match_ks.attr;
        return nullptr;
      }
      MS_LOG(INFO) << "Create NNACL kernel for " << match_ks.cnode->fullname_with_scope();
      return kernel;
    }

    auto acl_lib = KernelLibRegister::Instance().GetKernelLib(kAclKernelLibName);
    if (acl_lib != nullptr) {
      if (acl_lib->Support(match_ks.op_type, match_ks.attr, match_ks.backend)) {
        auto kernel = acl_lib->CreateKernelExec(match_ks, inputs, outputs, ctx);
        if (kernel == nullptr) {
          MS_LOG(ERROR) << "Create kernel from " << acl_lib->Name() << " failed, op_type: " << match_ks.op_type
                        << ", kernel attr: " << match_ks.attr;
          return nullptr;
        }
        MS_LOG(INFO) << "Create KernelMod kernel for " << match_ks.cnode->fullname_with_scope();
        return kernel;
      }
    }

    auto kernelmod_lib = KernelLibRegister::Instance().GetKernelLib(kDefaultKernelLibName);
    if (kernelmod_lib == nullptr) {
      MS_LOG(ERROR) << "Can not find kernelmod kernellib.";
      return nullptr;
    }
    if (kernelmod_lib->Support(match_ks.op_type, match_ks.attr, match_ks.backend)) {
      auto kernel = kernelmod_lib->CreateKernelExec(match_ks, inputs, outputs, ctx);
      if (kernel == nullptr) {
        MS_LOG(ERROR) << "Create kernel from " << kernelmod_lib->Name() << " failed, op_type: " << match_ks.op_type
                      << ", kernel attr: " << match_ks.attr;
        return nullptr;
      }
      MS_LOG(INFO) << "Create KernelMod kernel for " << match_ks.cnode->fullname_with_scope();
      return kernel;
    }
    return nullptr;
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_KERNEL_SELECTOR_NNACL_FIRST_KERNEL_SELECTOR_H
