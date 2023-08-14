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

#include "src/extendrt/kernel/default/default_kernel_lib.h"
#include "src/extendrt/kernel/default/kernel_mod_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "src/infer/graph_compiler.h"

namespace mindspore::kernel {
std::shared_ptr<KernelMod> DefaultKernelLib::CreateKernelMod(const PrimitiveType &op_type, const KernelAttr &attr,
                                                             const Format &format, const std::string &backend) {
  if (backend != kBackendCPU) {
    MS_LOG(INFO) << "DefaultKernelLib only support CPU backend, but got: " << backend << ".";
    return nullptr;
  }
  if (!MatchFormat(format, Format::NCHW)) {
    MS_LOG(INFO) << "DefaultKernelLib only support NCHW layout, but got " << FormatEnumToString(format);
    return nullptr;
  }
  auto kernel_mod = Factory<NativeCpuKernelMod>::Instance().Create(op_type.TypeName());
  if (kernel_mod == nullptr) {
    MS_LOG(INFO) << "Create kernel mod failed. kernel: " << op_type;
    return nullptr;
  }
  auto match_ret = MatchKernelAttr(attr, kernel_mod->GetOpSupport());
  if (!match_ret.first) {
    MS_LOG(INFO) << "For '" << op_type << "' does not support this kernel type: " << attr;
    return nullptr;
  }
  return kernel_mod;
}

bool DefaultKernelLib::Support(const PrimitiveType &op_type, const KernelAttr &attr, const std::string &backend,
                               const Format &format) const {
  return DefaultKernelLib::CreateKernelMod(op_type, attr, format, backend) != nullptr;
}

BaseKernel *DefaultKernelLib::CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                           const std::vector<InferTensor *> &outputs, const InferContext *ctx) const {
  auto kernel_mod = DefaultKernelLib::CreateKernelMod(spec.op_type, spec.attr, spec.format, spec.backend);
  if (kernel_mod == nullptr) {
    MS_LOG(ERROR) << "Create kernel mod failed. kernel: " << spec.op_type;
    return nullptr;
  }
  return new (std::nothrow) KernelModKernel(kernel_mod, spec.primitive, spec.cnode, inputs, outputs, ctx);
}

InferKernel *DefaultKernelLib::CreateKernelExec(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                                const std::vector<InferTensor *> &outputs,
                                                const InferContext *ctx) const {
  auto *kernel_exec = KernelLib::CreateKernelExec(spec, inputs, outputs, ctx);
  if (kernel_exec == nullptr) {
    return nullptr;
  }
  auto desc = kernel_exec->desc();
  desc.format = Format::NCHW;
  kernel_exec->set_desc(desc);
  return kernel_exec;
}

REG_KERNEL_LIB(kDefaultKernelLibName, DefaultKernelLib);
}  // namespace mindspore::kernel
