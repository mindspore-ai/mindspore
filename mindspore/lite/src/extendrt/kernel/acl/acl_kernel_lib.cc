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

#include "src/extendrt/kernel/acl/acl_kernel_lib.h"
#include "src/extendrt/kernel/ascend/api/ascend_kernel_api.h"
#include "src/extendrt/kernel/acl/acl_lite_kernel.h"
#include "src/extendrt/kernel/kernel_spec_infos.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "src/infer/graph_compiler.h"
#include "mindspore/lite/src/common/common.h"

namespace mindspore::kernel {
std::shared_ptr<KernelMod> AclKernelLib::CreateKernelMod(const PrimitiveType &op_type, const KernelAttr &attr,
                                                         const Format &format, const std::string &backend) {
  if (backend != kBackendAscend) {
    MS_LOG(INFO) << "AclKernelLib only support Ascend backend, but got: " << backend << ".";
    return nullptr;
  }
  if (!MatchFormat(format, Format::NCHW)) {
    MS_LOG(INFO) << "AclKernelLib only support NCHW layout, but got " << FormatEnumToString(format);
    return nullptr;
  }

  auto kernel_name = lite::kNameCustomAscend;
  std::shared_ptr<kernel::KernelMod> kernel_mod = kernel::Factory<kernel::KernelMod>::Instance().Create(kernel_name);

  if (kernel_mod == nullptr) {
    MS_LOG(INFO) << "Create kernel mod failed. kernel: " << op_type.TypeName();
    return nullptr;
  }
  // acl custom inputs and outputs number is not fixed, so do not checkout kernel attr here
  return kernel_mod;
}

bool AclKernelLib::Support(const PrimitiveType &op_type, const KernelAttr &attr, const std::string &backend,
                           const Format &format) const {
  return AclKernelLib::CreateKernelMod(op_type, attr, format, backend) != nullptr;
}

BaseKernel *AclKernelLib::CreateKernel(const KernelSpec &spec, const std::vector<InferTensor *> &inputs,
                                       const std::vector<InferTensor *> &outputs, const InferContext *ctx) const {
  auto kernel_mod = AclKernelLib::CreateKernelMod(spec.op_type, spec.attr, spec.format, spec.backend);
  if (kernel_mod == nullptr) {
    MS_LOG(ERROR) << "Create kernel mod failed. kernel: " << spec.op_type.TypeName();
    return nullptr;
  }
  return new (std::nothrow) AclLiteKernel(kernel_mod, spec.primitive, inputs, outputs, ctx);
}

REG_KERNEL_LIB(kAclKernelLibName, AclKernelLib);
}  // namespace mindspore::kernel
