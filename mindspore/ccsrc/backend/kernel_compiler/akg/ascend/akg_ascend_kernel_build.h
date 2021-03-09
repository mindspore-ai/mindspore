/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_

#include <string>
#include <utility>
#include <vector>
#include <map>
#include "ir/anf.h"
#include "backend/kernel_compiler/akg/akg_kernel_build.h"

namespace mindspore {
namespace kernel {
class AkgAscendKernelBuilder : public AkgKernelBuilder {
 public:
  AkgAscendKernelBuilder() = default;
  ~AkgAscendKernelBuilder() = default;

  kernel::KernelBuildClient *GetClient() override { return &(kernel::AscendKernelBuildClient::Instance()); }
  KernelPackPtr AkgSearchCache(const std::string &kernel_name, const std::string &processor) override;
  KernelPackPtr AkgInsertCache(const std::string &kernel_name, const std::string &processor) override;
  void AkgSetKernelMod(const KernelPackPtr &kernel_pack, const AkgKernelJsonGenerator &json_generator,
                       const AnfNodePtr &anf_node) override;
  void AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_AKG_ASCEND_AKG_ASCEND_KERNEL_BUILD_H_
