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

#include "backend/kernel_compiler/akg/ascend/akg_ascend_kernel_build.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "ir/dtype.h"
#include "ir/func_graph.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/kernel_compiler/akg/ascend/akg_ascend_kernel_mod.h"
#include "backend/kernel_compiler/akg/akg_kernel_attrs_process.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
KernelPackPtr AkgAscendKernelBuilder::AkgSearchCache(const std::string &kernel_name, const std::string &processor) {
  return tbe::TbeUtils::SearchCache(kernel_name, processor);
}

KernelPackPtr AkgAscendKernelBuilder::AkgInsertCache(const std::string &kernel_name, const std::string &processor) {
  return tbe::TbeUtils::InsertCache(kernel_name, processor);
}

void AkgAscendKernelBuilder::AkgSetKernelMod(const KernelPackPtr &kernel_pack,
                                             const AkgKernelJsonGenerator &json_generator, const AnfNodePtr &anf_node) {
  auto kernel_mod_ptr = std::make_shared<AkgKernelMod>(kernel_pack);
  kernel_mod_ptr->SetInputSizeList(json_generator.input_size_list());
  kernel_mod_ptr->SetOutputSizeList(json_generator.output_size_list());
  AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
}

void AkgAscendKernelBuilder::AkgSaveJsonInfo(const string &kernel_name, const string &kernel_json) {
  kernel::SaveJsonInfo(kernel_name, kernel_json);
}
}  // namespace kernel
}  // namespace mindspore
