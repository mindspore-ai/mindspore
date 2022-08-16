/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_BUILD_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_BUILD_H_
#include <string>
#include <map>
#include <vector>
#include "utils/anf_utils.h"
#include "kernel/akg/akg_kernel_json_generator.h"

namespace mindspore::graphkernel {
constexpr size_t PROCESS_LIMIT = 8;
constexpr size_t TIME_OUT = 100;

class AkgKernelBuilder {
 public:
  AkgKernelBuilder() = default;
  ~AkgKernelBuilder() = default;

  bool CompileJsonsInAnfnodes(const AnfNodePtrList &node_list);

  static DumpOption json_option() {
    DumpOption dump_json_option;
    dump_json_option.get_target_info = true;
    return dump_json_option;
  }
};

std::string SaveNodesInfo(const AnfNodePtrList &nodes, const std::string &dir, const DumpOption &option,
                          std::map<AnfNodePtr, std::string> *node_name, std::vector<std::string> *kernel_names);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_BUILD_H_
