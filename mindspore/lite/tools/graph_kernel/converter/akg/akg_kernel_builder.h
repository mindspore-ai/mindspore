/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_KERNEL_BUILDER_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_KERNEL_BUILDER_H_
#include <string>
#include <map>
#include <set>
#include <memory>
#include "ops/custom.h"
#include "utils/anf_utils.h"
#include "kernel/graph_kernel/graph_kernel_json_generator.h"

namespace mindspore::graphkernel {
class AkgKernelBuilder {
 public:
  virtual bool CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) = 0;
  virtual AnfNodePtr CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) = 0;
  virtual bool GenerateAkgKernelNodes(const FuncGraphPtr &func_graph, const AnfNodePtr &custom_node,
                                      const CNodePtr &old_cnode) {
    return true;
  }

  static DumpOption json_option() {
    DumpOption dump_json_option;
    dump_json_option.get_target_info = true;
    return dump_json_option;
  }

 protected:
  std::string dir_path_;
};
using AkgKernelBuilderPtr = std::shared_ptr<AkgKernelBuilder>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_AKG_KERNEL_BUILDER_H_
