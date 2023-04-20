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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_ASCEND_KERNEL_BUILDER_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_ASCEND_KERNEL_BUILDER_H_
#include <string>
#include <map>
#include "tools/graph_kernel/converter/akg/akg_kernel_builder.h"

namespace mindspore::graphkernel {
class AscendKernelBuilder : public AkgKernelBuilder {
 public:
  bool CompileJsonsInAnfnodes(const AnfNodePtrList &node_list) override;
  AnfNodePtr CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) override;

 private:
  std::string dir_path_;
  std::map<AnfNodePtr, std::string> node_info_map_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_ASCEND_KERNEL_BUILDER_H_
