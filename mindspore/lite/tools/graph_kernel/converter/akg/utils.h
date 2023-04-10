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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_UTILS_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_UTILS_H_
#include <string>
#include <map>
#include <set>
#include "utils/anf_utils.h"
#include "kernel/akg/akg_kernel_json_generator.h"

namespace mindspore::graphkernel {
std::string SaveNodesInfo(const AnfNodePtrList &nodes, const std::string &dir, const DumpOption &option,
                          std::map<AnfNodePtr, std::string> *node_kernel, std::set<std::string> *kernel_names);
std::string GetCNodeDynamicInputIndex(const CNodePtr &cnode);
std::string GetCNodeInputShapeStr(const CNodePtr &cnode);
std::string GetCNodeOutputShapeStr(const CNodePtr &cnode);
std::string GetCNodeOutputTypeStr(const CNodePtr &cnode);
std::string GetCNodeOutputFormatStr(const CNodePtr &cnode);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_AKG_UTILS_H_
