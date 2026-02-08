/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_json_generator.h"
#include <nlohmann/json.hpp>
#include "backend/ms_backend/graph_fusion/model/lite_graph.h"

namespace mindspore::graphkernel {
constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
constexpr auto kGraphKernelModule = "mindspore._extends.graph_kernel";
constexpr auto kGraphKernelEstimateOps = "estimate_ops";
constexpr auto kGraphKernelGetNodeCalAmount = "estimate_calculation_amount";
constexpr auto kGraphKernelSplitFunc = "split_with_json";
constexpr auto kGetGraphKernelOpExpander = "get_op_expander";
constexpr auto kGetGraphKernelExpanderOpList = "get_expander_op_list";
constexpr auto kJsonKeyMultiGraph = "multi_graph";
constexpr auto kJsonKeyGraphDesc = "graph_desc";
constexpr auto kJsonKeyGraphMode = "graph_mode";

struct DataInfo {
  std::string format{kOpFormat_DEFAULT};
  ShapeVector shape{1};
  TypePtr type{nullptr};
};

kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types);
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types,
                                                      const kernel::Processor &processor);
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc);
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc,
                   std::map<std::string, AnfNodePtr> *address_node_map);
bool AnfToJsonDesc(const std::vector<AnfNodePtrList> &graphs, const DumpOption &dump_option, nlohmann::json *op_desc);

std::string GetFormat(const AnfNodePtr &node);
TypePtr GetType(const AnfNodePtr &node);
ShapeVector GetShape(const AnfNodePtr &node);
ShapeVector GetDeviceShape(const AnfNodePtr &node);
std::vector<int64_t> GetReduceAxis(const AnfNodePtr &node);

CNodePtr CreateCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph, const DataInfo &out_info,
                     bool use_fake_abstract = false);
void SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node);

ValueNodePtr CreateTensorValueNode(const DataInfo &info, void *value_ptr, size_t data_length);
AbstractBasePtr GetOutputAbstract(const AnfNodePtr &node, size_t output_idx);
bool IsBufferStitchNode(const AnfNodePtr &node);
bool CheckDefaultFormat(const AnfNodePtr &node);
kernel::Processor GetProcessorFromContext();
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
