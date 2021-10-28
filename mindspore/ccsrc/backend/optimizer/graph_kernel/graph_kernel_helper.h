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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include <nlohmann/json.hpp>
#include "backend/optimizer/graph_kernel/model/lite_graph.h"

namespace mindspore {
namespace opt {
using kernel::DumpOption;

constexpr auto kIsFeatureMapOutput = "IsFeatureMapOutput";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
constexpr auto kGraphKernelModule = "mindspore._extends.graph_kernel";
constexpr auto kGraphKernelEstimateOps = "estimate_ops";
constexpr auto kGraphKernelGetNodeCalAmount = "estimate_calulation_amount";
constexpr auto kGraphKernelSplitFunc = "split_with_json";
constexpr auto kGetGraphKernelOpExpander = "get_op_expander";
constexpr auto kJsonKeyMultiGraph = "multi_graph";
constexpr auto kJsonKeyGraphDesc = "graph_desc";
constexpr auto kJsonKeyGraphMode = "graph_mode";
constexpr auto kAllTarget = "ALL";

constexpr auto kGraphKernelDumpPath = "graph_kernel_dump";
inline const PrimitivePtr kPrimUnPadAkg = std::make_shared<Primitive>("UnPadAkg");
inline const PrimitivePtr kPrimPadAkg = std::make_shared<Primitive>("PadAkg");
struct DataInfo {
  std::string format{kOpFormat_DEFAULT};
  ShapeVector shape{1};
  TypePtr type{nullptr};
};

bool ConvertNonscalarTensorToParameter(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr);
std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> MixedNodesTransToGraph(const AnfNodePtrList &fuse_nodes,
                                                                                AnfNodePtrList *src_outputs = nullptr);
void SetNewKernelInfo(const AnfNodePtr &new_node, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                      const AnfNodePtrList &outputs);
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types, const AnfNodePtr &node);
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types);
AnfNodePtr CreateNewFuseCNode(const FuncGraphPtr &kernel_graph, const FuncGraphPtr &fg, const AnfNodePtrList &inputs,
                              const AnfNodePtrList &outputs);
void ReplaceNewFuseCNode(const FuncGraphPtr &kernel_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs);
std::tuple<AnfNodePtr, AnfNodePtrList> FuseNodesToSubGraph(const std::vector<AnfNodePtr> &fuse_nodes,
                                                           const FuncGraphPtr &kernel_graph,
                                                           const std::string &postfix = "");
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc);
bool AnfToJsonDesc(const AnfNodePtrList &nodes, const DumpOption &dump_option, nlohmann::json *op_desc,
                   std::map<std::string, AnfNodePtr> *address_node_map);
bool AnfToJsonDesc(const std::vector<AnfNodePtrList> &graphs, const DumpOption &dump_option, nlohmann::json *op_desc);
FuncGraphPtr JsonDescToAnf(const std::string &json_desc);
std::string ExtractGraphKernelName(const AnfNodePtrList &cnodes, const string &prefix = "", const string &postfix = "");
void ResetKernelInfo(const AnfNodePtr &node, KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);

std::string GetFormat(const AnfNodePtr &node);
TypePtr GetType(const AnfNodePtr &node);
ShapeVector GetShape(const AnfNodePtr &node);
ShapeVector GetDeviceShape(const AnfNodePtr &node);
std::vector<int64_t> GetReduceAxis(const AnfNodePtr &node);

CNodePtr CreateCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph, const DataInfo &out_info,
                     bool use_fake_abstract = false);
void SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node);
bool IsKeepBasicNode(const AnfNodePtr &node);
void OpListFilter(std::vector<PrimitivePtr> *ops, const std::vector<std::string> &enable_ops_only,
                  const std::vector<std::string> &enable_ops, const std::vector<std::string> &disable_ops);
template <typename T>
ValueNodePtr CreateScalarTensorValueNode(const DataInfo &info, T value, size_t data_length) {
  // Create tensor value.
  if (info.shape.size() != 1 && info.shape[0] != 1) {
    MS_LOG(EXCEPTION) << "Only support create scalar tensor value node!!!";
  }

  if (info.type == nullptr) {
    MS_LOG(EXCEPTION) << "Data type is needed!!!";
  }

  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(info.type->type_id(), info.shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor::DeviceInfo device_info{info.format, info.type};
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(tensor->data().nbytes()), &value, data_length);
  if (ret_code != 0) {
    MS_LOG(EXCEPTION) << "Failed to copy data into scalar tensor.";
  }

  // Create value node.
  ValueNodePtr new_value_node = std::make_shared<ValueNode>(tensor);
  new_value_node->set_abstract(tensor->ToAbstract());
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{info.format});
  std::vector<TypeId> types = {info.type->type_id()};
  kernel_build_info_builder->SetOutputsDeviceType(types);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());

  return new_value_node;
}

AbstractBasePtr GetOutputAbstract(const AnfNodePtr &node, size_t output_idx);

// functions to graphkernel model
graphkernel::LiteGraphPtr AnfGraph2LiteGraph(const FuncGraphPtr &func_graph);
FuncGraphPtr LiteGraph2AnfGraph(const graphkernel::LiteGraphPtr &lite_graph, AnfNodePtrList *outputs = nullptr);

// remove parameter which is not used
void EliminateRedundantParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs);

std::vector<PrimitivePtr> GetValidOps(
  const std::vector<std::tuple<std::string, unsigned int, PrimitivePtr>> &ops_with_level, unsigned int level);

// return a func_graph's manager
FuncGraphManagerPtr GetFuncGraphManager(const FuncGraphPtr &func_graph);

void UpdateMng(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_HELPER_H_
