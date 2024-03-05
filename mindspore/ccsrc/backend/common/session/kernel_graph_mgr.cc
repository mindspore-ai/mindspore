/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "backend/common/session/kernel_graph_mgr.h"

#include <algorithm>
#include <queue>
#include <utility>
#include <functional>
#include <unordered_map>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "pipeline/pynative/grad/jit/jit_call_graph.h"
#include "utils/trace_base.h"
#include "ir/func_graph_cloner.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#endif
#include "include/common/utils/compile_cache_context.h"
#include "include/common/utils/config_manager.h"
#include "load_mindir/load_model.h"
#include "include/common/debug/dump_proto.h"

namespace mindspore {
namespace session {
namespace {
constexpr size_t kMaxDepth = 128;
constexpr size_t kFirstIndex = 1;
constexpr int64_t kPairIdx1 = 1;

bool IsGeReturnNode(const AnfNodePtr &node) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool enable_ge = context->backend_policy() == "ge";
  if (!enable_ge) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    // parameter and value node is a real kernel too
    return true;
  }
  if (cnode->size() == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Illegal null input of cnode(%s)" << node->DebugString()
                               << trace::DumpSourceLines(node);
  }
  return IsOneOfPrimitive(cnode->input(kAnfPrimitiveIndex), {prim::kPrimReturn});
}

bool RecursiveCheck(const FuncGraphManagerPtr &manager, const std::pair<AnfNodePtr, int64_t> &kernel, size_t *idx) {
  auto node = kernel.first;
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (kernel.second > kPairIdx1 && (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) ||
                                    common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad))) {
    return false;
  }
  if ((AnfUtils::IsRealKernel(node) || IsGeReturnNode(node) || AnfAlgo::IsSummaryNode(node)) &&
      !common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    return true;
  }
  (*idx) += 1;
  // max recursion depth
  if (*idx <= kMaxDepth) {
    auto users = manager->node_users()[node];
    if (std::any_of(users.begin(), users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
          return RecursiveCheck(manager, kernel, idx);
        })) {
      return true;
    }
  }
  return false;
}

bool IsUsedByRealKernel(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, const uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  auto node_users = manager->node_users()[node];
  // filter nodes not in current graph
  for (auto iter = node_users.begin(); iter != node_users.end();) {
    auto func_graph = iter->first->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    if (kernel_graph == nullptr) {
      MS_LOG(EXCEPTION) << "func graph cast kernel graph failed, related node is: " << iter->first->DebugString();
    }
    if (kernel_graph->graph_id() != graph_id) {
      iter = node_users.erase(iter);
    } else {
      iter++;
    }
  }

  size_t idx = 0;
  if (std::any_of(node_users.begin(), node_users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
        return RecursiveCheck(manager, kernel, &idx);
      })) {
    return true;
  }
  return false;
}

bool ExistGraphCaller(const AnfNodePtr &partial_node) {
  MS_EXCEPTION_IF_NULL(partial_node);
  auto partial_cnode = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto partial_graph = GetValueNode<FuncGraphPtr>(partial_cnode->input(kFirstDataInputIndex));
  // If graph is nullptr, it means that the funcgraph in the partial node is a deadnode, and the processing is skipped.
  if (partial_graph == nullptr) {
    return false;
  }
  auto graph_nodes = TopoSort(partial_graph->get_return());
  return std::any_of(graph_nodes.begin(), graph_nodes.end(), IsValueNode<FuncGraph>);
}

bool CheckPath(const std::optional<std::string> &path) {
  if (!path.has_value()) {
    return false;
  }
  std::ifstream f(path.value());
  bool file_is_good = f.good();
  f.close();
  if (!file_is_good) {
    MS_LOG(WARNING) << "Open the compilation cache file " << path.value() << " failed.";
    return false;
  }
  return true;
}

bool LoadJson(const std::string &filename, nlohmann::json *graph_json) {
  std::ifstream json_fs(filename);
  if (!json_fs.is_open()) {
    MS_LOG(ERROR) << "Open json file: " << filename << " error, backend graph cache Missed.";
    return false;
  }
  try {
    json_fs >> *graph_json;
    json_fs.close();
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Parse json file error: " << filename << ", sleep 500ms and retry again.";
    json_fs.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMilliSeconds));
    std::ifstream retry_tmp(filename);
    if (!retry_tmp.is_open()) {
      MS_LOG(ERROR) << "Open json file: " << filename << " error, please check cached file.";
      return false;
    }
    retry_tmp >> *graph_json;
    retry_tmp.close();
  }
  return true;
}

template <typename Type>
Type StringToNum(const std::string &str) {
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}

void LoadKernelInfoRuntimeCache(const nlohmann::json &kernel_info_value,
                                std::shared_ptr<KernelInfoDevice> kernel_info) {
  if (!kernel_info_value.contains(kRuntimeCacheValid)) {
    return;
  }
  auto &context = CompileCacheContext::GetInstance();
  auto &rt = kernel_info->runtime_cache().runtime_cache();
  rt.set_is_valid(kernel_info_value[kRuntimeCacheValid]);
  rt.set_device_target(kernel_info_value[kRuntimeCacheDeviceTarget]);
  rt.set_output_tensor_num(kernel_info_value[kRuntimeCacheOutputTensorNum]);
  rt.set_real_kernel(kernel_info_value[kRuntimeCacheIsRealKernel]);
  if (kernel_info_value.contains(kRuntimeCachePrevOutputs)) {
    const auto &prev_outputs = kernel_info_value[kRuntimeCachePrevOutputs];
    for (const auto &prev_output : prev_outputs) {
      const auto &first_index = prev_output.at(0);
      const auto &name = prev_output.at(kIndexOne);
      const auto &second_index = prev_output.at(kIndexTwo);
      auto output_node = context.FindBackNodeByBackName(name);
      MS_EXCEPTION_IF_NULL(output_node);
      rt.update_prev_node_output(first_index, std::make_pair(output_node, second_index));
    }
  }
}

void LoadAnfSelectKernelBuildInfo(const nlohmann::json &kernel_info_value, const AnfNodePtr &node) {
  if (!kernel_info_value.contains(kHasSelectKernelBuildInfo)) {
    return;
  }
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(kernel_build_info_builder);

  auto kernel_type = kernel_info_value[kKernelType];
  kernel_build_info_builder->SetKernelType(kernel_type);

  auto op_type = kernel_info_value[kOpType];
  kernel_build_info_builder->SetOpType(op_type);
  auto core_type = kernel_info_value[kCoreType];
  kernel_build_info_builder->SetCoreType(core_type);

  if (kernel_info_value.contains(kOriginDataFormat)) {
    auto origin_data_format = kernel_info_value[kOriginDataFormat];
    kernel_build_info_builder->SetOriginDataFormat(origin_data_format);
  }

  if (kernel_info_value.contains(kAllInputFormat)) {
    auto all_input_format = kernel_info_value[kAllInputFormat];
    kernel_build_info_builder->SetInputsFormat(all_input_format);
  }

  auto pattern = kernel_info_value[kOpPattern];
  kernel_build_info_builder->SetOpPattern(pattern);
  if (kernel_info_value.contains(kAllOutputFormat)) {
    auto all_output_format = kernel_info_value[kAllOutputFormat];
    kernel_build_info_builder->SetOutputsFormat(all_output_format);
  }
  if (kernel_info_value.contains(kAllInputReshapeType)) {
    auto all_input_reshape_type = kernel_info_value[kAllInputReshapeType];
    kernel_build_info_builder->SetInputsReshapeType(all_input_reshape_type);
  }

  if (kernel_info_value.contains(kAllOutputReshapeType)) {
    auto all_output_reshape_type = kernel_info_value[kAllOutputReshapeType];
    kernel_build_info_builder->SetOutputsReshapeType(all_output_reshape_type);
  }
  if (kernel_info_value.contains(kAllInputDeviceType)) {
    auto all_input_device_type = kernel_info_value[kAllInputDeviceType];
    kernel_build_info_builder->SetInputsDeviceType(all_input_device_type);
  }
  if (kernel_info_value.contains(kAllOutputDeviceType)) {
    auto all_output_device_type = kernel_info_value[kAllOutputDeviceType];
    kernel_build_info_builder->SetOutputsDeviceType(all_output_device_type);
  }
  if (kernel_info_value.contains(kInputKernelObjectTypes)) {
    auto input_kernel_object_types = kernel_info_value[kInputKernelObjectTypes];
    kernel_build_info_builder->SetInputsKernelObjectType(input_kernel_object_types);
  }
  if (kernel_info_value.contains(kOutputKernelObjectTypes)) {
    auto output_kernel_object_types = kernel_info_value[kOutputKernelObjectTypes];
    kernel_build_info_builder->SetOutputsKernelObjectType(output_kernel_object_types);
  }
  if (kernel_info_value.contains(kOutputElementsKernelObjectTypes)) {
    auto output_elements_kernel_object_types = kernel_info_value[kOutputElementsKernelObjectTypes];
    kernel_build_info_builder->SetOutputElementsKernelObjectType(output_elements_kernel_object_types);
  }

  if (kernel_info_value.contains(kOutputDataDesc)) {
    auto output_data_desc = kernel_info_value[kOutputDataDesc];
    kernel_build_info_builder->SetOutputDataDesc(output_data_desc);
  }
  auto fusion_type = kernel_info_value[kFusionType];
  kernel_build_info_builder->SetFusionType(fusion_type);
  auto processor = kernel_info_value[kProcessor];
  kernel_build_info_builder->SetProcessor(processor);
  auto valid = kernel_info_value[kKernelBuildInfoValid];
  kernel_build_info_builder->SetValid(valid);
  const auto &kernel_build = kernel_build_info_builder->Build();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build, node.get());
}

void LoadAnfKernelInfoFromJson(const nlohmann::json &kernel_infos_json) {
  auto &context = CompileCacheContext::GetInstance();
  for (const auto &[name, kernel_info_value] : kernel_infos_json.items()) {
    auto node = context.FindBackNodeByBackName(name);
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Load node " << node->DebugString() << " kernel info from json.";
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    if (kernel_info_value.contains(kOutInRef)) {
      const auto &out_in_ref_json = kernel_info_value[kOutInRef];
      for (const auto &[out, in] : out_in_ref_json.items()) {
        kernel_info->AddRefMap(StringToNum<size_t>(out), in);
      }
    }
    kernel_info->set_graph_id(kernel_info_value[kGraphId]);
    kernel_info->set_feature_map_flag(kernel_info_value[kIsFeatureMap]);
    node->set_kernel_info(kernel_info);
    LoadAnfSelectKernelBuildInfo(kernel_info_value, node);

    if (node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrIsUBFusionOp, node->cast<CNodePtr>()) &&
        common::AnfAlgo::GetNodeAttr<bool>(node->cast<CNodePtr>(), kAttrIsUBFusionOp)) {
      if (kernel_info_value.contains(kJsonName) && kernel_info_value.contains(kInputSizeList) &&
          kernel_info_value.contains(kOutputSizeList)) {
        CachedIOSizeInfo io_size;
        io_size.json_name = kernel_info_value[kJsonName];
        const auto &input_size_list = kernel_info_value[kInputSizeList];
        const auto &output_size_list = kernel_info_value[kOutputSizeList];
        (void)(io_size.input_size_list.insert(io_size.input_size_list.end(), input_size_list.begin(),
                                              input_size_list.end()));
        (void)(io_size.output_size_list.insert(io_size.output_size_list.end(), output_size_list.begin(),
                                               output_size_list.end()));
        context.PushFullnameIoSizeInfo(node->fullname_with_scope(), io_size);
      } else {
        MS_LOG(EXCEPTION) << "Load node " << node->DebugString() << " kernel_io_size_info failed.";
      }
    }
    LoadKernelInfoRuntimeCache(kernel_info_value, kernel_info);
  }
}

std::string GetAnfUniqueCacheName(const AnfNodePtr &node, bool must_have_unique_name = true) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &name = node->user_data<std::string>(kUniqueCacheName);
  if (must_have_unique_name && name == nullptr) {
    MS_LOG(EXCEPTION) << "The node " << node->DebugString()
                      << " has not unique name, indicating that it is not exported to mindir.";
  }
  return name != nullptr ? *name : "";
}

nlohmann::json SaveAnfToAnfMap(const HashMap<AnfNodePtr, AnfNodePtr> &save_map) {
  nlohmann::json ret;
  for (const auto &i : save_map) {
    const auto &first_name = GetAnfUniqueCacheName(i.first, false);
    const auto &second_name = GetAnfUniqueCacheName(i.second, false);
    // allow some node not to be exported to mindir.
    if (first_name.empty() || second_name.empty()) {
      continue;
    }
    ret[first_name] = second_name;
  }
  return ret;
}

std::vector<nlohmann::json> SaveAnfToAnfIndexMap(const HashMap<AnfNodePtr, AnfWithOutIndex> &save_map) {
  std::vector<nlohmann::json> ret_json;
  for (const auto &i : save_map) {
    nlohmann::json iter_json;
    const auto &first_name = GetAnfUniqueCacheName(i.first, false);
    const auto &second_name = GetAnfUniqueCacheName(i.second.first, false);
    // allow some node not to be exported to mindir.
    if (first_name.empty() || second_name.empty()) {
      continue;
    }
    iter_json.push_back(first_name);
    iter_json.push_back(second_name);
    iter_json.push_back(i.second.second);
    (void)(ret_json.emplace_back(iter_json));
  }
  return ret_json;
}

std::vector<nlohmann::json> SaveAnfIndexToAnfIndexMap(const std::map<AnfWithOutIndex, AnfWithOutIndex> &save_map) {
  std::vector<nlohmann::json> ret_json;
  for (const auto &i : save_map) {
    nlohmann::json iter_json;
    const auto &first_name = GetAnfUniqueCacheName(i.first.first, false);
    const auto &second_name = GetAnfUniqueCacheName(i.second.first, false);
    // allow some node not to be exported to mindir.
    if (first_name.empty() || second_name.empty()) {
      continue;
    }
    iter_json.push_back(first_name);
    iter_json.push_back(i.first.second);
    iter_json.push_back(second_name);
    iter_json.push_back(i.second.second);
    (void)(ret_json.emplace_back(iter_json));
  }
  return ret_json;
}

nlohmann::json SaveValueSet(const HashSet<ValueNodePtr> &save_anfs) {
  nlohmann::json iter_json;
  for (const auto &i : save_anfs) {
    const auto &name = GetAnfUniqueCacheName(i, false);
    // allow some value node not to be exported to mindir.
    if (name.empty()) {
      continue;
    }
    (void)(iter_json.emplace_back(name));
  }
  return iter_json;
}

template <typename T>
nlohmann::json SaveAnfVec(const std::vector<T> &save_anfs) {
  nlohmann::json ret_json;
  for (const auto &i : save_anfs) {
    const auto &name = GetAnfUniqueCacheName(i);
    (void)(ret_json.emplace_back(name));
  }
  return ret_json;
}

nlohmann::json SaveGraphVec(const std::vector<std::weak_ptr<KernelGraph>> &save_anfs) {
  nlohmann::json ret_json;
  for (const auto &i : save_anfs) {
    const auto &ptr = i.lock();
    MS_EXCEPTION_IF_NULL(ptr);
    (void)(ret_json.emplace_back(ptr->graph_id()));
  }
  return ret_json;
}

nlohmann::json SaveGraphsId(const HashMap<uint32_t, std::weak_ptr<session::KernelGraph>> &to_save) {
  nlohmann::json ret_json;
  for (const auto &i : to_save) {
    nlohmann::json iter_json;
    const auto &ptr = i.second.lock();
    MS_EXCEPTION_IF_NULL(ptr);
    ret_json.push_back(ptr->graph_id());
  }
  return ret_json;
}

std::vector<nlohmann::json> SavePrevOutputs(const std::map<size_t, std::pair<AnfNodeWeakPtr, size_t>> &save_map) {
  std::vector<nlohmann::json> ret_json;
  for (const auto &i : save_map) {
    nlohmann::json iter_json;
    const auto &node = i.second.first.lock();
    MS_EXCEPTION_IF_NULL(node);
    const auto &name = GetAnfUniqueCacheName(node, false);
    if (name.empty()) {
      continue;
    }
    iter_json.push_back(i.first);
    iter_json.push_back(name);
    iter_json.push_back(i.second.second);
    ret_json.push_back(iter_json);
  }
  return ret_json;
}

void SaveKernelInfoRuntimeCache(KernelInfoDevice *kernel_info, nlohmann::json *const single_json) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  MS_EXCEPTION_IF_NULL(single_json);
  const auto &rt = kernel_info->runtime_cache().runtime_cache();
  if (!rt.is_valid()) {
    return;
  }
  (*single_json)[kRuntimeCacheValid] = rt.is_valid();
  (*single_json)[kRuntimeCacheDeviceTarget] = rt.device_target();
  (*single_json)[kRuntimeCacheOutputTensorNum] = rt.output_tensor_num();
  (*single_json)[kRuntimeCacheIsRealKernel] = rt.is_real_kernel();
  const auto &prev_outputs_json = SavePrevOutputs(rt.GetPrevOutputs());
  if (!prev_outputs_json.empty()) {
    (*single_json)[kRuntimeCachePrevOutputs] = prev_outputs_json;
  }
}

nlohmann::json SaveAnfKernelInfo(const AnfNodePtr &node) {
  nlohmann::json single_json;
  if (AnfUtils::IsRealKernel(node)) {
    single_json[kOriginDataFormat] = AnfAlgo::GetOriginDataFormat(node);
    const auto &input_formats = AnfAlgo::GetAllInputFormats(node);
    if (!input_formats.empty()) {
      single_json[kAllInputFormat] = input_formats;
    }
    const auto &output_formats = AnfAlgo::GetAllOutputFormats(node);
    if (!output_formats.empty()) {
      single_json[kAllOutputFormat] = output_formats;
    }
    const auto &input_device_types = AnfAlgo::GetAllInputDeviceTypes(node);
    if (!input_device_types.empty()) {
      single_json[kAllInputDeviceType] = input_device_types;
    }
    const auto &output_device_types = AnfAlgo::GetAllOutputDeviceTypes(node);
    if (!output_device_types.empty()) {
      single_json[kAllOutputDeviceType] = output_device_types;
    }
  }
  if (AnfAlgo::HasSelectKernelBuildInfo(node)) {
    auto kernel_type = AnfAlgo::GetKernelType(node);
    single_json[kKernelType] = kernel_type;
    auto op_type = AnfAlgo::GetOpType(node);
    single_json[kOpType] = op_type;
    single_json[kCoreType] = AnfAlgo::GetCoreType(node);
    single_json[kOpPattern] = AnfAlgo::GetOpPattern(node);
    const auto &input_reshape_types_json = AnfAlgo::GetAllInputReshapeType(node);
    if (!input_reshape_types_json.empty()) {
      single_json[kAllInputReshapeType] = input_reshape_types_json;
    }
    const auto &output_reshape_types_json = AnfAlgo::GetAllOutputReshapeType(node);
    if (!output_reshape_types_json.empty()) {
      single_json[kAllOutputReshapeType] = output_reshape_types_json;
    }
    const auto &input_kernel_object_types_json = AnfAlgo::GetInputKernelObjectTypes(node);
    if (!input_kernel_object_types_json.empty()) {
      single_json[kInputKernelObjectTypes] = input_kernel_object_types_json;
    }
    const auto &output_kernel_object_types_json = AnfAlgo::GetOutputKernelObjectTypes(node);
    if (!output_kernel_object_types_json.empty()) {
      single_json[kOutputKernelObjectTypes] = output_kernel_object_types_json;
    }
    const auto &output_elements_kernel_object_types_json = AnfAlgo::GetOutputElementsKernelObjectTypes(node);
    if (!output_elements_kernel_object_types_json.empty()) {
      single_json[kOutputElementsKernelObjectTypes] = output_elements_kernel_object_types_json;
    }

    const auto &output_desc_json = AnfAlgo::GetOutputDataDesc(node);
    if (!output_desc_json.empty()) {
      single_json[kOutputDataDesc] = output_desc_json;
    }
    single_json[kFusionType] = AnfAlgo::GetFusionType(node);
    single_json[kProcessor] = AnfAlgo::GetProcessor(node);
    single_json[kKernelBuildInfoValid] = AnfAlgo::GetValid(node);
    single_json[kHasSelectKernelBuildInfo] = true;
  }
  const auto &kernel_info = node->kernel_info();
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &device_kernel_info = dynamic_cast<device::KernelInfo *>(kernel_info);
  MS_EXCEPTION_IF_NULL(device_kernel_info);
  nlohmann::json out_in_ref_json;
  const auto &out_in_ref = device_kernel_info->out_in_ref_map();
  (void)(std::for_each(out_in_ref.begin(), out_in_ref.end(),
                       [&out_in_ref_json](const auto &iter) { out_in_ref_json[iter.first] = iter.second; }));
  if (!out_in_ref_json.empty()) {
    single_json[kOutInRef] = out_in_ref_json;
  }
  const auto &graph_id = device_kernel_info->graph_id();
  single_json[kGraphId] = graph_id;
  const auto &is_feature_map = device_kernel_info->is_feature_map();
  single_json[kIsFeatureMap] = is_feature_map;

  if (node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrIsUBFusionOp, node->cast<CNodePtr>()) &&
      common::AnfAlgo::GetNodeAttr<bool>(node->cast<CNodePtr>(), kAttrIsUBFusionOp)) {
    auto &context = CompileCacheContext::GetInstance();
    const auto &io_size = context.GetIOSizeInfo(node->fullname_with_scope());
    single_json[kJsonName] = io_size.json_name;
    const auto input_size_list_json = io_size.input_size_list;
    if (!input_size_list_json.empty()) {
      single_json[kInputSizeList] = input_size_list_json;
    }
    const auto output_size_list_json = io_size.output_size_list;
    if (!output_size_list_json.empty()) {
      single_json[kOutputSizeList] = output_size_list_json;
    }
  }
  SaveKernelInfoRuntimeCache(kernel_info, &single_json);
  return single_json;
}

nlohmann::json SaveBackendParamToFrontendParamIndex(const KernelGraphPtr &kernel_graph, const FuncGraph *front_graph) {
  nlohmann::json ret;
  const auto &params = kernel_graph->parameters();
  auto &context = CompileCacheContext::GetInstance();
  const auto &front_params = front_graph->parameters();
  for (const auto &param : params) {
    if (!context.IsBackendParamGenFromFrontendParam(param)) {
      continue;
    }
    const auto &name = param->user_data<std::string>(kUniqueCacheName);
    MS_EXCEPTION_IF_NULL(name);
    const auto &front_param = kernel_graph->GetFrontAnfByBackendAnf(param);
    MS_EXCEPTION_IF_NULL(front_param);
    auto iter = std::find(front_params.begin(), front_params.end(), front_param);
    if (iter == front_params.end()) {
      MS_LOG(EXCEPTION) << "Backend param " << param->DebugString() << " correspond frontend param "
                        << front_param->DebugString() << " can not find in frontend graph params.";
    }
    ret[*name] = std::distance(front_params.begin(), iter);
  }
  return ret;
}

void SaveNodesKernelInfoAndParamsName(const KernelGraphPtr &kg, const std::vector<AnfNodePtr> &isolated_nodes,
                                      nlohmann::json *const kg_json) {
  std::vector<AnfNodePtr> nodes = TopoSort(kg->get_return(), SuccIncoming, AlwaysInclude);
  nlohmann::json kernels_info_json;
  (void)(nodes.insert(nodes.end(), isolated_nodes.begin(), isolated_nodes.end()));
  const auto &params = kg->parameters();
  std::vector<AnfNodePtr> isolated_params;
  (void)(std::set_difference(params.begin(), params.end(), nodes.begin(), nodes.end(),
                             std::back_inserter(isolated_params)));
  (void)(nodes.insert(nodes.end(), isolated_params.begin(), isolated_params.end()));
  nlohmann::json param_unique_name_to_name;
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->kernel_info() == nullptr && !node->isa<Parameter>()) {
      continue;
    }
    const auto &name = GetAnfUniqueCacheName(node);
    if (node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      param_unique_name_to_name[name] = param->name();
    }
    if (node->kernel_info() == nullptr) {
      MS_LOG(WARNING) << "The node " << node->DebugString() << " has not kernel_info.";
      continue;
    }
    const auto &kernel_info_json = SaveAnfKernelInfo(node);
    if (!kernel_info_json.empty()) {
      kernels_info_json[name] = kernel_info_json;
    }
  }
  (*kg_json)[kParameterUniqueNameToName] = param_unique_name_to_name;
  (*kg_json)[kNodesKernelInfo] = kernels_info_json;
}

std::vector<nlohmann::json> SaveSummaryNodes(const std::map<std::string, std::pair<AnfNodePtr, int>> &save_map) {
  std::vector<nlohmann::json> ret_json;
  for (const auto &i : save_map) {
    nlohmann::json iter_json;
    const auto &first = i.first;
    const auto &node = i.second.first;
    const auto &name = GetAnfUniqueCacheName(node);
    const auto &index = i.second.second;
    iter_json.push_back(first);
    iter_json.push_back(name);
    iter_json.push_back(index);
    ret_json.push_back(iter_json);
  }
  return ret_json;
}

nlohmann::json GenKernelGraphJson(const KernelGraphPtr &kg, const std::vector<AnfNodePtr> &isolated_nodes) {
  nlohmann::json kg_json;
  SaveNodesKernelInfoAndParamsName(kg, isolated_nodes, &kg_json);
  kg_json[kGraphId] = kg->graph_id();
  kg_json[kRunMode] = kg->RunMode();
  kg_json[kIsLoopCountSink] = kg->is_loop_count_sink();
  kg_json[kIsDynamicShape] = kg->is_dynamic_shape();
  kg_json[kDeviceTarget] = kg->device_target();
  kg_json[kRootGraphId] = kg->root_graph_id();
  kg_json[kExecutable] = kg->executable();
  kg_json[kHasRecursiveCall] = kg->recursive_call();
  kg_json[kHasSubgraphMultiCall] = kg->subgraph_multi_call();
  kg_json[kNeedInline] = kg->need_inline();
  kg_json[kIsNeedGil] = kg->is_need_gil();
  kg_json[kIsFromSingleOp] = kg->is_from_single_op();
  kg_json[kLabelNum] = kg->label_num();
#ifndef ENABLE_SECURITY
  kg_json[kSummaryNodeExist] = kg->summary_node_exist();
#endif
  const auto &back_front_anf_json = SaveAnfToAnfMap(kg->backend_front_anf_map());
  if (!back_front_anf_json.empty()) {
    kg_json[kBackendFrontAnf] = back_front_anf_json;
  }
  const auto &internal_params_to_front_node_json = SaveAnfToAnfIndexMap(kg->InternalParameterToFrontNodeMap());
  if (!internal_params_to_front_node_json.empty()) {
    kg_json[kInternalParameterToFrontNode] = internal_params_to_front_node_json;
  }
  const auto &ref_in_out_map_json = SaveAnfIndexToAnfIndexMap(kg->GetRefMap());
  if (!ref_in_out_map_json.empty()) {
    kg_json[kRefInOutMap] = ref_in_out_map_json;
  }
  const auto &graph_value_nodes = SaveValueSet(kg->graph_value_nodes());
  if (!graph_value_nodes.empty()) {
    kg_json[kGraphValueNodes] = graph_value_nodes;
  }
  const auto &exec_order_json = SaveAnfVec(kg->execution_order());
  if (!exec_order_json.empty()) {
    kg_json[kExecutionOrder] = exec_order_json;
  }
  const auto &inputs_json = SaveAnfVec(kg->inputs());
  if (!inputs_json.empty()) {
    kg_json[kInputs] = inputs_json;
  }
  const auto &parameters_json = SaveAnfVec(kg->parameters());
  if (!parameters_json.empty()) {
    kg_json[kParameters] = parameters_json;
  }
  const auto &child_graph_result_json = SaveAnfVec(kg->child_graph_result());
  if (!child_graph_result_json.empty()) {
    kg_json[kChildGraphResult] = child_graph_result_json;
  }
  const auto &child_graph_order_json = SaveGraphVec(kg->child_graph_order());
  if (!child_graph_order_json.empty()) {
    kg_json[kChildGraphOrder] = child_graph_order_json;
  }
  const auto &start = kg->get_start_label();
  if (start) {
    kg_json[kStartLabel] = GetAnfUniqueCacheName(start);
  }
  const auto &end = kg->get_end_goto();
  if (end) {
    kg_json[kEndGoto] = GetAnfUniqueCacheName(end);
  }
  const auto &valid_inputs = kg->valid_inputs();
  if (!valid_inputs.empty()) {
    kg_json[kValidInputs] = valid_inputs;
  }
  const auto &pre_graphs_json = SaveGraphsId(kg->get_pre_graphs());
  if (!pre_graphs_json.empty()) {
    kg_json[kPreGraphs] = pre_graphs_json;
  }
  const auto &post_graphs_json = SaveGraphsId(kg->GetPostGraphs());
  if (!post_graphs_json.empty()) {
    kg_json[kPostGraphs] = post_graphs_json;
  }
  const auto &index_set = kg->CommSubGraphIds();
  if (!index_set.empty()) {
    kg_json[kCommSubGraphIds] = index_set;
  }
#ifndef ENABLE_SECURITY
  const auto &summary_nodes_json = SaveSummaryNodes(kg->summary_nodes());
  if (!summary_nodes_json.empty()) {
    kg_json[kSummaryNodes] = summary_nodes_json;
  }
#endif
  auto &context = CompileCacheContext::GetInstance();
  auto front_graph = context.GetFrontendGraphByBackendGraph(kg);
  if (front_graph) {
    kg_json[kCorrespondFrontendGraph] = front_graph->ToString();
  }
  kg_json[kBackendParamToFrontendParamIndex] = SaveBackendParamToFrontendParamIndex(kg, front_graph);
  return kg_json;
}

bool DumpKernelGraphJson(const KernelGraphPtr &root_graph, const std::set<KernelGraphPtr> &child_graphs,
                         const std::map<KernelGraphPtr, std::vector<AnfNodePtr>> &isolated_nodes_map,
                         const std::string &path) {
  nlohmann::json kg_json;
  kg_json[root_graph->ToString()] = GenKernelGraphJson(root_graph, isolated_nodes_map.find(root_graph)->second);
  for (const auto &graph : child_graphs) {
    kg_json[graph->ToString()] = GenKernelGraphJson(graph, isolated_nodes_map.find(graph)->second);
  }
  return Common::SaveStringToFile(path, kg_json.dump());
}

void GetAllChildGraph(const KernelGraphPtr &kg, std::set<KernelGraphPtr> *visit, std::set<KernelGraphPtr> *graphs) {
  if (kg == nullptr || kg->IsLeafGraph()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(visit);
  MS_EXCEPTION_IF_NULL(graphs);
  if (visit->find(kg) != visit->end()) {
    return;
  }
  const auto &order = kg->child_graph_order();
  for (auto iter : order) {
    auto graph = iter.lock();
    MS_EXCEPTION_IF_NULL(graph);
    (void)(graphs->insert(graph));
  }
  (void)(visit->insert(kg));

  for (auto &i : order) {
    GetAllChildGraph(i.lock(), visit, graphs);
  }
}

void GetIsolatedNodes(const KernelGraphPtr &kg, std::vector<AnfNodePtr> *isolated_nodes) {
  MS_EXCEPTION_IF_NULL(kg);
  const auto &orders = kg->execution_order();
  std::vector<AnfNodePtr> possible_isolated(orders.begin(), orders.end());
  const auto &start = kg->get_start_label();
  if (start && std::find(possible_isolated.begin(), possible_isolated.end(), start) == possible_isolated.end()) {
    possible_isolated.push_back(start);
  }
  const auto &end = kg->get_end_goto();
  if (end && std::find(possible_isolated.begin(), possible_isolated.end(), end) == possible_isolated.end()) {
    possible_isolated.push_back(end);
  }
  auto topo_nodes = TopoSort(kg->get_return(), SuccIncoming, AlwaysInclude);
  (void)(std::set_difference(possible_isolated.begin(), possible_isolated.end(), topo_nodes.begin(), topo_nodes.end(),
                             std::back_inserter(*isolated_nodes)));
}

void HandleParamExistCorrespondFrontendParam(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &context = CompileCacheContext::GetInstance();
  const auto &front_graph = context.GetFrontendGraphByBackendGraph(graph);
  if (!front_graph) {
    return;
  }
  const auto &params = graph->parameters();
  const auto &front_params = front_graph->parameters();
  for (const auto &param : params) {
    auto front_param = graph->GetFrontAnfByBackendAnf(param);
    if (!front_param) {
      continue;
    }
    auto iter = std::find(front_params.begin(), front_params.end(), front_param);
    if (iter != front_params.end()) {
      context.InsertBackendParamGenFromFrontendParam(param);
    }
  }
}

bool NeedConvertValueNodeToParameter(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() != "ge") {
    return false;
  }
  if (!node->isa<ValueNode>()) {
    return false;
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->is_forward_output()) {
      return true;
    }
  }
  return false;
}

void ConvertValueNodeToParameter(const KernelGraphPtr &graph, const AnfNodePtr &node,
                                 std::vector<ParameterPtr> *added_parameters) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(added_parameters);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  auto new_parameter = graph->NewParameter(node->abstract());
  MS_EXCEPTION_IF_NULL(new_parameter);
  new_parameter->IncreaseUsedGraphCount();
  graph_inputs->push_back(new_parameter);

  MS_EXCEPTION_IF_NULL(node->cast<ValueNodePtr>());
  new_parameter->set_user_data(kForwardOutput, node->cast<ValueNodePtr>()->value());
  graph->FrontBackendMapAdd(node, new_parameter);
  (void)added_parameters->emplace_back(new_parameter);
  MS_LOG(DEBUG) << "Replace ValueNode " << node->DebugString() << " with Parameter " << new_parameter->DebugString();
}
}  // namespace

ParamInfoPtr GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->param_info();
}

#ifndef ENABLE_SECURITY
bool ExistSummaryNode(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &n : TopoSort(graph->get_return())) {
    if (AnfAlgo::IsSummaryNode(n)) {
      return true;
    }
  }
  return false;
}
#endif

GraphId KernelGraphMgr::graph_sum_ = 0;
GraphId KernelGraphMgr::pynative_graph_sum_ = 0;
HashMap<std::string, std::weak_ptr<AnfNode>> KernelGraphMgr::name_to_params_ =
  HashMap<std::string, std::weak_ptr<AnfNode>>();

ValueNodePtr KernelGraphMgr::CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  // Copy data from device if the tensor is an output of Op or Graph.
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (!tensor->is_forward_output() && !tensor->is_parameter()) {
      tensor->data_sync();
      MS_LOG(INFO) << "Data sync for Tensor " << tensor->ToString();
    }
  }
  auto new_value_node = value_node;
  if (!graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    new_value_node = graph->NewValueNode(value_node);
    graph->FrontBackendMapAdd(anf, new_value_node);
  }
  graph->AddValueNodeToGraph(new_value_node);
  return new_value_node;
}

GraphId KernelGraphMgr::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
  for (const auto &graph_item : graphs_) {
    auto graph = graph_item.second;
    MS_EXCEPTION_IF_NULL(graph);
    // if front_anf is a parameter,the backend parameter may have two
    if (graph->GetBackendAnfByFrontAnf(front_anf) != nullptr) {
      return graph_item.first;
    }
  }
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_LOG(DEBUG) << "Front_anf " << front_anf->DebugString() << " is not exist in any graph";
  return kInvalidGraphId;
}

KernelGraphPtr KernelGraphMgr::GetGraph(mindspore::GraphId graph_id) const {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(INFO) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void KernelGraphMgr::ClearGraph() {
  auto graph_iter = graphs_.begin();
  while (graph_iter != graphs_.end()) {
    graph_iter->second.reset();
    graph_iter = graphs_.erase(graph_iter);
  }
  graph_sum_ = 0;
  pynative_graph_sum_ = 0;
}

void KernelGraphMgr::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) const {
  MS_EXCEPTION_IF_NULL(out_node);
  MS_EXCEPTION_IF_NULL(parameter);
  MS_LOG(DEBUG) << "parameter:" << parameter->DebugString()
                << " abstract:" << (parameter->abstract() != nullptr ? parameter->abstract()->ToString() : "null");
  auto graph_id = GetGraphIdByNode(out_node);
  if (graph_id == kInvalidGraphId) {
    return;
  }
  auto node_graph = GetGraph(graph_id);
  if (node_graph == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Init parameter with pre graph output node: " << out_node->DebugString();
  auto ref_node_with_index = node_graph->GetInternalOutputByFrontNode(out_node);
  auto ref_node = ref_node_with_index.first;
  if (ref_node == nullptr) {
    MS_LOG(INFO) << "No corresponding internal output for output node";
    return;
  }
  size_t output_idx = ref_node_with_index.second;
  if (common::AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = common::AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = common::AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  MS_EXCEPTION_IF_NULL(ref_real_node);
  auto ref_real_node_index = real_kernel.second;
  if (ref_real_node->isa<CNode>() && node_graph->IsUniqueTargetInternalOutput(ref_real_node, ref_real_node_index)) {
    auto kernel_info = ref_real_node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      MS_LOG(INFO) << "No kernel info";
      return;
    }
    if (!common::AnfAlgo::IsNopNode(ref_real_node) && !AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index)) {
      MS_LOG(INFO) << "No kernel address";
      return;
    }
    if (!AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index, true)) {
      return;
    }

    // Update the kernel build info.
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    if (type == TypeId::kTypeUnknown) {
      return;
    }
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), parameter.get());

    // If the flag is enable, it means the graph would run in subgraph sink mode, the internal parameter cannot share
    // the same device address.
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index, false);
    if (!node_graph->has_flag(kFlagEnableZeroCopyInGraph)) {
      AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    }

    abstract::AbstractBasePtr abstract;
    auto shape = parameter->Shape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->isa<abstract::NoShape>()) {
      abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(type));
    } else if (shape->isa<abstract::DynamicSequenceShape>()) {
      return;
    } else {
      abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), shape->cast<abstract::BaseShapePtr>());
    }
    if (!parameter->abstract()->isa<abstract::AbstractAny>()) {
      parameter->set_abstract(abstract);
    }
  }
}

AnfNodePtr KernelGraphMgr::CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto new_parameter = graph->TransTupleToMakeTuple(graph->NewParameter(node->abstract()));
  auto parameters = common::AnfAlgo::GetAllOutput(new_parameter);
  std::vector<AnfNodePtr> pre_graph_out = {node};
  // If a cnode is a call, it's input0 is a cnode too, so it doesn't have primitive
  if (!pre_graph_out.empty() && !AnfUtils::IsRealKernel(node)) {
    pre_graph_out = common::AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem, prim::kPrimUpdateState});
  }

  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &parameter = parameters[i];
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      // In control flow, if the input of the cnode is a call node, it will be processed as a make_tuple input,
      // which needs to be linked when processing the internal node.
      graph->CacheInternalParameterToFrontNode(parameter, {node, i});
    }
    auto valid_inputs = graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    valid_inputs->push_back(true);
    graph_inputs->push_back(parameter);
  }
  size_t param_index = 0;
  for (const auto &out_node : pre_graph_out) {
    size_t output_size = AnfAlgo::GetOutputElementNum(out_node);
    for (size_t i = 0; i < output_size; i++) {
      if (param_index >= parameters.size()) {
        MS_LOG(EXCEPTION) << "Parameters size:" << parameters.size() << "out of range.Node:" << node->DebugString()
                          << ",out_node:" << out_node->DebugString();
      }
      InitInternalOutputParameter(out_node, parameters[param_index++]);
    }
  }
  return new_parameter;
}

ParameterPtr KernelGraphMgr::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_value = GetParamDefaultValue(anf);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  ParameterPtr new_parameter = nullptr;
  auto func_graph = anf->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  bool is_pynative_bprop_kernel_graph = graph->has_flag(kFlagIsPyNativeBpropKernelGraph);
  if (func_graph->manager() != nullptr && func_graph->exist_multi_target() &&
      graph->device_target() == device::DeviceType::kCPU) {
    auto iter = default_param_map_.find(anf);
    if (iter != default_param_map_.end()) {
      new_parameter = iter->second;
    }
    if (new_parameter != nullptr) {
      graph_inputs->push_back(new_parameter);
      MS_LOG(DEBUG) << "create new parameter for parameter:" << anf->DebugString() << " for graph:" << graph->ToString()
                    << " backend node:" << new_parameter->DebugString();
      return new_parameter;
    }
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = anf->cast<ParameterPtr>();
    if (!is_pynative_bprop_kernel_graph) {
      new_parameter = graph->NewParameter(new_parameter);
    }
    graph_inputs->push_back(new_parameter);
    valid_inputs->push_back(true);
    default_param_map_[anf] = new_parameter;
    return new_parameter;
  }
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (!is_pynative_bprop_kernel_graph) {
    if (param_value != nullptr) {
      new_parameter = param_value->parameter();
    }
    if (new_parameter == nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
      new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());

      auto input_node_iter = partial_parameters_map_.find(anf);
      if (input_node_iter != partial_parameters_map_.end()) {
        InitInternalOutputParameter(input_node_iter->second, new_parameter);
      }

      if (param_value != nullptr) {
        param_value->set_parameter(new_parameter);
      }
    }
    new_parameter->IncreaseUsedGraphCount();
  } else {
    new_parameter = anf->cast<ParameterPtr>();
  }
  (void)graph_inputs->emplace_back(new_parameter);
  (void)valid_inputs->emplace_back(true);
  return new_parameter;
}

AnfNodePtr KernelGraphMgr::CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  return CreateParameterFromTuple(anf, graph);
}

void KernelGraphMgr::GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    // push attr to inputs[0] of new cnode
    cnode_inputs->push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
  } else {
    auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs->push_back(std::make_shared<ValueNode>(new_fg));
  }
}

AnfNodePtr KernelGraphMgr::GetChildGraph(KernelGraph *graph, const AnfNodePtr &child_func_graph) {
  MS_EXCEPTION_IF_NULL(child_func_graph);
  std::vector<KernelGraphPtr> all_graphs;
  FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(child_func_graph);
  MS_EXCEPTION_IF_NULL(child_graph);
  if (front_backend_graph_map_.find(child_graph.get()) == front_backend_graph_map_.end()) {
    (void)ConstructKernelGraph(child_graph, &all_graphs, graph->device_target());
  }
  auto new_value_node = graph->GetBackendAnfByFrontAnf(child_func_graph);
  if (new_value_node != nullptr) {
    return new_value_node;
  }
  new_value_node = CreateValueNodeKernelGraph(child_func_graph, graph);
  MS_EXCEPTION_IF_NULL(new_value_node);
  return new_value_node;
}

namespace {
void AddValueNode(const AnfNodePtr &backend_node, KernelGraph *graph) {
  if (backend_node->isa<ValueNode>() && !IsValueNode<FuncGraph>(backend_node)) {
    graph->AddValueNodeToGraph(backend_node->cast<ValueNodePtr>());
  }
}
}  // namespace

void KernelGraphMgr::GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                                       mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto origin_inputs = cnode->inputs();
  const bool is_depend = IsPrimitiveCNode(cnode, prim::kPrimDepend);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool enable_ge = context->backend_policy() == "ge";
  AnfNodePtr child_func_graph = nullptr;
  std::vector<AnfNodePtr> params;
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < origin_inputs.size(); input_idx++) {
    auto anf = origin_inputs[input_idx];
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      const auto &backend_node = graph->GetBackendAnfByFrontAnf(anf);
      (void)params.emplace_back(backend_node);
      AddValueNode(backend_node, graph);
      continue;
    } else if ((is_depend && input_idx > kRealInputIndexInDepend && !enable_ge)) {
      (void)params.emplace_back(graph->NewValueNode(std::make_shared<Tensor>(SizeToInt(input_idx))));
      continue;
    } else if (other_graph_cnode->find(anf) != other_graph_cnode->end()) {
      (void)params.emplace_back((*other_graph_cnode)[anf]);
      continue;
    } else if (anf->isa<ValueNode>() && !IsValueNode<FuncGraph>(anf)) {
      // if input is a value node,
      auto new_value_node = CreateNewValueNode(anf, graph);
      if (new_value_node != nullptr) {
        (void)params.emplace_back(new_value_node);
      }
      continue;
    } else if (anf->isa<Parameter>()) {
      auto new_parameter = CreateNewParameterFromParameter(anf, graph);
      MS_EXCEPTION_IF_NULL(new_parameter);
      MS_LOG(DEBUG) << "Create new parameter:" << new_parameter->DebugString()
                    << " by front parameter:" << anf->DebugString();
      (void)params.emplace_back(new_parameter);
      graph->FrontBackendMapAdd(anf, new_parameter);
      continue;
    } else if (IsValueNode<FuncGraph>(anf) && cnode->HasPrimalAttr(kAttrNotCut)) {
      MS_EXCEPTION_IF_CHECK_FAIL(input_idx == 1, "Graph input index is not 1, anf: " + anf->DebugString() +
                                                   ", index: " + std::to_string(input_idx));
      child_func_graph = anf;
      continue;
    } else {
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, graph);
      if (parameter_from_cnode == nullptr) {
        parameter_from_cnode = NewValueNode(MakeValue(SizeToLong(input_idx)));
      }
      MS_EXCEPTION_IF_NULL(parameter_from_cnode);
      MS_LOG(DEBUG) << "graph:" << graph->ToString() << " front node:" << anf->DebugString()
                    << " abstract:" << (anf->abstract() != nullptr ? anf->abstract()->ToString() : "null")
                    << " parameter:" << parameter_from_cnode->DebugString() << " abstract:"
                    << (parameter_from_cnode->abstract() != nullptr ? parameter_from_cnode->abstract()->ToString()
                                                                    : "null");
      if (parameter_from_cnode->isa<Parameter>() && IsPrimitiveCNode(anf, prim::kPrimLoad)) {
        auto para = parameter_from_cnode->cast<ParameterPtr>();
        auto load_cnode = anf->cast<CNodePtr>();
        para->set_name(load_cnode->fullname_with_scope());
      }
      (void)params.emplace_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
    }
  }

  if (child_func_graph != nullptr) {
    (void)cnode_inputs->emplace_back(GetChildGraph(graph, child_func_graph));
  }
  (void)std::copy(params.begin(), params.end(), std::back_inserter(*cnode_inputs));
}

CNodePtr KernelGraphMgr::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                                        mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  auto primitive_input = cnode->input(kAnfPrimitiveIndex);
  // control flow sink to GE
  bool need_control_flow_sink =
    IsPrimitiveCNode(primitive_input, prim::kPrimSwitch) && cnode->HasPrimalAttr(kAttrNotCut);
  // backend inline
  bool need_backend_inline = cnode->HasPrimalAttr(kAttrNeedInline);
  if (need_backend_inline) {
    auto fn = cnode->input(kAnfPrimitiveIndex);
    MS_EXCEPTION_IF_NULL(fn);
    if (IsValueNode<FuncGraph>(fn)) {
      // Need to create a new kernel graph
      (void)GetChildGraph(graph, fn);
    }
  }
  if (need_control_flow_sink || need_backend_inline) {
    auto new_cnode = CreateNewCNode(cnode, graph);
    MS_EXCEPTION_IF_NULL(new_cnode);
    FlattenTuple(new_cnode);
    if (need_backend_inline) {
      new_cnode->AddPrimalAttr(kAttrNeedInline, MakeValue(true));
    }
    return new_cnode;
  }
  // get primitive of old node
  std::vector<AnfNodePtr> cnode_inputs;
  GetCNodeInfo(cnode, &cnode_inputs);
  GetNewCNodeInputs(cnode, graph, &cnode_inputs, other_graph_cnode);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNodeWithInfos(cnode_inputs, cnode);
  return new_cnode;
}

CNodePtr KernelGraphMgr::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (IsValueNode<FuncGraph>(attr_input)) {
    // cnode is a graph or a call
    cnode_inputs = CreateValueNode(cnode, graph);
  } else if (attr_input->isa<CNode>()) {
    // cnode ia a call (partial/switch/switch_layer)
    // 1. take the args of call to the partial node, as the real_args to call switch's or switch_layer's child graph
    // 2. the call in frontend is map to the partial/switch/switch_layer in backend and haven't been created
    cnode_inputs = CreateSwitchOrPartialNode(cnode, graph);
    if (cnode_inputs.empty()) {
      MS_LOG_ERROR << "Create switch or partial failed, cnode:" << cnode->DebugString();
      return nullptr;
    }
  } else {
    // get primitive of old node
    auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }
  // handle inputs of cnode except primitive
  CreateCNodeInputs(cnode, graph, &cnode_inputs);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNodeWithInfos(cnode_inputs, cnode);
  MS_EXCEPTION_IF_NULL(new_cnode);
  // if the cnode is call switch, remove call
  if (new_cnode->size() > 1) {
    auto first_input = new_cnode->input(kFirstDataInputIndex);
    MS_EXCEPTION_IF_NULL(first_input);
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitch)) {
      new_cnode = first_input->cast<CNodePtr>();
    }
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitchLayer)) {
      auto abstract = cnode->abstract();
      new_cnode = first_input->cast<CNodePtr>();
      new_cnode->set_abstract(abstract);
    }
  }
  return new_cnode;
}

CNodePtr KernelGraphMgr::CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (common::AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto backend_node = graph->GetBackendAnfByFrontAnf(node_input);
    MS_EXCEPTION_IF_NULL(backend_node);
    return backend_node->cast<CNodePtr>();
  } else if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    (void)(partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input)));
  } else {
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto parameter = CreateNewParameterFromCNode(cnode, kernel_graph.get());
    MS_EXCEPTION_IF_NULL(parameter);
    MS_EXCEPTION_IF_NULL(cnode);
    parameter->set_abstract(cnode->abstract());
    auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()));
    auto return_node = kernel_graph->NewCNode({primitive, parameter});
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    (void)(partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph)));
    (void)(partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input)));
  }
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

void KernelGraphMgr::CacheKernelGraph(const KernelGraphPtr &kg) {
  MS_EXCEPTION_IF_NULL(kg);
  auto &context = CompileCacheContext::GetInstance();
  auto fg = context.FrontGraph();
  if (!fg) {
    MS_LOG(EXCEPTION) << "The frontend graph to be cached is null";
  }
  if (!kg) {
    MS_LOG(EXCEPTION) << "The backend graph to be cached is null";
  }
  MS_LOG(INFO) << "Begin to cache kernel graph " << kg->ToString();
  std::set<KernelGraphPtr> visit;
  std::set<KernelGraphPtr> child_graphs;
  GetAllChildGraph(kg, &visit, &child_graphs);
#ifdef ENABLE_DUMP_IR
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->CanDump(kIntroductory)) {
    DumpIR("compile_cache_" + kg->ToString() + ".ir", kg);
    for (auto &graph : child_graphs) {
      DumpIR("compile_cache_" + graph->ToString() + ".ir", graph);
    }
  }
#endif
  std::vector<AnfNodePtr> temp_nodes;
  std::map<KernelGraphPtr, std::vector<AnfNodePtr>> isolated_nodes_map;
  HandleParamExistCorrespondFrontendParam(kg);
  GetIsolatedNodes(kg, &temp_nodes);
  isolated_nodes_map[kg] = temp_nodes;
  auto cache_path = context.GetBackendGraphCachePath(fg);
  const std::string &mindir_path = cache_path + kMindIrSuffix;
  for (const auto &graph : child_graphs) {
    temp_nodes.clear();
    HandleParamExistCorrespondFrontendParam(graph);
    GetIsolatedNodes(graph, &temp_nodes);
    isolated_nodes_map[graph] = temp_nodes;
  }
  std::vector<AnfNodePtr> isolated_nodes;
  for (const auto &iter : isolated_nodes_map) {
    const auto &nodes = iter.second;
    (void)(isolated_nodes.insert(isolated_nodes.end(), nodes.begin(), nodes.end()));
  }
  std::vector<FuncGraphPtr> child_graphs_for_dump(child_graphs.begin(), child_graphs.end());
  if (!DumpBinaryProto(kg, child_graphs_for_dump, isolated_nodes, mindir_path)) {
    MS_LOG(ERROR) << "Failed to cache kernel graph to mindir: " << fg->ToString();
    return;
  }
  (void)(std::for_each(front_backend_graph_map_.begin(), front_backend_graph_map_.end(),
                       [&context](const auto &fb) { context.AddBackendGraphToFrontendGraph(fb.second, fb.first); }));
  const std::string &json_path = cache_path + kJsonSuffix;
  if (!DumpKernelGraphJson(kg, child_graphs, isolated_nodes_map, json_path)) {
    MS_LOG(ERROR) << "Failed to cache kernel graph to json.";
    return;
  }
  context.Clear();
  MS_LOG(INFO) << "Cache kernel graph " << kg->ToString() << " success.";
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  MS_EXCEPTION_IF_NULL(cnode_input);
  auto switch_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  if (cnode->size() <= 1) {
    cnode_inputs = switch_cnode->inputs();
    return cnode_inputs;
  }
  std::vector<AnfNodePtr> switch_inputs = {switch_cnode->input(kAnfPrimitiveIndex),
                                           switch_cnode->input(kFirstDataInputIndex)};
  for (size_t index = kSwitchTrueBranchIndex; index < switch_cnode->size(); index++) {
    auto node = switch_cnode->input(index);
    MS_EXCEPTION_IF_NULL(node);
    // there is real input in call, should put it to true and false branch in switch
    if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      auto partial_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      std::vector<AnfNodePtr> partial_inputs = partial_node->inputs();
      // Put all call args at the end of partial inputs.
      for (size_t i = kFirstDataInputIndex; i < cnode->size(); ++i) {
        (void)partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(i)));
      }
      auto new_partial = graph->NewCNode(partial_inputs);
      (void)switch_inputs.emplace_back(new_partial);
    }
  }
  if (switch_inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch inputs size: " << switch_inputs.size() << "less than " << kSwitchInputSize;
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  (void)cnode_inputs.emplace_back(switch_node);
  return cnode_inputs;
}

void KernelGraphMgr::ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph,
                                        const std::vector<AnfNodePtr> &real_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  // func1 =switch(branch1, branch2)
  // func2 = func1(param1)
  // out = func2(param2)
  // process the last cnode(func2), not func1 which abstract is AbstractFunction
  if (cnode->abstract()->isa<abstract::AbstractFunction>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto return_input = ret->input(kFirstDataInputIndex);
  // return node is a function
  std::vector<AnfNodePtr> call_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial)) {
    auto return_input_cnode = return_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(return_input_cnode);
    auto partial_inputs = return_input_cnode->inputs();
    (void)call_inputs.insert(call_inputs.cend(), partial_inputs.cbegin() + kFirstDataInputIndex, partial_inputs.cend());
  } else if (IsValueNode<KernelGraph>(return_input)) {  // return node is kernel graph
    (void)(call_inputs.emplace_back(return_input));
  } else {  // return node is value node
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto valid_inputs = kernel_graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = kernel_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    std::vector<AnfNodePtr> cnode_inputs = {return_input};
    for (auto &real_input : real_inputs) {
      auto new_parameter = kernel_graph->NewParameter(real_input->abstract());
      valid_inputs->push_back(true);
      graph_inputs->push_back(new_parameter);
      cnode_inputs.push_back(new_parameter);
    }
    auto new_cnode = kernel_graph->NewCNode(cnode_inputs);
    new_cnode->set_abstract(cnode->abstract());
    std::vector<AnfNodePtr> return_inputs = {
      kernel_graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()))), new_cnode};
    auto return_node = kernel_graph->NewCNode(return_inputs);
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    call_inputs.push_back(std::make_shared<ValueNode>(kernel_graph));
  }

  // new call node inputs
  for (auto &input_node : real_inputs) {
    auto parameter_for_input = CreateNewParameterFromCNode(input_node, graph);
    (void)(call_inputs.emplace_back(parameter_for_input));
  }

  auto call_node = graph->NewCNode(call_inputs);
  MS_EXCEPTION_IF_NULL(call_node);
  call_node->set_abstract(cnode->abstract());
  // update return input
  ret->set_input(kFirstDataInputIndex, call_node);
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  MS_EXCEPTION_IF_NULL(cnode_input);
  auto switch_layer_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_layer_cnode);
  std::vector<AnfNodePtr> switch_layer_inputs = {switch_layer_cnode->input(kAnfPrimitiveIndex),
                                                 switch_layer_cnode->input(kFirstDataInputIndex)};
  auto make_tuple_node = switch_layer_cnode->input(kSwitchLayerBranchesIndex);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  auto node = make_tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto make_tuple_inputs = node->inputs();
  // there are real inputs in call, should put it to make_tuple in switch_layer
  std::vector<AnfNodePtr> real_inputs;
  for (size_t idx = kFirstDataInputIndex; idx < cnode->size(); ++idx) {
    (void)(real_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(idx))));
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())))};
  for (size_t idx = kFirstDataInputIndex; idx < make_tuple_inputs.size(); idx++) {
    auto partial_idx = make_tuple_inputs[idx];
    MS_EXCEPTION_IF_NULL(cnode->abstract());
    std::vector<AnfNodePtr> new_partial_inputs;
    KernelGraphPtr partial_kernel_graph;
    // switch_layer node input is partial cnode
    if (common::AnfAlgo::CheckPrimitiveType(partial_idx, prim::kPrimPartial)) {
      auto partial_node = partial_idx->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      auto partial_input = partial_node->input(kFirstDataInputIndex);
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_input);
      new_partial_inputs = partial_node->inputs();
    } else if (IsValueNode<KernelGraph>(partial_idx)) {  // switch_layer node input is kernel graph value node
      (void)(new_partial_inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))));
      (void)(new_partial_inputs.emplace_back(partial_idx));
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_idx);
    }
    // when branch in swich_layer return function
    MS_EXCEPTION_IF_NULL(partial_kernel_graph);
    auto ret = partial_kernel_graph->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    auto return_input = ret->input(kFirstDataInputIndex);
    if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial) || return_input->isa<ValueNode>()) {
      ProcessNodeRetFunc(cnode, partial_kernel_graph.get(), real_inputs);
    }
    // partial node add input args
    (void)new_partial_inputs.insert(new_partial_inputs.cend(), real_inputs.cbegin(), real_inputs.cend());
    // create new partial node
    auto new_partial = graph->NewCNode(new_partial_inputs);
    (void)(new_make_tuple_inputs.emplace_back(new_partial));
  }
  auto new_make_tuple = graph->NewCNode(new_make_tuple_inputs);
  auto abstract = make_tuple_node->abstract();
  if (abstract == nullptr) {
    abstract = std::make_shared<abstract::AbstractTuple>(AbstractBasePtrList());
  }
  new_make_tuple->set_abstract(abstract);
  (void)(switch_layer_inputs.emplace_back(new_make_tuple));
  auto new_switch_layer = graph->NewCNode(switch_layer_inputs);
  (void)(cnode_inputs.emplace_back(new_switch_layer));
  return cnode_inputs;
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  // create primitive of cnode:call(partial or switch or switch_layer)
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  if (cnode_input == nullptr) {
    MS_LOG(ERROR) << "CNode input[0] is CNode:" << attr_input->DebugString() << ", but input[0] has not been created.";
    return {};
  }
  // if the node is partial, insert the inputs of partial to the call
  if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    (void)std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                         std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                           MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                           return graph->GetBackendAnfByFrontAnf(node);
                         });
    return cnode_inputs;
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    return CreateCallSwitchInputs(cnode, graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitchLayer)) {
    return CreateCallSwitchLayerInputs(cnode, graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimTupleGetItem)) {
    // only support tuple get item from a call subgraph output
    auto tuple_get_node = cnode_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_node);
    auto get_from_node = tuple_get_node->input(kFirstIndex);
    MS_EXCEPTION_IF_NULL(get_from_node);
    if (common::AnfAlgo::CheckPrimitiveType(get_from_node, prim::kPrimCall)) {
      auto call_node = get_from_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(call_node);
      auto call_graph = call_node->input(kFirstIndex);
      auto sub_kernel_graph = AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
      MS_EXCEPTION_IF_NULL(sub_kernel_graph);
      if (kernel_graph_partial_map_.find(sub_kernel_graph.get()) == kernel_graph_partial_map_.end()) {
        MS_LOG(EXCEPTION) << "Kernel Graph: " << sub_kernel_graph->ToString()
                          << " has not a return value is a Partial Func.";
      }
      auto tuple_get_idx = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_node);
      auto info = kernel_graph_partial_map_[sub_kernel_graph.get()];
      call_node->set_abstract(info.abstract);
      (void)cnode_inputs.emplace_back(info.sub_graph);
      auto context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context);
      if (context->CellReuseLevel() == CellReuseLevel::kLazyInline) {
        // call_graph and info.sub_graph need inline when cell reuse.
        sub_kernel_graph->set_need_inline(true);
        auto partial_sub_graph = AnfRuntimeAlgorithm::GetValueNodeKernelGraph(info.sub_graph);
        MS_EXCEPTION_IF_NULL(partial_sub_graph);
        partial_sub_graph->set_need_inline(true);
        MS_LOG(INFO) << "Inline graph " << sub_kernel_graph->graph_id() << " and graph "
                     << partial_sub_graph->graph_id();
      }
      MS_LOG(INFO) << "Use cell reuse: " << sub_kernel_graph->graph_id();
      if (info.param_begin != tuple_get_idx + std::max(static_cast<int>(info.multi_tuple) - 1, 0)) {
        MS_LOG(EXCEPTION) << "Call param is not a graph, the TupleGetItem index: " << tuple_get_idx
                          << ", the partial graph index: " << info.param_begin
                          << ", need idx: " << tuple_get_idx + std::max(static_cast<int>(info.multi_tuple) - 1, 0)
                          << ", call graph: " << call_graph->fullname_with_scope();
      }
      for (size_t i = info.param_begin; i < info.param_end; i++) {
        auto idx = NewValueNode(SizeToLong(i));
        MS_EXCEPTION_IF_NULL(idx);
        auto imm = std::make_shared<Int64Imm>(i);
        idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
        auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_node, idx});
        std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(call_node, i)};
        auto shapes = {common::AnfAlgo::GetOutputInferShape(call_node, i)};
        common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
        (void)cnode_inputs.emplace_back(getitem);
      }
      return cnode_inputs;
    }
  }
  MS_LOG(ERROR) << "CNode:" << cnode->DebugString() << " input[0]" << cnode_input->DebugString()
                << "must be partial or switch or switch_layer.";
  return {};
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateValueNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (common::AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  } else {
    // create primitive of cnode:call
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
    // create a ValueNode<KernelGraph> as input of cnode:call
    if (graph->GetBackendAnfByFrontAnf(attr_input) != nullptr) {
      (void)(cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(attr_input)));
    } else {
      auto new_value_node = CreateValueNodeKernelGraph(attr_input, graph);
      if (new_value_node != nullptr) {
        (void)(cnode_inputs.emplace_back(new_value_node));
      }
    }
  }
  return cnode_inputs;
}

void KernelGraphMgr::CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph,
                                       std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
    for (size_t index = kSwitchTrueBranchIndex; index < cnode->size(); index++) {
      auto node_input = cnode->input(index);
      auto switch_input = CreateSwitchInput(cnode, node_input, graph);
      (void)cnode_inputs->emplace_back(switch_input);
    }
  } else {
    for (size_t input_idx = kFirstDataInputIndex; input_idx < cnode->size(); input_idx++) {
      auto anf = cnode->input(input_idx);
      MS_EXCEPTION_IF_NULL(anf);
      // anf has been created before
      if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
        (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
        continue;
      } else if (anf->isa<Parameter>()) {
        auto new_parameter = CreateNewParameterFromParameter(anf, graph);
        MS_EXCEPTION_IF_NULL(new_parameter);
        (void)cnode_inputs->emplace_back(new_parameter);
        graph->FrontBackendMapAdd(anf, new_parameter);
        continue;
      } else if (anf->isa<ValueNode>()) {
        auto new_value_node = CreateNewValueNode(anf, graph);
        MS_EXCEPTION_IF_NULL(new_value_node);
        (void)cnode_inputs->emplace_back(new_value_node);
        continue;
      }
      MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
    }
  }
}

ValueNodePtr KernelGraphMgr::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto sub_func_graph = common::AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph.get()) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph.get()];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr KernelGraphMgr::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (param_value != nullptr) {
    new_parameter = param_value->parameter();
    if (new_parameter == nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
      new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
      param_value->set_parameter(new_parameter);
    }
  } else {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  }

  new_parameter->IncreaseUsedGraphCount();

  return new_parameter;
}

void KernelGraphMgr::FlattenTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall)) {
    auto call_graph = node->input(kFirstIndex);
    auto sub_kernel_graph = AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
    MS_EXCEPTION_IF_NULL(sub_kernel_graph);
    auto iter = kernel_graph_partial_map_.find(sub_kernel_graph.get());
    if (iter != kernel_graph_partial_map_.end() && iter->second.multi_tuple != 0) {
      (void)need_flatten_.insert(node);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    auto input = node->input(kFirstIndex);
    auto get_idx = common::AnfAlgo::GetTupleGetItemOutIndex(node);
    if (need_flatten_.find(input) != need_flatten_.end() && get_idx == 0) {
      need_flatten_tuple_map_[node] = input;
    }
  }
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(node); i++) {
    auto input = common::AnfAlgo::GetInputNode(node, i);
    auto iter = need_flatten_tuple_map_.find(input);
    if (iter != need_flatten_tuple_map_.end()) {
      node->set_input(i + 1, iter->second);
    }
  }
}

bool KernelGraphMgr::CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // create a new cnode object
  auto new_cnode = CreateNewCNode(cnode, graph);
  if (new_cnode == nullptr) {
    return false;
  }
  new_cnode->set_abstract(cnode->abstract());
  std::string fullname = cnode->fullname_with_scope();
  auto prim_input = cnode->input(kAnfPrimitiveIndex);
  // cnode is a call (partial/switch/switch_layer), full scope name is "1_2".
  // it is hard to analysis bug when it used as ge node name.
  if (!prim_input->isa<CNode>()) {
    new_cnode->set_fullname_with_scope(fullname);
  }
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendMapAdd(node, new_cnode);
  SetReturnNode(new_cnode, graph);
  FlattenTuple(new_cnode);
  return true;
}

void KernelGraphMgr::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto backend_parameter = graph->GetBackendAnfByFrontAnf(parameter);
    if (backend_parameter == nullptr) {
      // for example "def f(x,y,z) {return x + y}", parameter z in unused
      auto new_parameter = CreateNewParameter(parameter, graph);
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendMapAdd(parameter, new_parameter);
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    graph_inputs->push_back(backend_parameter);
  }
}

// 1. Convert the node to make_tuple if the node is a ValueNode<ValueTuple> and it's the input of 'return' node.
// 2. Set the return of graph if node is "Return" node.
// 3. If the return of graph has a Partial Func, should inline it in return value.
void KernelGraphMgr::SetReturnNode(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
    return;
  }
  constexpr auto kReturnInputIdx = 1;
  auto return_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(return_node);
  graph->set_return(return_node);
  auto graph_output = return_node->input(kReturnInputIdx);
  MS_EXCEPTION_IF_NULL(graph_output);

  // If return's input is value node, then the graph has no kernel, and the pass 'trans tuple to make_tuple' cannot
  // match this pattern because that pass begin with output node but return node. So we add transform value tuple
  // to make_tuple here.
  if (common::AnfAlgo::IsTupleOutput(graph_output) && graph_output->isa<ValueNode>()) {
    return_node->set_input(kReturnInputIdx, graph->TransTupleToMakeTuple(graph_output));
  }

  // inline partial to call graph
  auto return_tuple = return_node->input(kReturnInputIdx);
  MS_EXCEPTION_IF_NULL(return_tuple);
  if (return_tuple->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(return_tuple, prim::kPrimMakeTuple)) {
    auto make_tuple = return_tuple->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(make_tuple);
    // only support the last return node is a partial func now
    auto last_input_node = common::AnfAlgo::GetInputNode(make_tuple, tuple_input_num - 1);
    MS_EXCEPTION_IF_NULL(last_input_node);
    if (last_input_node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(last_input_node, prim::kPrimPartial)) {
      size_t multi_tuple = 0;
      auto partial_node = last_input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      size_t partial_input_num = common::AnfAlgo::GetInputTensorNum(partial_node);
      std::vector<AnfNodePtr> make_tuple_inputs;
      (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      // skip last return node (is a partial)
      size_t param_begin = 0;
      for (size_t i = 0; i < tuple_input_num - 1; i++) {
        auto input = common::AnfAlgo::GetInputNode(make_tuple, i);
        auto node_abs = input->abstract();
        MS_EXCEPTION_IF_NULL(node_abs);
        if (node_abs->isa<abstract::AbstractSequence>()) {
          MS_EXCEPTION_IF_CHECK_FAIL(
            i == 0, "Input index: " + std::to_string(i) + " is a make tuple, input node: " + input->DebugString());
          MS_LOG(DEBUG) << "Flatten the make tuple, input node: " << input->DebugString()
                        << ", output num: " << AnfUtils::GetOutputTensorNum(input);
          // flatten the make tuple
          for (size_t j = 0; j < AnfUtils::GetOutputTensorNum(input); j++) {
            auto idx = NewValueNode(SizeToLong(j));
            MS_EXCEPTION_IF_NULL(idx);
            auto imm = std::make_shared<Int64Imm>(j);
            idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
            auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, idx});
            std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(input, j)};
            auto shapes = {common::AnfAlgo::GetOutputInferShape(input, j)};
            common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
            param_begin++;
            multi_tuple++;
            (void)make_tuple_inputs.emplace_back(getitem);
          }
        } else {
          param_begin++;
          (void)make_tuple_inputs.emplace_back(input);
        }
      }
      // skip partial graph
      for (size_t i = kFirstIndex; i < partial_input_num; i++) {
        (void)make_tuple_inputs.emplace_back(common::AnfAlgo::GetInputNode(partial_node, i));
      }
      auto g_output = graph->NewCNode(make_tuple_inputs);
      MS_EXCEPTION_IF_NULL(g_output);
      std::vector<AbstractBasePtr> abstract_list;
      for (size_t i = kFirstIndex; i < make_tuple_inputs.size(); ++i) {
        auto inputs_node = make_tuple_inputs[i];
        MS_EXCEPTION_IF_NULL(inputs_node);
        (void)abstract_list.emplace_back(inputs_node->abstract());
      }
      auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
      MS_EXCEPTION_IF_NULL(g_output);
      g_output->set_abstract(abstract);
      graph->set_output(g_output);
      kernel_graph_partial_map_[graph] = {abstract, common::AnfAlgo::GetInputNode(partial_node, 0), param_begin,
                                          common::AnfAlgo::GetInputTensorNum(g_output), multi_tuple};
    }
  }
}

KernelGraphPtr KernelGraphMgr::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs,
                                                    DeviceType device_target, bool common_opt,
                                                    bool is_enable_zero_copy) {
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  std::vector<std::weak_ptr<KernelGraph>> child_graph_order;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  // Set the zero copy flag in subgraph sink mode.
  if (is_enable_zero_copy) {
    MS_LOG(INFO) << "Set zero copy flag for graph:" << graph->ToString();
    graph->set_flag(kFlagEnableZeroCopyInGraph, true);
  }
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  graph->set_device_target(device_target);
  for (const auto &node : lst) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // create a new cnode object
    auto new_cnode = CreateNewCNode(cnode, graph.get(), &other_graph_cnode);
    MS_EXCEPTION_IF_NULL(new_cnode);
    if (IsOneOfPrimitiveCNode(new_cnode, {prim::kPrimCall, prim::kPrimPartial})) {
      auto fn = new_cnode->input(kIndexOne);
      MS_EXCEPTION_IF_NULL(fn);
      auto child_kernel_graph = AnfRuntimeAlgorithm::GetValueNodeKernelGraph(fn);
      MS_EXCEPTION_IF_NULL(child_kernel_graph);
      child_graph_order.push_back(std::weak_ptr<KernelGraph>(child_kernel_graph));
    }

    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    new_cnode->set_attrs(cnode->attrs());
    if (cnode->user_data<pynative::JitCallGraph>()) {
      new_cnode->set_user_data(cnode->user_data<pynative::JitCallGraph>());
    }
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendMapAdd(node, new_cnode);
  }
  // add a make_tuple at the end of graph as output
  graph->set_child_graph_order(child_graph_order);
  graph->set_output(ConstructOutput(outputs, graph));
  FuncGraphManagerPtr manager = MakeManager({graph}, false);
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    UnifyMindIR(graph);
    graph->UpdateGraphAquireGilAttr();
    if (common_opt) {
      opt::BackendCommonOptimization(graph);
    }
    graph->SetInputNodes();
    SetInputNodeUsage(graph, manager);
    graph->SetOptimizerFlag();
  }
  graph->set_parameters(graph->inputs());
  return graph;
}

std::shared_ptr<KernelGraph> KernelGraphMgr::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                  std::vector<KernelGraphPtr> *all_out_graph,
                                                                  DeviceType device_target) {
  auto graph = NewKernelGraph();
  front_backend_graph_map_[func_graph.get()] = graph;
  ConstructKernelGraphInner(func_graph, all_out_graph, device_target, graph);
  return graph;
}

std::shared_ptr<KernelGraph> KernelGraphMgr::ConstructPackKernelGraph(const FuncGraphPtr &func_graph,
                                                                      std::vector<KernelGraphPtr> *all_out_graph,
                                                                      DeviceType device_target) {
  auto graph = NewPynativeKernelGraph();
  ConstructKernelGraphInner(func_graph, all_out_graph, device_target, graph);
  return graph;
}

void KernelGraphMgr::ConstructKernelGraphInner(const FuncGraphPtr &func_graph,
                                               std::vector<KernelGraphPtr> *all_out_graph, DeviceType device_target,
                                               const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  MS_EXCEPTION_IF_NULL(graph);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE) && context->CellReuseLevel() == CellReuseLevel::kLazyInline) {
    MS_LOG(INFO) << "Need backend inline: " << graph->graph_id();
    graph->set_need_inline(true);
  }
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  graph->set_device_target(device_target);
  // Create parameter
  for (const auto &node : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new node, node = " << node->DebugString();
    auto graph_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    auto new_parameter = CreateNewParameter(node, graph.get());
    graph_inputs->push_back(new_parameter);
    graph->FrontBackendMapAdd(node, new_parameter);
  }

  std::vector<ParameterPtr> added_parameters;
  std::vector<std::weak_ptr<KernelGraph>> child_kernel_graphs;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      continue;
    }
    MS_LOG(DEBUG) << "Start create new node, node = " << node->DebugString();
    // Create value node
    if (node->isa<ValueNode>()) {
      if (NeedConvertValueNodeToParameter(node)) {
        ConvertValueNodeToParameter(graph, node, &added_parameters);
        continue;
      }
      // Create common value node
      if (!IsValueNode<FuncGraph>(node)) {
        (void)CreateNewValueNode(node, graph.get());
        continue;
      }
      // Create child kernel graph according ValueNode<FuncGraph>
      FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(node);
      auto child_kernel_graph = front_backend_graph_map_.find(child_graph.get()) == front_backend_graph_map_.end()
                                  ? ConstructKernelGraph(child_graph, all_out_graph, device_target)
                                  : front_backend_graph_map_[child_graph.get()];
      (void)child_kernel_graphs.emplace_back(std::weak_ptr<KernelGraph>(child_kernel_graph));
      (void)CreateValueNodeKernelGraph(node, graph.get());
      continue;
    }
    // Create cnode
    if (!CreateCNodeOfKernelGraph(node, graph.get())) {
#ifdef ENABLE_DUMP_IR
      DumpIR("construct_kernel_graph_fail.ir", func_graph);
#endif
      MS_LOG(EXCEPTION) << "Construct func graph " << func_graph->ToString() << " failed."
                        << trace::DumpSourceLines(node);
    }
  }

  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  // Add ValueNode-Parameter to graph.
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  for (auto &parameter : added_parameters) {
    (void)graph_inputs->emplace_back(parameter);
  }

  FuncGraphManagerPtr manager = MakeManager({graph});
  graph->SetInputNodes();
  SetInputNodeUsage(graph, manager);
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif

  all_out_graph->push_back(graph);
  graph->set_parameters(graph->inputs());
  graph->set_child_graph_order(child_kernel_graphs);
}

void HandleGraphInputsOutputs(const nlohmann::json &graph_json, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &context = CompileCacheContext::GetInstance();
  if (graph_json.contains(kInputs)) {
    const auto &inputs_json = graph_json[kInputs];
    auto mutable_inputs = graph->MutableInputs();
    for (const auto &name : inputs_json) {
      AnfNodePtr node = context.FindBackNodeByBackName(name);
      MS_EXCEPTION_IF_NULL(node);
      context.InsertBackNameToBackNode(name, node);
      mutable_inputs->push_back(node);
    }
  }
  if (graph_json.contains(kParameters)) {
    const auto &parameters_json = graph_json[kParameters];
    std::vector<AnfNodePtr> parameters;
    for (const auto &name : parameters_json) {
      auto node = context.FindBackNodeByBackName(name);
      MS_EXCEPTION_IF_NULL(node);
      parameters.push_back(node);
    }
    graph->set_parameters(parameters);
  }

  if (graph_json.contains(kValidInputs)) {
    const auto &valid_inputs_json = graph_json[kValidInputs];
    auto mutable_valid_inputs = graph->MutableValidInputs();
    std::vector<bool> valid_inputs;
    (void)(std::transform(valid_inputs_json.begin(), valid_inputs_json.end(), std::back_inserter(valid_inputs),
                          [](const auto &val) { return val; }));
    *mutable_valid_inputs = valid_inputs;
  }
  const auto &front_graph_name = graph_json[kCorrespondFrontendGraph];
  const auto &front_graph_node = context.FindFrontNodeByFrontName(front_graph_name);
  FuncGraphPtr front_graph = GetValueNode<FuncGraphPtr>(front_graph_node);
  MS_EXCEPTION_IF_NULL(front_graph);
  graph->FrontBackendMapAdd(front_graph->get_return(), graph->get_return());
}

void HandleGraphSimpleAttr(const nlohmann::json &graph_json, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Handle graph " << graph->ToString() << " simple attr.";
  auto &context = CompileCacheContext::GetInstance();
  graph->set_run_mode(graph_json[kRunMode]);
  graph->set_is_loop_count_sink(graph_json[kIsLoopCountSink]);
  graph->SetGraphDynamicAttr(graph_json[kIsDynamicShape]);
  graph->set_device_target(graph_json[kDeviceTarget]);
  graph->set_root_graph_id(graph_json[kRootGraphId]);
  graph->set_executable(graph_json[kExecutable]);
  graph->set_recursive_call(graph_json[kHasRecursiveCall]);
  graph->set_need_inline(graph_json[kNeedInline]);
  graph->set_is_need_gil(graph_json[kIsNeedGil]);
  graph->set_is_from_single_op(graph_json[kIsFromSingleOp]);
  graph->set_subgraph_multi_call(graph_json[kHasSubgraphMultiCall]);
  graph->set_label_num(graph_json[kLabelNum]);
#ifndef ENABLE_SECURITY
  // set summary_node of graph
  graph->set_summary_node_exist(graph_json[kSummaryNodeExist]);
#endif
  if (graph_json.contains(kStartLabel)) {
    auto start_label = context.FindBackNodeByBackName(graph_json[kStartLabel]);
    if (start_label) {
      auto cstart_label = start_label->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cstart_label);
      graph->set_start_label(cstart_label);
    }
  }
  if (graph_json.contains(kEndGoto)) {
    auto end_goto = context.FindBackNodeByBackName(graph_json[kEndGoto]);
    if (end_goto) {
      auto cend_goto = end_goto->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cend_goto);
      graph->set_end_goto(cend_goto);
    }
  }
  if (graph_json.contains(kParameterUniqueNameToName)) {
    const auto &unique_name_to_name_json = graph_json[kParameterUniqueNameToName];
    for (const auto &[unique_name, name] : unique_name_to_name_json.items()) {
      auto node = context.FindBackNodeByBackName(unique_name);
      MS_EXCEPTION_IF_NULL(node);
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      param->set_name(name);
    }
  }
  MS_LOG(INFO) << "Handle graph " << graph->ToString() << " simple attr success.";
}

void HandleAttrAboutOtherGraph(const mindspore::HashMap<GraphId, std::shared_ptr<KernelGraph>> &graphs,
                               const nlohmann::json &graph_json, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &context = CompileCacheContext::GetInstance();
  if (graph_json.contains(kPreGraphs)) {
    const auto &pre_graphs_json = graph_json[kPreGraphs];
    for (const auto &iter : pre_graphs_json) {
      auto pre_graph = graphs.at(iter);
      MS_EXCEPTION_IF_NULL(pre_graph);
      graph->AddPreGraph(pre_graph);
    }
  }
  if (graph_json.contains(kPostGraphs)) {
    const auto &post_graphs_json = graph_json[kPostGraphs];
    for (const auto &iter : post_graphs_json) {
      auto post_graph = graphs.at(iter);
      MS_EXCEPTION_IF_NULL(post_graph);
      graph->AddPostGraph(post_graph);
    }
  }
  if (graph_json.contains(kChildGraphResult)) {
    const auto &child_graph_result_json = graph_json[kChildGraphResult];
    for (const auto &iter : child_graph_result_json) {
      auto node = context.FindBackNodeByBackName(iter);
      MS_EXCEPTION_IF_NULL(node);
      graph->AddChildGraphResult(node);
    }
  }
  if (graph_json.contains(kChildGraphOrder)) {
    const auto &child_graph_order_json = graph_json[kChildGraphOrder];
    std::vector<std::weak_ptr<KernelGraph>> child_graph_order;
    for (const auto &iter : child_graph_order_json) {
      auto child_graph = graphs.at(iter);
      MS_EXCEPTION_IF_NULL(child_graph);
      child_graph_order.push_back(std::weak_ptr<KernelGraph>(child_graph));
    }
    graph->set_child_graph_order(child_graph_order);
  }
}

void HandleGraphComplexAttr(const mindspore::HashMap<GraphId, std::shared_ptr<KernelGraph>> &graphs,
                            const nlohmann::json &graph_json, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Handle graph " << graph->ToString() << " complex attr.";
  auto &context = CompileCacheContext::GetInstance();
  std::vector<CNodePtr> execution_order;
  const auto &execution_order_json = graph_json[kExecutionOrder];
  for (const auto &order : execution_order_json) {
    auto node = context.FindBackNodeByBackName(order);
    MS_EXCEPTION_IF_NULL(node);
    execution_order.push_back(node->cast<CNodePtr>());
  }
  graph->set_execution_order(execution_order);
  if (graph_json.contains(kCommSubGraphIds)) {
    const auto &comm_sub_grpah_ids_json = graph_json[kCommSubGraphIds];
    for (const auto &iter : comm_sub_grpah_ids_json) {
      graph->RecordNewCommSubGraphId(iter);
    }
  }
  HandleAttrAboutOtherGraph(graphs, graph_json, graph);
  if (graph_json.contains(kInternalParameterToFrontNode)) {
    const auto &internal_parameter_to_front_node_json = graph_json[kInternalParameterToFrontNode];
    HashMap<AnfNodePtr, AnfWithOutIndex> internal_parameter_to_front_node;
    for (const auto &iter : internal_parameter_to_front_node_json) {
      const auto &back_name = iter.at(0);
      const auto &front_name = iter.at(kIndexOne);
      const auto &index = iter.at(kIndexTwo);
      auto back_node = context.FindBackNodeByBackName(back_name);
      MS_EXCEPTION_IF_NULL(back_node);
      auto front_node = context.FindFrontNodeByFrontName(front_name);
      MS_EXCEPTION_IF_NULL(front_node);
      internal_parameter_to_front_node[back_node] = AnfWithOutIndex(front_node, index);
    }
    graph->SetInternalParameterToFrontNodeMap(internal_parameter_to_front_node);
  }
  if (graph_json.contains(kRefInOutMap)) {
    const auto &ref_in_out_map_json = graph_json[kRefInOutMap];
    for (const auto &iter : ref_in_out_map_json) {
      const auto &first_name = iter.at(0);
      const auto &first_index = iter.at(kIndexOne);
      const auto &second_name = iter.at(kIndexTwo);
      const auto &second_index = iter.at(kIndexThree);
      auto first_node = context.FindBackNodeByBackName(first_name);
      MS_EXCEPTION_IF_NULL(first_node);
      auto second_node = context.FindBackNodeByBackName(second_name);
      MS_EXCEPTION_IF_NULL(second_node);
      graph->AddRefCorrespondPairs(AnfWithOutIndex(first_node, first_index),
                                   AnfWithOutIndex(second_node, second_index));
    }
  }
  if (graph_json.contains(kNodesKernelInfo)) {
    const auto &kernel_infos_json = graph_json[kNodesKernelInfo];
    LoadAnfKernelInfoFromJson(kernel_infos_json);
  }
  if (graph_json.contains(kGraphValueNodes)) {
    const auto &graph_value_nodes_json = graph_json[kGraphValueNodes];
    for (const auto &iter : graph_value_nodes_json) {
      auto node = context.FindBackNodeByBackName(iter);
      MS_EXCEPTION_IF_NULL(node);
      auto value_node = node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      graph->AddValueNodeToGraph(value_node);
    }
  }
#ifndef ENABLE_SECURITY
  if (graph_json.contains(kSummaryNodes)) {
    const auto &summary_nodes_json = graph_json[kSummaryNodes];
    std::map<std::string, std::pair<AnfNodePtr, int>> summary_nodes;
    for (const auto &iter : summary_nodes_json) {
      const auto &first = iter.at(0);
      const auto &name = iter.at(kIndexOne);
      auto node = context.FindBackNodeByBackName(name);
      MS_EXCEPTION_IF_NULL(node);
      const auto &index = iter.at(kIndexTwo);
      summary_nodes[first] = std::make_pair(node, index);
    }
    graph->set_summary_nodes(summary_nodes);
  }
#endif
  MS_LOG(INFO) << "Handle graph " << graph->ToString() << " complex attr success.";
}

bool KernelGraphMgr::ParseKernelGraphNodesAndAttrs(const nlohmann::json &model_json) {
  auto &context = CompileCacheContext::GetInstance();
  for (auto &[graph_name, graph_json] : model_json.items()) {
    MS_LOG(DEBUG) << "Parse graph " << graph_name << " nodes and attrs.";
    KernelGraphPtr graph = graphs_.at(graph_json[kGraphId]);
    MS_EXCEPTION_IF_NULL(graph);
    HandleGraphInputsOutputs(graph_json, graph.get());
    const auto &back_to_front = graph_json[kBackendFrontAnf];
    for (const auto &[back_node_name, front_node_name] : back_to_front.items()) {
      auto back_node = context.FindBackNodeByBackName(back_node_name);
      if (!back_node) {
        MS_LOG(EXCEPTION) << "The backend node is nullptr, its unique name is " << back_node_name;
      }
      auto front_node = context.FindFrontNodeByFrontName(front_node_name);
      if (!front_node) {
        MS_LOG(EXCEPTION) << "The frontend node is nullptr, its unique name is " << front_node_name;
      }
      if (graph->FrontendNodeExistInFrontBackendMap(front_node)) {
        if (graph->BackendNodeExistInFrontBackendMap(back_node)) {
          continue;
        }
        auto old_back_node = graph->GetBackendAnfByFrontAnf(front_node);
        graph->FrontBackendMapAdd(old_back_node, back_node);
      } else {
        graph->FrontBackendMapAdd(front_node, back_node);
      }
    }
    HandleGraphSimpleAttr(graph_json, graph.get());
    HandleGraphComplexAttr(graphs_, graph_json, graph.get());

    FuncGraphManagerPtr manager = MakeManager({graph});
    if (manager) {
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
    }
    graph->SetInputNodes();
    SetInputNodeUsage(graph, manager);
    graph->SetOptimizerFlag();
  }
  return true;
}

void ResetGetNextSharedName(const FuncGraphPtr &graph) {
  auto &config_mgr = ConfigManager::GetInstance();
  auto queue_name = config_mgr.QueueName();
  auto cnodes = graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    auto prim = GetValuePtr<Primitive>(cnode->input(0));
    if (prim != nullptr && prim->HasAttr("shared_name")) {
      prim->set_attr("shared_name", MakeValue(queue_name));
      break;
    }
  }
}

std::shared_ptr<KernelGraph> KernelGraphMgr::ConstructKernelGraph(std::vector<KernelGraphPtr> *all_out_graph) {
  MS_LOG(WARNING) << "Use the compile cache to construct kernel graph, Be aware of correctness risks.";
  auto &context = CompileCacheContext::GetInstance();
  auto frontend_graph = context.FrontGraph();
  if (!frontend_graph) {
    MS_LOG(EXCEPTION) << "The frontend graph is null";
  }
  auto cache_path = context.GetBackendGraphCachePath(frontend_graph);
  std::string json_path = cache_path + kJsonSuffix;
  nlohmann::json model_json;
  auto load_json_success = LoadJson(json_path, &model_json);
  if (!load_json_success) {
    MS_LOG(EXCEPTION) << "Load json file " << json_path << " failed.";
  }
  // construct kernel graph and its params that exist correspond frontend param
  mindspore::HashMap<std::string, AnfNodePtr> name_to_node;
  (void)std::for_each(name_to_params_.begin(), name_to_params_.end(), [&name_to_node](const auto &ele) {
    if (!ele.second.expired()) {
      name_to_node[ele.first] = ele.second.lock();
    }
  });
  MS_LOG(DEBUG) << "Construct kernel graph and its params that exist correspond frontend param.";
  for (size_t i = 0; i < model_json.size(); i++) {
    auto kernel_graph = NewKernelGraph();
    all_out_graph->push_back(kernel_graph);
    const auto &graph_name = kernel_graph->ToString();
    if (!model_json.contains(graph_name)) {
      MS_LOG(EXCEPTION) << "Load graph " << graph_name << " from json failed.";
    }
    auto &graph_json = model_json[graph_name];
    if (!graph_json.contains(kCorrespondFrontendGraph)) {
      continue;
    }
    const auto &front_graph_name = graph_json[kCorrespondFrontendGraph];
    const auto &front_graph_node = context.FindFrontNodeByFrontName(front_graph_name);
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(front_graph_node);
    MS_EXCEPTION_IF_NULL(fg);
    front_backend_graph_map_[fg.get()] = kernel_graph;
    if (graph_json.contains(kBackendParamToFrontendParamIndex)) {
      const auto &backend_param_to_frontend_param_index = graph_json[kBackendParamToFrontendParamIndex];
      const auto &frontend_graph_params = fg->parameters();
      for (const auto &[param_unique_name, index] : backend_param_to_frontend_param_index.items()) {
        const auto front_param = frontend_graph_params.at(index);
        MS_EXCEPTION_IF_NULL(front_param);
        MS_LOG(DEBUG) << "Start create new node, old node = " << front_param->DebugString()
                      << ", new node = " << param_unique_name;
        auto new_parameter = CreateNewParameter(front_param, kernel_graph.get());
        kernel_graph->FrontBackendMapAdd(front_param, new_parameter);
        name_to_node[param_unique_name] = new_parameter;
      }
    }
  }

  std::vector<FuncGraphPtr> graphs_for_load;
  (void)(std::transform(all_out_graph->begin(), all_out_graph->end(), std::back_inserter(graphs_for_load),
                        [](const KernelGraphPtr &g) { return g; }));

  MindIRLoader mindir_loader;
  std::string mindir_path = cache_path + kMindIrSuffix;
  auto real_path = Common::CreatePrefixPath(mindir_path, true);
  if (!CheckPath(real_path)) {
    MS_LOG(EXCEPTION) << "The mindir path is " << mindir_path << ", and it is a invalid path!";
  }
  if (!mindir_loader.LoadMindIR(real_path.value(), graphs_for_load, &name_to_node)) {
    MS_LOG(EXCEPTION) << "Load mindir from " << real_path.value() << " failed.";
  }
  (void)std::for_each(name_to_node.begin(), name_to_node.end(), [](const auto &ele) {
    auto node = ele.second;
    MS_EXCEPTION_IF_NULL(node);
    if (node->template isa<Parameter>()) {
      name_to_params_[ele.first] = std::weak_ptr<AnfNode>(node);
    }
  });
  context.SetBackNameToBackNode(name_to_node);
  // the value of attr "shared_name" will changed every time, so reset GetNext shared_name
  ResetGetNextSharedName(all_out_graph->front());
  if (!ParseKernelGraphNodesAndAttrs(model_json)) {
    MS_LOG(EXCEPTION) << "Parse kernel graph nodes and attrs failed.";
  }

#ifdef ENABLE_DUMP_IR
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->CanDump(kIntroductory)) {
    for (const auto &iter : graphs_) {
      auto dump_name = std::string("loaded_") + iter.second->ToString() + ".ir";
      DumpIR(dump_name, iter.second);
    }
  }
#endif
  MS_LOG(WARNING)
    << "Use the compile cache to construct kernel graph success, and will execute the preprocess before run directly.";
  return all_out_graph->front();
}

void KernelGraphMgr::SetInputNodeUsage(const KernelGraphPtr &graph, const FuncGraphManagerPtr &manager) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(manager);
  auto input_nodes = graph->input_nodes();
  for (auto &input_node : input_nodes) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>()) {
      auto node_ptr = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(node_ptr);
      if (!IsUsedByRealKernel(manager, input_node, graph->graph_id())) {
        node_ptr->SetNotUsedByRealKernelInGraph(graph->graph_id());
      }
      auto shape = node_ptr->Shape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->isa<abstract::Shape>() && shape->IsDynamic()) {
        node_ptr->set_has_dynamic_shape(true);
      }
      if (input_node->abstract() != nullptr && input_node->abstract()->isa<abstract::AbstractSequence>()) {
        // If the parameter is dynamic sequence, it is regard as dynamic shape.
        const auto &tuple_abs = input_node->abstract()->cast<abstract::AbstractSequencePtr>();
        MS_EXCEPTION_IF_NULL(tuple_abs);
        if (tuple_abs->dynamic_len()) {
          MS_LOG(INFO) << "Input node:" << input_node->DebugString() << " set dynamic flag to true";
          node_ptr->set_has_dynamic_shape(true);
        }
      }
    }
  }
}

namespace {
bool CNodeFirstInputIsPrimitive(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto prim = cnode->input(kAnfPrimitiveIndex);
  if (prim == nullptr || !IsValueNode<Primitive>(prim)) {
    return false;
  }
  return true;
}

std::vector<AnfNodePtr> ExtendNodeUsers(const FuncGraphManagerPtr &front_func_graph_manager,
                                        const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_func_graph_manager);
  auto &users = front_func_graph_manager->node_users()[front_node];
  std::vector<AnfNodePtr> result;
  for (auto &user : users) {
    if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimDepend) ||
        common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimLoad)) {
      auto depend_cnode = user.first->cast<CNodePtr>();
      if (depend_cnode == nullptr) {
        continue;
      }
      if (front_node != depend_cnode->input(1)) {
        continue;
      }
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      (void)result.insert(result.cend(), res.cbegin(), res.cend());
    } else if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimMakeTuple)) {
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      (void)result.insert(result.cend(), res.cbegin(), res.cend());
    } else {
      (void)result.emplace_back(user.first);
    }
  }
  return result;
}

AnfNodePtr GetSupportedInternalNode(const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    return nullptr;
  }
  if (AnfUtils::IsRealKernel(front_node)) {
    return front_node;
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimTupleGetItem)) {
    return front_node;
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimMakeTuple)) {
    auto cnode = front_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.size() > 1) {
      return GetSupportedInternalNode(inputs[1]);
    }
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimDepend)) {
    auto cnode = front_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.size() >= kDependInputSize) {
      return GetSupportedInternalNode(inputs[kRealInputIndexInDepend]);
    }
  }
  return nullptr;
}

bool IsUnusedInternlOutput(const AnfNodePtr &user) {
  if (!CNodeFirstInputIsPrimitive(user)) {
    return true;
  }
  if (IsPrimitiveCNode(user, prim::kPrimSwitch) || IsPrimitiveCNode(user, prim::kPrimSwitchLayer)) {
    return true;
  }
  if (!AnfUtils::IsRealKernel(user)) {
    return true;
  }
  return false;
}
}  // namespace

constexpr auto kMixTarget = "MixTarget";
constexpr auto kNoTarget = "NoTarget";
std::string KernelGraphMgr::AddPartialParametersMap(const AnfNodePtr &partial_node) {
  MS_EXCEPTION_IF_NULL(partial_node);
  auto iter = partial_target_map_.find(partial_node);
  if (iter != partial_target_map_.end()) {
    return iter->second;
  }
  auto partial_cnode = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto partial_graph = GetValueNode<FuncGraphPtr>(partial_cnode->input(kFirstDataInputIndex));
  // If graph is nullptr, it means that the funcgraph in the partial node is a deadnode, and the processing is skipped.
  if (partial_graph == nullptr) {
    return kNoTarget;
  }
  auto parameters = partial_graph->parameters();
  auto partial_inputs = partial_cnode->inputs();
  const size_t kNonParameterNum = 2;
  if (parameters.size() + kNonParameterNum != partial_inputs.size()) {
    return kMixTarget;
  }
  for (size_t i = 0; i < parameters.size(); ++i) {
    partial_parameters_map_[parameters[i]] = partial_inputs[kNonParameterNum + i];
  }
  auto graph_nodes = TopoSort(partial_graph->get_return());
  std::string graph_target = kNoTarget;
  for (auto &node : graph_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!AnfUtils::IsRealKernel(node)) {
      continue;
    }
    std::string cur_target = GetCNodeTarget(node);
    if (graph_target == kNoTarget) {
      graph_target = cur_target;
    }
    if (graph_target != cur_target) {
      graph_target = kMixTarget;
      break;
    }
  }
  (void)partial_target_map_.emplace(std::pair<AnfNodePtr, std::string>(partial_node, graph_target));
  return graph_target;
}

namespace {
bool IsNeedAddPartialParameter(const AnfNodePtr &user, const std::string &kernel_target,
                               const std::shared_ptr<KernelGraph> &graph) {
  // If the flag is enable, it means the graph would run in subgraph sink mode, the real parameter on partial
  // cannot share the same device address with the formal parameter.
  MS_EXCEPTION_IF_NULL(graph);
  return common::AnfAlgo::CheckPrimitiveType(user, prim::kPrimPartial) && kernel_target != kGPUDevice &&
         !ExistGraphCaller(user) && (!graph->has_flag(kFlagEnableZeroCopyInGraph));
}
}  // namespace

void KernelGraphMgr::HandleInternalOutput(const AnfNodePtr &input_front_node, const AnfNodePtr &backend_node,
                                          const FuncGraphManagerPtr &front_func_graph_manager,
                                          const std::shared_ptr<KernelGraph> &backend_graph) {
  MS_EXCEPTION_IF_NULL(backend_graph);
  auto front_node = GetSupportedInternalNode(input_front_node);
  if (front_node == nullptr) {
    return;
  }
  auto front_real_kernel_pair = common::AnfAlgo::VisitKernel(front_node, 0);
  auto backend_real_kernel_pair = common::AnfAlgo::VisitKernel(backend_node, 0);
  auto backend_real_kernel = backend_real_kernel_pair.first;
  if (backend_real_kernel == nullptr || !backend_real_kernel->isa<CNode>()) {
    return;
  }
  auto front_real_kernel = front_real_kernel_pair.first;
  std::string kernel_target = GetCNodeTarget(front_real_kernel);
  bool internal_output = CNodeFirstInputIsPrimitive(front_real_kernel);
  bool unique_target = true;
  if (internal_output && common::AnfAlgo::IsNopNode(front_real_kernel)) {
    auto pre_node_pair = common::AnfAlgo::GetPrevNodeOutput(front_real_kernel, 0);
    auto pre_node_target = GetCNodeTarget(pre_node_pair.first);
    if (pre_node_target != kernel_target) {
      unique_target = false;
    }
  }
  if (internal_output) {
    auto users = ExtendNodeUsers(front_func_graph_manager, front_node);
    for (auto &user : users) {
      if (IsNeedAddPartialParameter(user, kernel_target, backend_graph)) {
        auto partial_target = AddPartialParametersMap(user);
        if (partial_target != kNoTarget && partial_target != kernel_target) {
          unique_target = false;
        }
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(user, prim::kPrimUpdateState)) {
        continue;
      }
      if (IsUnusedInternlOutput(user)) {
        internal_output = false;
        break;
      }
      if (kernel_target != GetCNodeTarget(user)) {
        unique_target = false;
      }
    }
  }
  if (internal_output) {
    MS_LOG(INFO) << "AddInternalOutput: " << front_node->DebugString() << " To " << backend_real_kernel->DebugString()
                 << ", unique_target: " << unique_target;
    backend_graph->AddInternalOutput(front_node, backend_real_kernel, backend_real_kernel_pair.second, unique_target);
  }
}

CNodePtr KernelGraphMgr::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "Output:" << output->DebugString();
  }
  auto FindEqu = [graph, outputs, this](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
        return backend_anf;
      }

      MS_EXCEPTION_IF_NULL(out);
      auto out_func_graph = out->func_graph();
      MS_EXCEPTION_IF_NULL(out_func_graph);
      auto out_func_graph_manager = out_func_graph->manager();
      if (out_func_graph_manager == nullptr) {
        return backend_anf;
      }
      HandleInternalOutput(out, backend_anf, out_func_graph_manager, graph);
      return backend_anf;
    }
    MS_LOG(EXCEPTION) << "Can't find the node in the equiv map!";
  };
  output_args.push_back(mindspore::NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())));
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_args),
                       [&](const AnfNodePtr &out) -> AnfNodePtr { return FindEqu(out); });
  auto output_node = graph->NewCNode(output_args);
  MS_EXCEPTION_IF_NULL(output_node);
  // Create abstract for output maketuple node.
  AbstractBasePtrList output_abs_list;
  const auto &inputs = output_node->inputs();
  (void)std::transform(
    inputs.begin() + 1, inputs.end(), std::back_inserter(output_abs_list), [](const AnfNodePtr &input) {
      return input->abstract() == nullptr ? std::make_shared<abstract::AbstractNone>() : input->abstract();
    });
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(output_abs_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  output_node->set_abstract(abstract_tuple);
  return output_node;
}

KernelGraphPtr KernelGraphMgr::NewPynativeKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_is_from_pynative(true);
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id(pynative_graph_sum_);
  pynative_graph_sum_++;
  return graph;
}

KernelGraphPtr KernelGraphMgr::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  SetKernelGraphId(graph);
  return graph;
}

void KernelGraphMgr::SetKernelGraphId(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = kernel_graph;
}

void KernelGraphMgr::UnifyMindIR(const KernelGraphPtr &graph) { opt::CommonUnifyMindIR(graph); }

namespace {
void CopyCNodeInfo(const FuncGraphPtr &func_graph, const uint32_t &target_graph_id, const AnfNodePtr &ori_node,
                   const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(new_node);
  MS_EXCEPTION_IF_NULL(ori_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(new_node->kernel_info());
  // deep copy kernel info
  if (kernel_info != nullptr && kernel_info->has_build_info()) {
    // some check
    MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->MutableKernelMod() == nullptr,
                               "Inline ERROR: " + ori_node->DebugString() + ", kernel mod is not nullptr");
    MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->output_address_list().empty(),
                               "Inline ERROR: " + ori_node->DebugString() + ", output_address_list is not empty");
    MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->workspace_address_list().empty(),
                               "Inline ERROR: " + ori_node->DebugString() + ", workspace_address_list is not empty");

    auto new_kernel_info = std::make_shared<device::KernelInfo>();
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(
      AnfRuntimeAlgorithm::GetSelectKernelBuildInfo(new_node));
    MS_EXCEPTION_IF_NULL(builder);
    MS_EXCEPTION_IF_NULL(new_kernel_info);
    new_kernel_info->set_select_kernel_build_info(builder->Build());
    new_kernel_info->set_graph_id(target_graph_id);
    new_kernel_info->set_feature_map_flag(kernel_info->is_feature_map());
    new_kernel_info->set_ref_map(false, kernel_info->out_in_ref_map());
    new_node->set_kernel_info(new_kernel_info);
  } else {
    auto new_kernel_info = std::make_shared<device::KernelInfo>();
    new_node->set_kernel_info(new_kernel_info);
  }
  if (ori_node->isa<CNode>()) {
    auto ori_cnode = ori_node->cast<CNodePtr>();
    if (common::AnfAlgo::HasNodeAttr(kAttrIsUBFusionOp, ori_cnode) &&
        common::AnfAlgo::GetNodeAttr<bool>(ori_node, kAttrIsUBFusionOp)) {
      // already done fusion compile
      auto ori_full_name = ori_cnode->fullname_with_scope();
      common::AnfAlgo::SetNodeAttr(kAttrOriFusionName, MakeValue(ori_full_name), new_node);
    }
    common::AnfAlgo::SetNodeAttr(kAttrNeedInline, MakeValue(ori_node->fullname_with_scope()), new_node);
    common::AnfAlgo::SetNodeAttr(kAttrPreKernelGraph, MakeValue(func_graph), new_node);
  }
}
}  // namespace

AnfNodePtr KernelGraphMgr::DoInline(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                                    const AnfNodePtrList &func_graph_args, const ScopePtr &scope,
                                    const uint32_t &target_graph_id,
                                    const std::map<session::AnfWithOutIndex, session::AnfWithOutIndex> &ref_map,
                                    const KernelGraphPtr &graph, bool is_switch_inline) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  KernelGraphPtr target_kernel_graph = nullptr;
  if (target_func_graph->isa<KernelGraph>()) {
    target_kernel_graph = target_func_graph->cast<KernelGraphPtr>();
  }
  Cloner cloner({}, false);
  if (scope != nullptr) {
    cloner.set_scope(scope);
  }
  cloner.AddClone(func_graph, target_func_graph, func_graph_args, kInline);
  auto node_list = TopoSort(func_graph->output());
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> condition_node_map;
  for (auto &ori_node : node_list) {
    MS_EXCEPTION_IF_NULL(ori_node);
    if (ori_node->isa<Parameter>()) {
      continue;
    }
    auto new_node = cloner[ori_node];
    MS_EXCEPTION_IF_NULL(new_node);
    if (new_node->isa<ValueNode>()) {
      auto value_node = new_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      graph->AddValueNodeToGraph(value_node);
    }
    // Add sub graph kernel for switch inline kernel graph.
    if (new_node->isa<CNode>() && target_kernel_graph != nullptr && is_switch_inline) {
      MS_LOG(DEBUG) << "Add inline sub graph for kernel:" << new_node->fullname_with_scope()
                    << " graph:" << func_graph->ToString();
      std::string sub_graph_name = func_graph->ToString();
      if (func_graph->isa<KernelGraph>()) {
        const auto &kernel_graph = func_graph->cast<KernelGraphPtr>();
        MS_EXCEPTION_IF_NULL(kernel_graph);
        const auto &sub_graph_iter = kernel_graph->inline_sub_graph_kernels().find(ori_node);
        if (sub_graph_iter != kernel_graph->inline_sub_graph_kernels().end()) {
          sub_graph_name = sub_graph_iter->second;
        }
      }
      target_kernel_graph->AddInlineSubgraphKernel(new_node, sub_graph_name);
      if (common::AnfAlgo::CheckPrimitiveType(new_node, prim::kPrimConditionGather) ||
          common::AnfAlgo::CheckPrimitiveType(new_node, prim::kPrimConditionSwitch)) {
        condition_node_map[ori_node] = new_node;
      }
    }
    CopyCNodeInfo(func_graph, target_graph_id, ori_node, new_node);
  }
  // Collect condition gather node and condition switch node.
  if (func_graph->isa<KernelGraph>() && is_switch_inline) {
    const auto &kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    const auto &gather_to_switch = kernel_graph->condition_gather_to_switch();
    for (const auto &pair : gather_to_switch) {
      MS_EXCEPTION_IF_NULL(pair.first);
      MS_EXCEPTION_IF_NULL(pair.second);
      const auto &gather_iter = condition_node_map.find(pair.first);
      const auto &switch_iter = condition_node_map.find(pair.second);
      if (gather_iter == condition_node_map.end() || switch_iter == condition_node_map.end()) {
        MS_LOG(EXCEPTION) << "Failed to get new gather node:" << pair.first->fullname_with_scope()
                          << " or switch node:" << pair.second->fullname_with_scope()
                          << " in graph:" << func_graph->ToString();
      }
      MS_EXCEPTION_IF_NULL(gather_iter->second);
      MS_EXCEPTION_IF_NULL(switch_iter->second);
      if (target_kernel_graph != nullptr) {
        target_kernel_graph->AddConditionGatherSwitchPair(gather_iter->second, switch_iter->second);
        MS_LOG(INFO) << "Add condition node pair:" << gather_iter->second->fullname_with_scope()
                     << " and:" << switch_iter->second->fullname_with_scope()
                     << " for graph:" << target_kernel_graph->ToString();
      }
    }
  }

  for (const auto &kv : ref_map) {
    auto final_pair = kv.first;
    auto origin_pair = kv.second;
    final_pair.first = cloner[final_pair.first];
    origin_pair.first = cloner[origin_pair.first];
    auto new_origin_pair = common::AnfAlgo::VisitKernel(origin_pair.first, origin_pair.second);
    graph->AddRefCorrespondPairs(final_pair, new_origin_pair);
  }
  return cloner[func_graph->output()];
}
}  // namespace session
}  // namespace mindspore
