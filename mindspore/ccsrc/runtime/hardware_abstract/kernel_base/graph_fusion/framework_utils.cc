/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/utils/anfalgo.h"
#include "utils/convert_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "ir/format_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/oplib.h"
#include "ir/graph_utils.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"
#include "mindapi/base/type_id.h"
#include "include/utils/common.h"
#include "include/utils/compile_cache_context.h"
#include "primitive/framework_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/sequence_ops.h"
#include "utils/file_utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
namespace {
bool IsPathSeparator(char c) { return c == '/' || c == '\\'; }

bool PathCharEqual(char lhs, char rhs) {
#if defined(_WIN32) || defined(_WIN64)
  if (IsPathSeparator(lhs) && IsPathSeparator(rhs)) {
    return true;
  }
  return std::tolower(static_cast<unsigned char>(lhs)) == std::tolower(static_cast<unsigned char>(rhs));
#else
  return lhs == rhs;
#endif
}

// Both paths must be canonical. Only direct children are accepted so that a symlink cannot redirect the load to a
// nested or external directory.
bool IsDirectChildPath(const std::string &directory, const std::string &file_path) {
  if (directory.empty() || file_path.size() <= directory.size() ||
      !std::equal(directory.begin(), directory.end(), file_path.begin(), PathCharEqual)) {
    return false;
  }
  size_t relative_pos = directory.size();
  if (!IsPathSeparator(directory.back())) {
    if (!IsPathSeparator(file_path[relative_pos])) {
      return false;
    }
    ++relative_pos;
  }
  if (relative_pos >= file_path.size()) {
    return false;
  }
  return std::none_of(file_path.begin() + relative_pos, file_path.end(), IsPathSeparator);
}
}  // namespace

std::string GetCompilerCachePath() { return CompileCacheContext::GetInstance().GetUserDefineCachePath(); }

std::optional<std::string> GetValidKernelFilePath(const std::string &kernel_meta_path, const std::string &file_name) {
  if (!Common::IsFilenameValid(file_name, MAX_OS_FILENAME_LENGTH, "Invalid AKG kernel file name.")) {
    return std::nullopt;
  }
  auto real_kernel_meta_path = FileUtils::GetRealPath(kernel_meta_path.c_str());
  if (!real_kernel_meta_path.has_value()) {
    MS_LOG(ERROR) << "Invalid AKG kernel meta path: " << kernel_meta_path;
    return std::nullopt;
  }
  std::string candidate_path = kernel_meta_path;
  if (!candidate_path.empty() && !IsPathSeparator(candidate_path.back())) {
    candidate_path += '/';
  }
  candidate_path += file_name;
  auto real_file_path = FileUtils::GetRealPath(candidate_path.c_str());
  if (!real_file_path.has_value()) {
    MS_LOG(ERROR) << "Invalid AKG kernel file path: " << candidate_path;
    return std::nullopt;
  }
  if (!IsDirectChildPath(real_kernel_meta_path.value(), real_file_path.value())) {
    MS_LOG(ERROR) << "AKG kernel file is outside the kernel meta directory: " << real_file_path.value();
    return std::nullopt;
  }
  struct stat file_stat {};
  if (stat(real_file_path->c_str(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
    MS_LOG(ERROR) << "AKG kernel file is not a regular file: " << real_file_path.value();
    return std::nullopt;
  }
  return real_file_path;
}

void KernelMeta::Initialize(const std::string &backend) {
  auto config_path = GetCompilerCachePath();
  kernel_meta_path_ = config_path + backend + std::string(kKernelMetaSuffix);
  (void)(FileUtils::CreateNotExistDirs(kernel_meta_path_, true));
  initialized_ = true;
}

std::string KernelMeta::Search(const std::string &kernel_name) const {
  if (!initialized_) {
    return "";
  }

  auto iter = kernel_meta_map_.find(kernel_name);
  if (iter == kernel_meta_map_.end()) {
    return "";
  } else {
    return iter->second;
  }
}

bool KernelMeta::Insert(const std::string &kernel_name, const std::string &kernel_json) {
  if (!initialized_) {
    return false;
  }
  kernel_meta_map_[kernel_name] = kernel_json;
  return true;
}

void SaveJsonInfo(const std::string &json_name, const std::string &info, const std::string &base_path) {
  std::string path = base_path + json_name + kInfoSuffix;
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream filewrite(realpath.value());
  if (!filewrite.is_open()) {
    MS_LOG(ERROR) << "Open file '" << realpath.value() << "' failed!";
    return;
  }
  filewrite << info << std::endl;
  filewrite.close();
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list) {
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_lists = TopoSort(func_graph->get_return());
  for (auto const &node : node_lists) {
    if (!AnfUtils::IsRealKernel(node) || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex))) {
      node_list->push_back(node);
    }
  }
}

void GetValidKernelNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *node_list,
                         std::vector<AnfNodePtr> *input_list, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node_list);
  MS_EXCEPTION_IF_NULL(input_list);

  GetValidKernelNodes(func_graph, node_list);

  auto parameters = func_graph->parameters();
  (void)input_list->insert(input_list->cbegin(), parameters.begin(), parameters.end());

  GetFuncGraphOutputNodes(func_graph, output_list);
}

void GetFuncGraphOutputNodes(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *output_list) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(output_list);
  auto func_output = func_graph->output();
  MS_EXCEPTION_IF_NULL(func_output);
  if (func_output->isa<CNode>()) {
    // multi output.
    auto cnode = func_output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input0 = cnode->input(kAnfPrimitiveIndex);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      for (size_t input_idx = 1; input_idx < cnode->size(); ++input_idx) {
        auto input_node = cnode->input(input_idx);
        MS_EXCEPTION_IF_NULL(input_node);
        if (input_node->isa<CNode>() && common::AnfAlgo::GetInputTensorNum(input_node) == 0) {
          continue;
        }
        output_list->push_back(common::AnfAlgo::VisitKernel(input_node, 0).first);
      }
    } else {
      // single output.
      output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
    }
  } else {
    // single output.
    output_list->push_back(common::AnfAlgo::VisitKernel(func_output, 0).first);
  }
}

std::string GetStrProcessorFromContext() {
  string str_processor = kernel::kProcessorUnknown;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_info = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_info == kGPUDevice) {
    str_processor = kernel::kProcessorCuda;
  } else if (device_info == kAscendDevice) {
    str_processor = kernel::kProcessorAiCore;
  } else if (device_info == kCPUDevice) {
    str_processor = kernel::kProcessorCpu;
  }
  return str_processor;
}

bool IsDynamicParamKernel(const std::string &op_name) {
  const auto &op_info = kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU);
  constexpr auto kParamDynamic = "dynamic";

  if (op_info == nullptr) {
    return false;
  }

  const auto &input_io_info = op_info->inputs_ptr();
  if (input_io_info.size() != 1 || input_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  const auto &output_io_info = op_info->outputs_ptr();
  if (output_io_info.size() != 1 || output_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  return true;
}

// In compile stage, run resize when kernel is not dynamic shape or has no value depend list.
bool CheckResizeCondition(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->input(0));
  auto value_node = node->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value().get();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<FuncGraph>()) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(DEBUG) << "Skip resize for " << node->DebugString() << ", for reason is dynamic shape";
      return false;
    }
    if (common::AnfAlgo::IsDynamicValue(node)) {
      MS_LOG(DEBUG) << "Skip resize for " << node->DebugString() << ", for reason is dynamic value";
      return false;
    }
  }

  return true;
}

}  // namespace kernel
}  // namespace mindspore
