/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "utils/hyper_offload/hyper_offload_mem_estimate.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "include/utils/utils.h"
#include "mindapi/base/type_id.h"
#include "ir/dtype/type.h"
#include "utils/anf_utils.h"
#include "utils/compile_config.h"
#include "utils/file_utils.h"
#include "utils/shape_utils.h"
#include "include/utils/anfalgo.h"
#include "utils/hyper_offload/hyper_offload_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

using NodeFirstLastMap = std::map<AnfNodePtr, std::pair<size_t, size_t>>;
using OutputLastUsageMap = std::map<std::pair<AnfNodePtr, size_t>, size_t>;

struct HorizonSearchContext {
  const UserInfoList &user_info_list;
  const std::map<AnfNodePtr, size_t> &node_to_index;
  const OutputLastUsageMap &base_last;
};

struct HorizonSearchState {
  std::unordered_map<AnfNodePtr, size_t> *memo;
  std::unordered_set<AnfNodePtr> *rec_stack;
};

namespace {
constexpr size_t kCsvSizeColumnIndex = 5;
constexpr size_t kCsvActualMemoryColumnIndex = 6;
constexpr size_t kCsvNodeNameColumnIndex = 13;
constexpr size_t kCsvExtraInfoColumnIndex = 15;
constexpr size_t kProfilePaddingSlots = 2;
constexpr size_t kMemAlignSize = 512;
// Keep the historical extra slack used by the fallback estimator before rounding to 512-byte alignment.
constexpr size_t kMemAlignSlackBytes = 31;
constexpr size_t kMinimumAlignedBytes = 1;
constexpr size_t kDefaultDynamicShapePlaceholderBytes = 128ULL * 1024 * 1024;
constexpr char kCsvNodeNameUnavailable[] = "N/A";
constexpr char kCsvWorkspacePulseMarker[] = "{}";
constexpr char kOpSuffixPrefix[] = "-op";
constexpr size_t kOpSuffixPrefixLength = sizeof(kOpSuffixPrefix) - 1;

void ExtendOutputLastUsageForBufferAliasing(const UserInfoList &user_info_list,
                                            const std::map<AnfNodePtr, size_t> &node_to_index,
                                            const CNodePtrList &index_order, OutputLastUsageMap *out_last_usage);

bool HasTrackedConsumerBeyondTarget(const UserInfo &info, const std::map<AnfNodePtr, size_t> &node_to_index,
                                    size_t target_last) {
  for (const auto &user : info.users) {
    const auto index_it = node_to_index.find(user.first);
    if (index_it != node_to_index.end() && index_it->second >= target_last) {
      return true;
    }
  }
  return false;
}

bool AppendSyntheticTailUser(const CNodePtrList &index_order, size_t target_last, const KernelWithIndex &data_node,
                             UserInfo *info) {
  MS_EXCEPTION_IF_NULL(info);
  if (index_order.empty()) {
    return false;
  }
  const size_t clamped_index = std::min(target_last, index_order.size() - 1);
  const auto &tail = index_order[clamped_index];
  if (tail == nullptr || !tail->isa<CNode>()) {
    return false;
  }
  const auto tail_cnode = tail->cast_ptr<CNode>();
  if (tail_cnode == nullptr) {
    return false;
  }
  constexpr size_t kFirstInputIndex = 1;
  for (size_t input_index = kFirstInputIndex; input_index < tail_cnode->inputs().size(); ++input_index) {
    if (GetRealDataNodeWithIndex(tail_cnode->input(input_index)) != data_node) {
      continue;
    }
    const KernelWithIndex candidate{tail, input_index};
    const auto duplicate = std::find(info->users.begin(), info->users.end(), candidate);
    if (duplicate == info->users.end()) {
      info->users.emplace_back(candidate);
      return true;
    }
    return false;
  }
  return false;
}

size_t GetAlignedMemoryBytes(size_t original_size) {
  if (original_size == 0) {
    return 0;
  }
  return ((original_size + kMemAlignSize + kMemAlignSlackBytes) / kMemAlignSize) * kMemAlignSize;
}

size_t GetDynamicShapePlaceholderBytes() {
  static const size_t placeholder_bytes = []() -> size_t {
    const std::string value = common::GetCompileConfig("HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES");
    if (value.empty()) {
      return kDefaultDynamicShapePlaceholderBytes;
    }
    try {
      const auto parsed = std::stoull(value, nullptr, 10);
      if (parsed == 0ULL || parsed > std::numeric_limits<size_t>::max()) {
        MS_LOG(WARNING) << "[HyperOffload] invalid HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES=" << value
                        << ", use default " << kDefaultDynamicShapePlaceholderBytes;
        return kDefaultDynamicShapePlaceholderBytes;
      }
      return static_cast<size_t>(parsed);
    } catch (const std::exception &) {
      MS_LOG(WARNING) << "[HyperOffload] failed to parse HYPER_OFFLOAD_DYNAMIC_SHAPE_PLACEHOLDER_BYTES=" << value
                      << ", use default " << kDefaultDynamicShapePlaceholderBytes;
      return kDefaultDynamicShapePlaceholderBytes;
    }
  }();
  return placeholder_bytes;
}
}  // namespace

void FillLastUsageFromUserInfo(const UserInfoList &user_info_list, const std::map<AnfNodePtr, size_t> &node_to_index,
                               OutputLastUsageMap *out_last_usage) {
  for (const auto &info : user_info_list) {
    auto node = info.node.first;
    size_t out_idx = info.node.second;
    if (node_to_index.find(node) == node_to_index.end()) {
      continue;
    }
    size_t last_usage_idx = node_to_index.at(node);
    for (const auto &user : info.users) {
      auto user_node = user.first;
      auto u_it = node_to_index.find(user_node);
      if (u_it != node_to_index.end() && u_it->second > last_usage_idx) {
        last_usage_idx = u_it->second;
      }
    }
    (*out_last_usage)[{node, out_idx}] = last_usage_idx;
  }
}

void BuildOutputLastUsageMap(const UserInfoList &user_info_list, const std::map<AnfNodePtr, size_t> &node_to_index,
                             const CNodePtrList &index_order, OutputLastUsageMap *out_last_usage) {
  out_last_usage->clear();
  FillLastUsageFromUserInfo(user_info_list, node_to_index, out_last_usage);
  ExtendOutputLastUsageForBufferAliasing(user_info_list, node_to_index, index_order, out_last_usage);
}

void FillLastUsageFromGraphOutputs(const FuncGraphPtr &graph, const std::map<AnfNodePtr, size_t> &node_to_index,
                                   size_t graph_end_index, OutputLastUsageMap *out_last_usage) {
  if (graph == nullptr || graph->output() == nullptr) {
    return;
  }
  try {
    const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output : outputs) {
      auto producer = GetRealDataNodeWithIndex(output.first);
      if (producer.first != nullptr && producer.first->isa<CNode>() &&
          node_to_index.find(producer.first) != node_to_index.end()) {
        (*out_last_usage)[producer] = graph_end_index;
      }
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "[HyperOffload] failed to resolve graph outputs for fallback memory estimation: " << e.what();
  }
}

void FillLastUsageFromSummaryNodes(const FuncGraphPtr &graph, const std::map<AnfNodePtr, size_t> &node_to_index,
                                   size_t graph_end_index, OutputLastUsageMap *out_last_usage) {
  if (graph == nullptr) {
    return;
  }
  const auto &summary_outputs = graph->user_data<std::vector<KernelWithIndex>>(kHyperOffloadSummaryOutputs);
  if (summary_outputs == nullptr) {
    return;
  }
  for (const auto &resolved : *summary_outputs) {
    if (resolved.first == nullptr) {
      continue;
    }
    if (node_to_index.find(resolved.first) != node_to_index.end()) {
      (*out_last_usage)[resolved] = graph_end_index;
    }
  }
}

void AugmentUserInfoListForOffloadSpan(UserInfoList *user_info_list, const CNodePtrList &index_order,
                                       const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size) {
  if (user_info_list == nullptr || index_order.empty()) {
    return;
  }
  (void)order_size;
  OutputLastUsageMap out_last_usage;
  BuildOutputLastUsageMap(*user_info_list, node_to_index, index_order, &out_last_usage);
  for (auto &info : *user_info_list) {
    auto out_it = out_last_usage.find(info.node);
    if (out_it == out_last_usage.end()) {
      continue;
    }
    const size_t target_last = out_it->second;
    if (HasTrackedConsumerBeyondTarget(info, node_to_index, target_last)) {
      continue;
    }
    (void)AppendSyntheticTailUser(index_order, target_last, info.node, &info);
  }
}

namespace {

// The CSV may contain multiple rows per node name. The first valid row seeds a provisional
// baseline and later rows preserve per-output ordering for multi-output nodes.
using NameToSizesMap = std::unordered_map<std::string, std::vector<size_t>>;
NameToSizesMap g_name_to_sizes;
// Preserve file order for repeated node names so output binding stays deterministic.
NameToSizesMap g_name_to_g_column_bytes;
std::unordered_map<std::string, size_t> g_name_to_workspace_pulse_bytes;
// The G column is used only for baseline estimation. Candidate sizes always come from F.
size_t g_csv_baseline_bytes = 0;

static std::string TrimCell(const std::string &s) {
  size_t start_pos = 0;
  while (start_pos < s.size() && std::isspace(static_cast<unsigned char>(s[start_pos]))) {
    ++start_pos;
  }
  size_t end_pos = s.size();
  while (end_pos > start_pos && std::isspace(static_cast<unsigned char>(s[end_pos - 1]))) {
    --end_pos;
  }
  return s.substr(start_pos, end_pos - start_pos);
}

static void SplitCsvLineToCells(const std::string &line, std::vector<std::string> *cols) {
  MS_EXCEPTION_IF_NULL(cols);
  cols->clear();
  std::istringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    cols->push_back(cell);
  }
}

// Memory values are stored as decimal byte counts without unit suffixes.
static bool ParseFColumnBytes(const std::string &cell, size_t *out_bytes) {
  if (out_bytes == nullptr) {
    return false;
  }
  *out_bytes = 0;
  const std::string t = TrimCell(cell);
  if (t.empty() || t == kCsvNodeNameUnavailable) {
    return false;
  }
  const bool is_decimal_bytes =
    std::all_of(t.begin(), t.end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; });
  if (!is_decimal_bytes) {
    return false;
  }
  try {
    const auto parsed_value = std::stoull(t, nullptr, 10);
    if (parsed_value == 0ULL) {
      return false;
    }
    if (parsed_value > std::numeric_limits<size_t>::max()) {
      return false;
    }
    *out_bytes = static_cast<size_t>(parsed_value);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

// Load the memory-block rows used by peak modeling. Output rows feed per-output binding,
// while rows with P=="{}" are treated as one-step workspace pulses for the owning operator.
// Column F is the preferred modeled size and column G is the observed memory fallback.
static void ProcessMemoryBlockCsvRow(const std::vector<std::string> &cols,
                                     std::unordered_map<std::string, std::vector<size_t>> *name_to_sizes,
                                     std::unordered_map<std::string, std::vector<size_t>> *name_to_g_column_bytes,
                                     std::unordered_map<std::string, size_t> *name_to_workspace_pulse_bytes,
                                     bool *baseline_initialized) {
  MS_EXCEPTION_IF_NULL(name_to_sizes);
  MS_EXCEPTION_IF_NULL(name_to_g_column_bytes);
  MS_EXCEPTION_IF_NULL(name_to_workspace_pulse_bytes);
  MS_EXCEPTION_IF_NULL(baseline_initialized);
  if (cols.size() <= kCsvNodeNameColumnIndex) {
    return;
  }
  const std::string node_name = TrimCell(cols[kCsvNodeNameColumnIndex]);
  if (node_name.empty() || node_name == kCsvNodeNameUnavailable) {
    return;
  }
  size_t actual_memory_bytes = 0;
  const bool has_actual_memory_bytes = ParseFColumnBytes(cols[kCsvActualMemoryColumnIndex], &actual_memory_bytes);
  size_t size_column_bytes = 0;
  const bool has_size_column_bytes = ParseFColumnBytes(cols[kCsvSizeColumnIndex], &size_column_bytes);
  const bool is_workspace_pulse =
    cols.size() > kCsvExtraInfoColumnIndex && TrimCell(cols[kCsvExtraInfoColumnIndex]) == kCsvWorkspacePulseMarker;
  if (is_workspace_pulse) {
    const size_t workspace_pulse_bytes = has_size_column_bytes
                                           ? size_column_bytes
                                           : (has_actual_memory_bytes ? actual_memory_bytes : static_cast<size_t>(0));
    if (workspace_pulse_bytes > 0) {
      (*name_to_workspace_pulse_bytes)[node_name] += workspace_pulse_bytes;
    }
    return;
  }
  if (!*baseline_initialized) {
    const size_t baseline_size_bytes = has_size_column_bytes ? size_column_bytes : static_cast<size_t>(0);
    if (has_actual_memory_bytes && has_size_column_bytes) {
      g_csv_baseline_bytes = (actual_memory_bytes >= baseline_size_bytes) ? (actual_memory_bytes - baseline_size_bytes)
                                                                          : static_cast<size_t>(0);
    } else {
      g_csv_baseline_bytes = 0;
    }
    *baseline_initialized = true;
  }
  // Keep one vector slot per CSV row so repeated node names preserve file order.
  (*name_to_sizes)[node_name].push_back(has_size_column_bytes ? size_column_bytes : static_cast<size_t>(0));
  (*name_to_g_column_bytes)[node_name].push_back(has_actual_memory_bytes ? actual_memory_bytes
                                                                         : static_cast<size_t>(0));
}

static bool CommitMemoryBlockCsvData(const CNodePtrList &index_order,
                                     const std::unordered_map<std::string, std::vector<size_t>> &name_to_sizes,
                                     const std::unordered_map<std::string, std::vector<size_t>> &name_to_g_column_bytes,
                                     const std::unordered_map<std::string, size_t> &name_to_workspace_pulse_bytes,
                                     bool baseline_initialized) {
  if (!baseline_initialized && name_to_sizes.empty()) {
    return false;
  }
  g_name_to_sizes = name_to_sizes;
  g_name_to_g_column_bytes = name_to_g_column_bytes;
  g_name_to_workspace_pulse_bytes = name_to_workspace_pulse_bytes;
  return std::any_of(index_order.begin(), index_order.end(), [](const CNodePtr &node) { return node != nullptr; });
}

static bool LoadMemoryBlockCsv(const std::string &path, const CNodePtrList &index_order) {
  g_name_to_sizes.clear();
  g_name_to_g_column_bytes.clear();
  g_name_to_workspace_pulse_bytes.clear();
  g_csv_baseline_bytes = 0;
  const auto real_path = FileUtils::GetRealPath(common::SafeCStr(path));
  if (!real_path.has_value()) {
    return false;
  }
  std::ifstream in(real_path.value());
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line)) {
    return false;
  }
  std::unordered_map<std::string, std::vector<size_t>> name_to_sizes;
  std::unordered_map<std::string, std::vector<size_t>> name_to_g_column_bytes;
  std::unordered_map<std::string, size_t> name_to_workspace_pulse_bytes;
  bool baseline_initialized = false;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> cols;
    SplitCsvLineToCells(line, &cols);
    ProcessMemoryBlockCsvRow(cols, &name_to_sizes, &name_to_g_column_bytes, &name_to_workspace_pulse_bytes,
                             &baseline_initialized);
  }
  return CommitMemoryBlockCsvData(index_order, name_to_sizes, name_to_g_column_bytes, name_to_workspace_pulse_bytes,
                                  baseline_initialized);
}

size_t GetNodeSizeFromKernelMod(const AnfNodePtr &node, size_t output_index) {
  if (node == nullptr) {
    return 0;
  }
  const auto &output_sizes = node->user_data<std::vector<size_t>>(kHyperOffloadKernelModOutputSizes);
  if (output_sizes == nullptr || output_index >= output_sizes->size()) {
    return 0;
  }
  return (*output_sizes)[output_index];
}

size_t GetNodeSizeFromInfer(const AnfNodePtr &node, size_t output_index) {
  if (node == nullptr) {
    return 0;
  }
  try {
    const ShapeVector shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
    const TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, output_index);
    const size_t element_size = abstract::TypeIdSize(type_id);
    if (element_size == 0) {
      return 0;
    }
    int64_t element_count = 1;
    for (int64_t dim : shape) {
      if (dim <= 0) {
        return 0;
      }
      element_count *= dim;
    }
    return static_cast<size_t>(element_count) * element_size;
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "[HyperOffload] failed to infer tensor size for node " << node->DebugString()
                    << ", output_index=" << output_index << ": " << e.what();
    return 0;
  }
}

bool IsDynamicShapeNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  try {
    return common::AnfAlgo::IsDynamic(node);
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "[HyperOffload] failed to query dynamic-shape state for node " << node->DebugString() << ": "
                    << e.what();
    return false;
  }
}

size_t GetNodeSizeFromDeviceMetadata(const AnfNodePtr &node, size_t output_index) {
  if (node == nullptr || IsDynamicShapeNode(node)) {
    return 0;
  }
  const auto &output_sizes = node->user_data<std::vector<size_t>>(kHyperOffloadOutputMemSizes);
  if (output_sizes == nullptr || output_index >= output_sizes->size()) {
    return 0;
  }
  return (*output_sizes)[output_index];
}

size_t EstimateNodeMemorySize(const AnfNodePtr &node, size_t output_index) {
  if (IsDynamicShapeNode(node)) {
    return GetDynamicShapePlaceholderBytes();
  }
  const size_t device_metadata_size = GetNodeSizeFromDeviceMetadata(node, output_index);
  if (device_metadata_size > 0) {
    return device_metadata_size;
  }
  const size_t kernel_mod_size = GetNodeSizeFromKernelMod(node, output_index);
  if (kernel_mod_size > 0) {
    return kernel_mod_size;
  }
  const size_t raw_size = GetNodeSizeFromInfer(node, output_index);
  if (raw_size > 0) {
    return GetAlignedMemoryBytes(raw_size);
  }
  return GetAlignedMemoryBytes(kMinimumAlignedBytes);
}

size_t GetWorkspaceSizeFromKernelMod(const AnfNodePtr &node) {
  if (node == nullptr) {
    return 0;
  }
  const auto &workspace_size = node->user_data<size_t>(kHyperOffloadWorkspaceSize);
  return workspace_size == nullptr ? 0 : *workspace_size;
}

size_t EstimateNodeWorkspaceSize(const AnfNodePtr &node) {
  const size_t workspace_size = GetWorkspaceSizeFromKernelMod(node);
  if (workspace_size > 0) {
    return workspace_size;
  }
  if (IsDynamicShapeNode(node)) {
    return GetDynamicShapePlaceholderBytes();
  }
  return 0;
}

// Return true when the full name already ends with "-op<digits>".
static bool HasTrailingOpNameSuffix(const std::string &s) {
  const size_t pos = s.rfind(kOpSuffixPrefix);
  if (pos == std::string::npos) {
    return false;
  }
  size_t j = pos + kOpSuffixPrefixLength;
  if (j >= s.size()) {
    return false;
  }
  if (!std::isdigit(static_cast<unsigned char>(s[j]))) {
    return false;
  }
  while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
    ++j;
  }
  return j == s.size();
}

// Resolve the CSV key used for one logical output:
// 1. If the full name already ends with "-op<output_index>", use it directly.
// 2. If output 0 already ends with "-op<digits>", keep the original full name.
// 3. Otherwise append "-op<output_index>".
static std::string MemoryBlockTableKeyForOutput(const std::string &fullname_with_scope, size_t output_index) {
  const std::string op_suffix = std::string(kOpSuffixPrefix) + std::to_string(output_index);
  if (fullname_with_scope.size() >= op_suffix.size() &&
      fullname_with_scope.compare(fullname_with_scope.size() - op_suffix.size(), op_suffix.size(), op_suffix) == 0) {
    return fullname_with_scope;
  }
  if (output_index == 0 && HasTrailingOpNameSuffix(fullname_with_scope)) {
    return fullname_with_scope;
  }
  return fullname_with_scope + op_suffix;
}

static size_t TableOutputCount(const AnfNodePtr &node) {
  if (node == nullptr) {
    return 0;
  }
  const auto &output_count = node->user_data<size_t>(kHyperOffloadOutputCount);
  if (output_count != nullptr) {
    return *output_count;
  }
  auto cnode = node->cast_ptr<CNode>();
  if (cnode == nullptr) {
    return 1;
  }
  return AnfUtils::GetOutputTensorNum(node);
}

static const std::vector<size_t> *FindCsvRowsByKey(const std::string &key) {
  auto it = g_name_to_sizes.find(key);
  return (it != g_name_to_sizes.end() && !it->second.empty()) ? &it->second : nullptr;
}

static bool TryResolveExplicitCsvRowBinding(const std::string &explicit_key, size_t output_index, size_t output_count,
                                            std::string *map_key, size_t *row_in_vector) {
  MS_EXCEPTION_IF_NULL(map_key);
  MS_EXCEPTION_IF_NULL(row_in_vector);
  const std::vector<size_t> *explicit_rows = FindCsvRowsByKey(explicit_key);
  if (explicit_rows == nullptr) {
    return false;
  }
  if (output_count <= 1) {
    if (explicit_rows->size() == 1UL) {
      *map_key = explicit_key;
      *row_in_vector = 0;
      return true;
    }
    if (explicit_rows->size() > 1UL) {
      MS_LOG(WARNING) << "[HyperOffload] CSV has " << explicit_rows->size() << " rows for N=\"" << explicit_key
                      << "\" but kernel output_count=1; using first row only (expect one row per single-output key).";
      *map_key = explicit_key;
      *row_in_vector = 0;
      return true;
    }
    return false;
  }
  if (explicit_rows->size() == output_count) {
    *map_key = explicit_key;
    *row_in_vector = output_index;
    return true;
  }
  if (explicit_rows->size() == 1UL) {
    *map_key = explicit_key;
    *row_in_vector = 0;
    return true;
  }
  if (explicit_rows->size() < output_count) {
    if (output_index == 0) {
      MS_LOG(WARNING) << "[HyperOffload] CSV N=\"" << explicit_key << "\" row_count=" << explicit_rows->size()
                      << " < output_count=" << output_count << "; binding outputs [0," << explicit_rows->size()
                      << ") only, remaining outputs size=0.";
    }
    if (output_index < explicit_rows->size()) {
      *map_key = explicit_key;
      *row_in_vector = output_index;
      return true;
    }
    return false;
  }
  MS_LOG(WARNING) << "[HyperOffload] CSV N=\"" << explicit_key << "\" row_count=" << explicit_rows->size()
                  << " > output_count=" << output_count << "; skip F/G binding for this output.";
  return false;
}

static bool TryResolveOutputZeroCsvRowBinding(const std::string &fullname, size_t output_index, size_t output_count,
                                              std::string *map_key, size_t *row_in_vector) {
  if (output_count <= 1U || output_index == 0) {
    return false;
  }
  const std::string output_zero_key = MemoryBlockTableKeyForOutput(fullname, 0);
  const std::vector<size_t> *output_zero_rows = FindCsvRowsByKey(output_zero_key);
  if (output_zero_rows == nullptr) {
    return false;
  }
  if (output_zero_rows->size() == output_count) {
    *map_key = output_zero_key;
    *row_in_vector = output_index;
    return true;
  }
  if (output_zero_rows->size() < output_count && output_index < output_zero_rows->size()) {
    *map_key = output_zero_key;
    *row_in_vector = output_index;
    return true;
  }
  return false;
}

// Bind one `(node, output_index)` pair to a concrete CSV row.
// Multi-output nodes prefer exact row counts, tolerate short tables by zero-filling
// later outputs, and skip ambiguous oversized tables.
static bool ResolveCsvRowBinding(const AnfNodePtr &node, size_t output_index, std::string *map_key,
                                 size_t *row_in_vector) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(map_key);
  MS_EXCEPTION_IF_NULL(row_in_vector);
  if (g_name_to_sizes.empty()) {
    return false;
  }
  const size_t output_count = TableOutputCount(node);
  const std::string fullname = node->fullname_with_scope();
  const std::string explicit_key = MemoryBlockTableKeyForOutput(fullname, output_index);

  if (TryResolveExplicitCsvRowBinding(explicit_key, output_index, output_count, map_key, row_in_vector)) {
    return true;
  }
  if (TryResolveOutputZeroCsvRowBinding(fullname, output_index, output_count, map_key, row_in_vector)) {
    return true;
  }
  return false;
}

size_t LookupOperatorMemorySize(const AnfNodePtr &node, size_t exec_order_index, size_t output_index) {
  (void)exec_order_index;
  if (node == nullptr || g_name_to_sizes.empty()) {
    return 0;
  }
  std::string map_key;
  size_t row_in_vector = 0;
  if (!ResolveCsvRowBinding(node, output_index, &map_key, &row_in_vector)) {
    return 0;
  }
  auto name_it = g_name_to_sizes.find(map_key);
  if (name_it == g_name_to_sizes.end() || row_in_vector >= name_it->second.size()) {
    return 0;
  }
  return name_it->second[row_in_vector];
}

// Read the G column for baseline estimation only.
size_t LookupOperatorGColumnBytes(const AnfNodePtr &node, size_t exec_order_index, size_t output_index) {
  (void)exec_order_index;
  if (node == nullptr || g_name_to_g_column_bytes.empty()) {
    return 0;
  }
  std::string map_key;
  size_t row_in_vector = 0;
  if (!ResolveCsvRowBinding(node, output_index, &map_key, &row_in_vector)) {
    return 0;
  }
  auto g_column_it = g_name_to_g_column_bytes.find(map_key);
  if (g_column_it == g_name_to_g_column_bytes.end() || row_in_vector >= g_column_it->second.size()) {
    return 0;
  }
  return g_column_it->second[row_in_vector];
}

// Derive the baseline from the first execution-order entry that can be bound back to the CSV.
static bool ApplyProfileBaselineFromFirstExecOrderTableHit(const CNodePtrList &index_order) {
  for (size_t i = 0; i < index_order.size(); ++i) {
    const auto &kernel = index_order[i];
    if (kernel == nullptr) {
      continue;
    }
    const size_t output_count = TableOutputCount(kernel);
    for (size_t output_index = 0; output_index < output_count; ++output_index) {
      const size_t modeled_size_bytes = LookupOperatorMemorySize(kernel, i, output_index);
      const size_t actual_memory_bytes = LookupOperatorGColumnBytes(kernel, i, output_index);
      if (modeled_size_bytes == 0 && actual_memory_bytes == 0) {
        continue;
      }
      g_csv_baseline_bytes = (actual_memory_bytes >= modeled_size_bytes) ? (actual_memory_bytes - modeled_size_bytes)
                                                                         : static_cast<size_t>(0);
      return true;
    }
  }
  g_csv_baseline_bytes = 0;
  MS_LOG(WARNING)
    << "[HyperOffload] skip profile modeling for this graph: no exec-order op with CSV row (G or F) for any "
       "output key.";
  return false;
}

// Priority: env override, compile-config override, then the hard-coded fallback path.
static std::string GetMemoryBlockCsvPath() {
  std::string path = common::GetEnv("HYPER_OFFLOAD_MEMORY_BLOCK_PATH");
  if (path.empty()) {
    path = common::GetCompileConfig("HYPER_OFFLOAD_MEMORY_BLOCK_PATH");
  }
  return path;
}

bool UseMemoryBlockCsvModeling() { return !g_name_to_sizes.empty(); }

size_t LookupOperatorWorkspacePulseFromCsv(const AnfNodePtr &node) {
  if (node == nullptr || g_name_to_workspace_pulse_bytes.empty()) {
    return 0;
  }
  const auto explicit_it = g_name_to_workspace_pulse_bytes.find(node->fullname_with_scope());
  if (explicit_it != g_name_to_workspace_pulse_bytes.end()) {
    return explicit_it->second;
  }
  const std::string output_zero_key = MemoryBlockTableKeyForOutput(node->fullname_with_scope(), 0);
  const auto output_zero_it = g_name_to_workspace_pulse_bytes.find(output_zero_key);
  if (output_zero_it != g_name_to_workspace_pulse_bytes.end()) {
    return output_zero_it->second;
  }
  return 0;
}

// Return true when the CSV provides a resolvable row for this output.
static bool TableHasKeyForOutput(const AnfNodePtr &node, size_t out_idx) {
  if (node == nullptr || g_name_to_sizes.empty()) {
    return false;
  }
  std::string map_key;
  size_t row_in_vector = 0;
  return ResolveCsvRowBinding(node, out_idx, &map_key, &row_in_vector);
}

static bool NodeNameListedInCsvNColumn(const AnfNodePtr &node) {
  if (node == nullptr || g_name_to_sizes.empty()) {
    return false;
  }
  const size_t output_count = TableOutputCount(node);
  for (size_t output_index = 0; output_index < output_count; ++output_index) {
    if (TableHasKeyForOutput(node, output_index)) {
      return true;
    }
  }
  return false;
}

static bool ProducerOutputListedInCsvNColumn(const AnfNodePtr &producer, size_t out_idx) {
  return TableHasKeyForOutput(producer, out_idx);
}

// After FillLastUsageFromUserInfo only: max exec index and max per-output last for this producer.
static size_t ProducerLastInBase(const AnfNodePtr &producer, const OutputLastUsageMap &base_last,
                                 const std::map<AnfNodePtr, size_t> &node_to_index) {
  size_t max_exec_index = 0;
  bool any = false;
  auto it = node_to_index.find(producer);
  if (it != node_to_index.end()) {
    max_exec_index = it->second;
    any = true;
  }
  for (const auto &kv : base_last) {
    if (kv.first.first == producer) {
      any = true;
      if (kv.second > max_exec_index) {
        max_exec_index = kv.second;
      }
    }
  }
  return any ? max_exec_index : 0;
}

// Return true when every scheduled user of the producer can be found in the CSV.
static bool AllScheduledUsersListedInCsvNColumn(const AnfNodePtr &producer, const UserInfoList &user_info_list,
                                                const std::map<AnfNodePtr, size_t> &node_to_index) {
  return std::all_of(user_info_list.begin(), user_info_list.end(), [&producer, &node_to_index](const auto &info) {
    if (info.node.first != producer) {
      return true;
    }
    return std::all_of(info.users.begin(), info.users.end(), [&node_to_index](const auto &uk) {
      const auto &un = uk.first;
      if (un == nullptr || node_to_index.find(un) == node_to_index.end()) {
        return true;
      }
      return NodeNameListedInCsvNColumn(un);
    });
  });
}

static size_t NodeExecIndexOrZero(const AnfNodePtr &node, const std::map<AnfNodePtr, size_t> &node_to_index) {
  const auto node_exec_it = node_to_index.find(node);
  return (node_exec_it != node_to_index.end()) ? node_exec_it->second : 0;
}

static void ValidateHorizonSearchState(const HorizonSearchState &state) {
  MS_EXCEPTION_IF_NULL(state.memo);
  MS_EXCEPTION_IF_NULL(state.rec_stack);
}

static size_t HorizonFromMissingTableConsumer(const AnfNodePtr &node, const HorizonSearchContext &context,
                                              const HorizonSearchState &state);

static size_t MissingTableUserHorizon(const AnfNodePtr &user_node, const HorizonSearchContext &context,
                                      const HorizonSearchState &state);

static std::optional<size_t> MissingTableNextUserHorizon(const AnfNodePtr &next_user,
                                                         const HorizonSearchContext &context,
                                                         const HorizonSearchState &state) {
  if (next_user == nullptr) {
    return std::nullopt;
  }
  const auto next_exec_it = context.node_to_index.find(next_user);
  if (next_exec_it == context.node_to_index.end()) {
    return std::nullopt;
  }
  if (NodeNameListedInCsvNColumn(next_user)) {
    return next_exec_it->second;
  }
  return HorizonFromMissingTableConsumer(next_user, context, state);
}

static size_t ComputeMissingTableConsumerHorizon(const AnfNodePtr &node, const HorizonSearchContext &context,
                                                 const HorizonSearchState &state) {
  if (AllScheduledUsersListedInCsvNColumn(node, context.user_info_list, context.node_to_index)) {
    return ProducerLastInBase(node, context.base_last, context.node_to_index);
  }
  return MissingTableUserHorizon(node, context, state);
}

// Walk through missing-table consumers until a schedulable horizon can be recovered.
// `memo` avoids repeated work and `rec_stack` protects against cycles.
static size_t HorizonFromMissingTableConsumer(const AnfNodePtr &node, const HorizonSearchContext &context,
                                              const HorizonSearchState &state) {
  ValidateHorizonSearchState(state);
  if (node == nullptr) {
    return 0;
  }
  auto memo_it = state.memo->find(node);
  if (memo_it != state.memo->end()) {
    return memo_it->second;
  }
  if (state.rec_stack->count(node) != 0) {
    return ProducerLastInBase(node, context.base_last, context.node_to_index);
  }
  state.rec_stack->insert(node);
  const size_t result = ComputeMissingTableConsumerHorizon(node, context, state);
  (void)state.rec_stack->erase(node);
  (*state.memo)[node] = result;
  return result;
}

static size_t MissingTableUserHorizon(const AnfNodePtr &user_node, const HorizonSearchContext &context,
                                      const HorizonSearchState &state) {
  size_t result = NodeExecIndexOrZero(user_node, context.node_to_index);
  for (const auto &info : context.user_info_list) {
    if (info.node.first != user_node) {
      continue;
    }
    for (const auto &uk : info.users) {
      const auto next_horizon = MissingTableNextUserHorizon(uk.first, context, state);
      if (next_horizon.has_value()) {
        result = std::max(result, next_horizon.value());
      }
    }
  }
  return result;
}

static size_t HorizonFromAliasConsumer(const AnfNodePtr &node, const HorizonSearchContext &context,
                                       const HorizonSearchState &state);

static size_t AliasUserHorizon(const AnfNodePtr &user_node, const HorizonSearchContext &context,
                               const HorizonSearchState &state);

static size_t ComputeAliasConsumerHorizon(const AnfNodePtr &node, const HorizonSearchContext &context,
                                          const HorizonSearchState &state) {
  size_t result = ProducerLastInBase(node, context.base_last, context.node_to_index);
  if (!IsViewOrInplaceNode(node)) {
    return result;
  }
  for (const auto &info : context.user_info_list) {
    if (info.node.first != node) {
      continue;
    }
    for (const auto &uk : info.users) {
      result = std::max(result, AliasUserHorizon(uk.first, context, state));
    }
  }
  return result;
}

static size_t HorizonFromAliasConsumer(const AnfNodePtr &node, const HorizonSearchContext &context,
                                       const HorizonSearchState &state) {
  ValidateHorizonSearchState(state);
  if (node == nullptr) {
    return 0;
  }
  auto memo_it = state.memo->find(node);
  if (memo_it != state.memo->end()) {
    return memo_it->second;
  }
  if (state.rec_stack->count(node) != 0) {
    return ProducerLastInBase(node, context.base_last, context.node_to_index);
  }
  state.rec_stack->insert(node);
  const size_t result = ComputeAliasConsumerHorizon(node, context, state);
  (void)state.rec_stack->erase(node);
  (*state.memo)[node] = result;
  return result;
}

static size_t AliasUserHorizon(const AnfNodePtr &user_node, const HorizonSearchContext &context,
                               const HorizonSearchState &state) {
  if (user_node == nullptr) {
    return 0;
  }
  const auto user_it = context.node_to_index.find(user_node);
  if (user_it == context.node_to_index.end()) {
    return 0;
  }
  if (IsViewOrInplaceNode(user_node)) {
    return HorizonFromAliasConsumer(user_node, context, state);
  }
  return user_it->second;
}

// Extend last-use indices across missing-table consumers when the producer itself is tracked.
// In fallback modeling without CSV, only alias-related users are followed recursively.
void ExtendOutputLastUsageForBufferAliasing(const UserInfoList &user_info_list,
                                            const std::map<AnfNodePtr, size_t> &node_to_index,
                                            const CNodePtrList &index_order, OutputLastUsageMap *out_last_usage) {
  (void)index_order;
  if (out_last_usage == nullptr) {
    return;
  }
  const bool use_memory_block_csv = !g_name_to_sizes.empty();
  const OutputLastUsageMap base_last = *out_last_usage;
  const HorizonSearchContext horizon_context{user_info_list, node_to_index, base_last};
  for (const auto &info : user_info_list) {
    const auto &prod_ki = info.node;
    const AnfNodePtr &producer = prod_ki.first;
    if (producer == nullptr) {
      continue;
    }
    auto p_exec_it = node_to_index.find(producer);
    if (p_exec_it == node_to_index.end()) {
      continue;
    }
    if (use_memory_block_csv && !ProducerOutputListedInCsvNColumn(producer, prod_ki.second)) {
      continue;
    }
    auto pit = out_last_usage->find(prod_ki);
    const size_t current_last_usage = (pit != out_last_usage->end()) ? pit->second : p_exec_it->second;
    size_t extended_last_usage = current_last_usage;
    std::unordered_map<AnfNodePtr, size_t> memo;
    for (const auto &user_ki : info.users) {
      const AnfNodePtr &user_node = user_ki.first;
      if (user_node == nullptr) {
        continue;
      }
      auto u_exec_it = node_to_index.find(user_node);
      if (u_exec_it == node_to_index.end()) {
        continue;
      }
      const size_t u_exec = u_exec_it->second;
      if (!use_memory_block_csv) {
        if (IsViewOrInplaceNode(user_node)) {
          std::unordered_set<AnfNodePtr> rec_stack;
          const HorizonSearchState horizon_state{&memo, &rec_stack};
          extended_last_usage =
            std::max(extended_last_usage, HorizonFromAliasConsumer(user_node, horizon_context, horizon_state));
        }
        continue;
      }
      if (NodeNameListedInCsvNColumn(user_node)) {
        extended_last_usage = std::max(extended_last_usage, u_exec);
      } else {
        std::unordered_set<AnfNodePtr> rec_stack;
        const HorizonSearchState horizon_state{&memo, &rec_stack};
        extended_last_usage =
          std::max(extended_last_usage, HorizonFromMissingTableConsumer(user_node, horizon_context, horizon_state));
      }
    }
    (*out_last_usage)[prod_ki] = extended_last_usage;
  }
}

// Candidate/profile size comes from CSV when available, otherwise from static tensor-size estimation.
size_t GetNodeSizeForProfile(const AnfNodePtr &node, size_t out_idx, size_t exec_order_index) {
  if (UseMemoryBlockCsvModeling()) {
    return LookupOperatorMemorySize(node, exec_order_index, out_idx);
  }
  return EstimateNodeMemorySize(node, out_idx);
}

void BuildNodeFirstLastMap(const CNodePtrList &execution_order, NodeFirstLastMap *node_first_last) {
  node_first_last->clear();
  for (size_t i = 0; i < execution_order.size(); ++i) {
    const auto &node = execution_order[i];
    if (node == nullptr) {
      continue;
    }
    auto it = node_first_last->find(node);
    if (it == node_first_last->end()) {
      (*node_first_last)[node] = {i, i};
    } else {
      it->second.second = i;
    }
  }
}

// Build a baseline-plus-delta memory profile for each execution-order slot.
// Modeling is performed per `(kernel, output_index)` rather than per whole operator.
void BuildProfileBaselineAndDiff(const CNodePtrList &index_order, size_t order_size,
                                 const NodeFirstLastMap &node_first_last, const OutputLastUsageMap &out_last_usage,
                                 size_t baseline_bytes, std::vector<size_t> *memory_profile) {
  MS_EXCEPTION_IF_NULL(memory_profile);
  memory_profile->clear();
  if (order_size == 0) {
    return;
  }
  const size_t profile_size = order_size + kProfilePaddingSlots;
  std::vector<int64_t> delta(profile_size, 0);
  for (size_t i = 0; i < order_size; ++i) {
    const auto &kernel = index_order[i];
    if (kernel == nullptr) {
      continue;
    }
    const size_t workspace_size =
      UseMemoryBlockCsvModeling() ? LookupOperatorWorkspacePulseFromCsv(kernel) : EstimateNodeWorkspaceSize(kernel);
    if (workspace_size > 0 && i < profile_size) {
      delta[i] += static_cast<int64_t>(workspace_size);
      const size_t workspace_drop_index = i + 1;
      if (workspace_drop_index < profile_size) {
        delta[workspace_drop_index] -= static_cast<int64_t>(workspace_size);
      }
    }
    const size_t output_count = TableOutputCount(kernel);
    for (size_t out_idx = 0; out_idx < output_count; ++out_idx) {
      const size_t raw_tensor_size = GetNodeSizeForProfile(kernel, out_idx, i);
      if (raw_tensor_size == 0) {
        continue;
      }
      const size_t aligned_tensor_size = raw_tensor_size;
      auto fl_it = node_first_last.find(kernel);
      const size_t creation = (fl_it != node_first_last.end()) ? fl_it->second.first : i;
      auto ol_it = out_last_usage.find({kernel, out_idx});
      const size_t last = (ol_it != out_last_usage.end()) ? ol_it->second : creation;
      if (last < creation) {
        continue;
      }
      if (creation < profile_size) {
        delta[creation] += static_cast<int64_t>(aligned_tensor_size);
      }
      const size_t drop_at = last + 1;
      if (drop_at < profile_size) {
        delta[drop_at] -= static_cast<int64_t>(aligned_tensor_size);
      }
    }
  }
  int64_t run = static_cast<int64_t>(baseline_bytes);
  memory_profile->resize(profile_size);
  for (size_t k = 0; k < profile_size; ++k) {
    run += delta[k];
    (*memory_profile)[k] = run > 0 ? static_cast<size_t>(run) : 0;
  }
}

void CollectPeakEstimateCandidates(const UserInfoList &user_info_list,
                                   const std::map<AnfNodePtr, size_t> &node_to_index,
                                   const NodeFirstLastMap &node_first_last, const OutputLastUsageMap &out_last_usage,
                                   std::vector<PeakEstimateCandidate> *all_candidates) {
  for (const auto &info : user_info_list) {
    if (!IsValidOffloadNode(info)) {
      continue;
    }
    auto node = info.node.first;
    auto n_it = node_to_index.find(node);
    if (n_it == node_to_index.end()) {
      continue;
    }
    auto fl_it = node_first_last.find(node);
    const size_t creation_idx = (fl_it != node_first_last.end()) ? fl_it->second.first : n_it->second;
    auto out_it = out_last_usage.find({node, info.node.second});
    const size_t last_usage_idx = (out_it != out_last_usage.end()) ? out_it->second : creation_idx;
    if (last_usage_idx <= creation_idx) {
      continue;
    }
    size_t size = GetNodeSizeForProfile(node, info.node.second, creation_idx);
    if (size == 0) {
      continue;
    }
    all_candidates->push_back({info, size, creation_idx, last_usage_idx});
  }
}

// Load the CSV table once and derive the baseline used by profile construction.
static bool LoadMemoryBlockTableForModeling(const CNodePtrList &index_order) {
  const std::string path = GetMemoryBlockCsvPath();
  if (path.empty() || !LoadMemoryBlockCsv(path, index_order)) {
    g_name_to_sizes.clear();
    g_name_to_g_column_bytes.clear();
    g_name_to_workspace_pulse_bytes.clear();
    g_csv_baseline_bytes = 0;
    MS_LOG(INFO) << "[HyperOffload] memory_block.csv is unavailable, falling back to static tensor-size estimation. "
                 << "path=" << path;
    return false;
  }
  if (ApplyProfileBaselineFromFirstExecOrderTableHit(index_order)) {
    return true;
  }
  g_name_to_sizes.clear();
  g_name_to_g_column_bytes.clear();
  g_name_to_workspace_pulse_bytes.clear();
  g_csv_baseline_bytes = 0;
  MS_LOG(INFO) << "[HyperOffload] memory_block.csv could not be bound to execution-order outputs, "
               << "falling back to static tensor-size estimation.";
  return false;
}

// Build the memory profile without changing the execution order.
// CSV modeling keeps the original table-based size/baseline path. Fallback modeling switches
// to static tensor-size estimation and uses graph outputs/summary nodes to extend last-use.
void BuildMemoryProfile(const UserInfoList &user_info_list, const std::map<AnfNodePtr, size_t> &node_to_index,
                        size_t order_size, size_t parameter_memory_baseline, const CNodePtrList &index_order,
                        const FuncGraphPtr &graph, std::vector<size_t> *memory_profile,
                        NodeFirstLastMap *node_first_last, OutputLastUsageMap *out_last_usage) {
  MS_EXCEPTION_IF_NULL(out_last_usage);
  const bool use_memory_block_csv = LoadMemoryBlockTableForModeling(index_order);
  BuildNodeFirstLastMap(index_order, node_first_last);

  if (use_memory_block_csv) {
    BuildOutputLastUsageMap(user_info_list, node_to_index, index_order, out_last_usage);
  } else {
    out_last_usage->clear();
    FillLastUsageFromUserInfo(user_info_list, node_to_index, out_last_usage);
    const size_t graph_end_index = order_size > 0 ? order_size - 1 : 0;
    FillLastUsageFromGraphOutputs(graph, node_to_index, graph_end_index, out_last_usage);
    FillLastUsageFromSummaryNodes(graph, node_to_index, graph_end_index, out_last_usage);
    ExtendOutputLastUsageForBufferAliasing(user_info_list, node_to_index, index_order, out_last_usage);
  }

  const size_t baseline_bytes = use_memory_block_csv ? g_csv_baseline_bytes : parameter_memory_baseline;
  BuildProfileBaselineAndDiff(index_order, order_size, *node_first_last, *out_last_usage, baseline_bytes,
                              memory_profile);
}

}  // namespace

// Before offload: phase 1, build profile + candidates. The implementation prefers CSV modeling
// when available and otherwise falls back to static size estimation. Peak strategy loop uses only
// memory_profile deltas and PeakEstimateCandidate.size after this point.
// Invariant: memory analysis does not modify execution order (index_order is read-only).
void BuildMemoryProfileAndCandidates(const UserInfoList &user_info_list,
                                     const std::map<AnfNodePtr, size_t> &node_to_index, size_t order_size,
                                     size_t parameter_memory_baseline, std::vector<size_t> *memory_profile,
                                     std::vector<PeakEstimateCandidate> *all_candidates,
                                     const CNodePtrList &index_order, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(memory_profile);
  MS_EXCEPTION_IF_NULL(all_candidates);
  NodeFirstLastMap node_first_last;
  OutputLastUsageMap out_last_usage;
  BuildMemoryProfile(user_info_list, node_to_index, order_size, parameter_memory_baseline, index_order, graph,
                     memory_profile, &node_first_last, &out_last_usage);

  all_candidates->clear();
  CollectPeakEstimateCandidates(user_info_list, node_to_index, node_first_last, out_last_usage, all_candidates);
}

// Subtract candidate tensor size over lifecycle [start, end) on modeled baseline+diff profile.
void ReduceMemoryProfile(size_t start_index, size_t end_index, size_t size, std::vector<size_t> *memory_profile) {
  MS_EXCEPTION_IF_NULL(memory_profile);
  if (start_index >= end_index) {
    return;
  }
  for (size_t i = start_index; i < end_index; ++i) {
    if (i < memory_profile->size()) {
      if ((*memory_profile)[i] >= size) {
        (*memory_profile)[i] -= size;
      } else {
        (*memory_profile)[i] = 0;
      }
    }
  }
}

size_t ComputeTotalParameterMemory(const FuncGraphPtr &graph) {
  if (graph == nullptr) {
    return 0;
  }
  return std::accumulate(graph->parameters().begin(), graph->parameters().end(), size_t{0},
                         [](size_t total_bytes, const AnfNodePtr &parameter) {
                           if (parameter == nullptr || !parameter->isa<Parameter>()) {
                             return total_bytes;
                           }
                           return total_bytes + EstimateNodeMemorySize(parameter, 0);
                         });
}

size_t GetNodeWorkspacePulseForProfile(const AnfNodePtr &node) {
  if (node == nullptr) {
    return 0;
  }
  return UseMemoryBlockCsvModeling() ? LookupOperatorWorkspacePulseFromCsv(node) : EstimateNodeWorkspaceSize(node);
}

}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
