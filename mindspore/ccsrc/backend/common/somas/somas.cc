/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "backend/common/somas/somas.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <random>

#include "backend/common/somas/somas_node.h"
#include "backend/common/somas/somas_solver_pre.h"
#include "backend/common/somas/somas_stream.h"
#include "backend/common/somas/somas_tensor.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/string_recorder.h"
#endif
#include "include/common/thread_pool.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"

using mindspore::profiler::ascend::MemoryProfiling;
using mindspore::profiler::ascend::NodeMemory;
using mindspore::profiler::ascend::TensorMemory;
#endif
namespace mindspore {
namespace somas {
constexpr auto kRetryIntervalSeconds = 500;
constexpr auto kRefNodeTensorNum = 2;
constexpr auto kOnlyOneDestinationNode = 1;
constexpr auto kOnlyTwoDestinationNode = 2;
constexpr auto kNopNodeRealInputIndex = 1;
constexpr auto kZeroAlignSize = 1;

constexpr auto kGraphId = "graph_id";
constexpr auto kHashId = "hash_id";
constexpr auto kReused_memory_size = "reused_memory_size";
constexpr auto kNodeSize = "node_size";
constexpr auto kTensorSize = "tensor_size";
constexpr auto kContiguousSize = "contiguous_size";
constexpr auto kRefNodeSize = "ref_node_size";
constexpr auto kStreamSize = "stream_size";
constexpr auto kStreamGroupSize = "stream_group_size";
constexpr auto kTensors = "tensors";

constexpr auto kTensorId = "tensor_id";
constexpr auto kSize = "size";
constexpr auto kOriSize = "ori_size";
constexpr auto kLifelongValue = "lifelong_value";
constexpr auto kLifeStart = "life_start";
constexpr auto kLifeEnd = "life_end";
constexpr auto kOffset = "offset";
constexpr auto kCachedResultThreshold = 2000;
constexpr size_t kLogMergedBlockSize = 10;

// set somas result
void SetSomasResult(std::vector<std::pair<size_t, size_t>> &&output_somas_result,
                    std::vector<std::pair<size_t, size_t>> &&workspace_somas_result, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (!kernel_info->SetSomasResult(std::move(output_somas_result), std::move(workspace_somas_result))) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << "set somas result fail. ";
  }
}

void MergeBlocks(std::vector<Block> *block_list, std::stack<Block> *merged_blocks) {
  if (block_list->empty()) {
    MS_LOG(INFO) << "No block to merge.";
    return;
  }
  std::sort(block_list->begin(), block_list->end(), [](const Block &block1, const Block &block2) {
    return (block1.start_offset_ < block2.start_offset_) ||
           ((block1.start_offset_ == block2.start_offset_) && (block1.end_offset_ < block2.end_offset_));
  });
  merged_blocks->emplace((*block_list)[0].start_offset_, (*block_list)[0].size_);
  for (size_t i = 1; i < block_list->size(); i++) {
    Block &top = merged_blocks->top();
    auto &block = (*block_list)[i];
    if (block.start_offset_ >= top.end_offset_) {
      merged_blocks->emplace(block.start_offset_, block.size_);
    } else if (block.end_offset_ > top.end_offset_) {
      top.end_offset_ = block.end_offset_;
      top.size_ = top.end_offset_ - top.start_offset_;
    }
  }
}

bool Somas::IsSupportSomas(const session::KernelGraph &graph) {
  if (graph.is_dynamic_shape()) {
    MS_LOG(WARNING) << "Somas can't allocate graph with dynamic shape now.";
    return false;
  }

  auto &execution_order = graph.execution_order();
  for (auto &kernel : execution_order) {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel);
    if ((kernel_name == kRpcSendOpName) || (kernel_name == kRpcRecvOpName)) {
      MS_LOG(WARNING) << "Somas can't allocate graph with rpc op now.";
      return false;
    }
  }

  return true;
}

bool Somas::Assign(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "Start Somas Assign for graph " << graph.graph_id();
  if (!IsSupportSomas(graph)) {
    return false;
  }
  auto ret = ConfigSomas(graph);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Config Somas Failed.";
  }
  MS_LOG(INFO) << "Somas Configure success, configuration info: "
               << "\nDevice Name: " << device_name_ << "\nRun by execution order: " << depend_exec_order_
               << "\nEnable debug log: " << save_debug_info_ << "\nDebug log path: " << debug_info_path_;
  MS_LOG(INFO) << "Start Initialize SOMAS Model";

  InitSomasModel(graph);
  MS_LOG(INFO) << "End Initialize SOMAS Model";

  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Somas Tensor in graph " << graph.graph_id();
    return true;
  }

  if (enable_cache_) {
    ret = LoadSomasCache(graph);
    if (ret) {
      GenGraphStatisticInfo();
      UpdateSomasResultToGraph(graph);
      DumpSomasModelInfo("somas_tensor_offset", graph.graph_id());
      MS_LOG(INFO) << "Somas Allocate end.";
      return true;
    }
  }

  // Computing Conflict pairs
  MS_LOG(INFO) << "Start Computing Conflict Matrix";
  ComputeConflictMatrix();
  MS_LOG(INFO) << "End Computing Conflict Matrix";

  Solve(graph);

  if (enable_cache_) {
    SaveSomasResult(graph);
  }

  UpdateSomasResultToGraph(graph);
  DumpSomasModelInfo("somas_tensor_offset", graph.graph_id());

  MS_LOG(INFO) << "Somas Allocate end.";
  return true;
}

bool Somas::Assign(const KernelGraphPtr &graph_ptr) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(graph_ptr);
#ifndef ENABLE_SECURITY
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "somas_input_graph_" + std::to_string(graph_ptr->graph_id()) + ".ir";
    DumpIR(file_name, graph_ptr, true, kWholeStack);
  }
#endif
  return Assign(*graph_ptr);
}

size_t Somas::GetCommunicationReservedSize() const { return 0; }

void Somas::CommunicationTensorProcess(const std::vector<SomasTensorPtr> &tensors) const {}

bool Somas::GetEnableCacheFlag(const session::KernelGraph &graph) const {
  return graph.execution_order().size() >= kCachedResultThreshold;
}

std::pair<bool, std::string> Somas::GetDebugConfig() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto enable_save_graphs = context_ptr->CanDump(kIntroductory);
  auto save_graphs_path = context_ptr->GetSaveGraphsPath();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  return std::make_pair(enable_save_graphs, save_graphs_path);
}

std::vector<vector<uint32_t>> Somas::GetStreamGroupInfo() const {
  std::vector<vector<uint32_t>> stream_group;
  return stream_group;
}

std::map<std::string, UnReuseType> Somas::GetUnReuseNodeType() const {
  std::map<std::string, UnReuseType> node_type;
  return node_type;
}

std::map<std::string, UnReuseType> Somas::GetUnReuseNodeName() const {
  std::map<std::string, UnReuseType> name_type;
  return name_type;
}

bool Somas::ConfigSomas(const session::KernelGraph &graph) {
  auto ret = Initialize();
  if (!ret) {
    MS_LOG(ERROR) << "Somas Initialize failed. Please Check!!!";
    return false;
  }
  device_name_ = GetDeviceName();
  communication_gap_size_ = GetCommunicationReservedSize();
  enable_cache_ = GetEnableCacheFlag(graph);
  depend_exec_order_ = GetDependExecOrderFlag(graph);
  auto debug_config = GetDebugConfig();
  save_debug_info_ = debug_config.first;
  debug_info_path_ = debug_config.second;
  streams_groups_ = GetStreamGroupInfo();
  un_reuse_node_type_.clear();
  auto device_un_reuse_type = GetUnReuseNodeType();
  un_reuse_node_type_.insert(device_un_reuse_type.begin(), device_un_reuse_type.end());
  un_reuse_node_name_.clear();
  auto device_un_reuse_name = GetUnReuseNodeName();
  un_reuse_node_name_.insert(device_un_reuse_name.begin(), device_un_reuse_name.end());
  return true;
}

bool Somas::LoadSomasCache(const session::KernelGraph &graph) {
  MS_LOG(DEBUG) << "Somas LoadSomasCache start...";
  bool ret = CalcSomasModelHash(graph);
  if (ret) {
    std::string filename = Common::GetCompilerCachePath() + "/somas_meta/somas_graph_" +
                           std::to_string(graph.graph_id()) + "_" + hash_id_ + ".json";
    ret = LoadSomasResult(filename);
    if (ret) {
      MS_LOG(INFO) << "Load Somas Cache file " << filename << " Successfully.";
    }
  } else {
    MS_LOG(ERROR) << "Calculate SOMAS model hash id failed.";
  }
  MS_LOG(DEBUG) << "Somas LoadSomasCache end.";
  return ret;
}

bool Somas::CalcSomasModelHash(const session::KernelGraph &graph) {
  auto model_str = SomasInfo(true);
  hash_id_ = std::to_string(std::hash<std::string>()(model_str));
  MS_LOG(INFO) << "Graph " << graph.graph_id() << "'s SOMAS Model hash id is " << hash_id_;
  std::string filename = Common::GetCompilerCachePath() + "/somas_meta/somas_graph_" +
                         std::to_string(graph.graph_id()) + "_" + hash_id_ + ".info";
  return Common::SaveStringToFile(filename, model_str);
}

void Somas::SaveSomasResult(const session::KernelGraph &graph) {
  nlohmann::json somas_json;
  somas_json[kGraphId] = graph.graph_id();
  somas_json[kHashId] = hash_id_;
  somas_json[kReused_memory_size] = reused_memory_size_;
  somas_json[kNodeSize] = nodes_list_.size();
  somas_json[kTensorSize] = tensors_list_.size();
  somas_json[kContiguousSize] = contiguous_tensors_list_.size();
  somas_json[kRefNodeSize] = union_tensors_list_.size();
  somas_json[kStreamSize] = streams_map_.size();
  somas_json[kStreamGroupSize] = streams_groups_.size();
  std::vector<nlohmann::json> tensors_json;
  for (auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    nlohmann::json tensor_json;
    tensor_json[kTensorId] = tensor->GetId();
    tensor_json[kSize] = tensor->GetAlignedSize();
    tensor_json[kOriSize] = tensor->GetOriginalSize();
    tensor_json[kLifelongValue] = tensor->lifelong_value_;
    tensor_json[kLifeStart] = tensor->lifetime_.start_;
    tensor_json[kLifeEnd] = tensor->lifetime_.end_;
    tensor_json[kOffset] = tensor->GetOffset();
    tensors_json.emplace_back(tensor_json);
  }
  somas_json[kTensors] = tensors_json;

  std::string filename = Common::GetCompilerCachePath() + "/somas_meta/somas_graph_" +
                         std::to_string(graph.graph_id()) + "_" + hash_id_ + ".json";
  (void)Common::SaveStringToFile(filename, somas_json.dump());
}

void Somas::UpdateSomasResultToGraph(const session::KernelGraph &graph) {
  auto &execution_nodes = graph.execution_order();
  std::vector<Block> block_list;
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->GetAlignedSize() > 0) {
      block_list.emplace_back(tensor->GetOffset(), tensor->GetAlignedSize());
    }
  }

  // Contiguous gaps postprocessing
  for (auto list : contiguous_tensors_list_) {
    if (list.empty()) {
      MS_LOG(EXCEPTION) << "list is empty.";
    }
    MS_EXCEPTION_IF_NULL(tensors_list_[list[0]]);
    size_t offset = tensors_list_[list[0]]->offset_;
    size_t all_size = 0;
    for (auto i : list) {
      if (i >= tensors_list_.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << "is out of range " << tensors_list_.size();
      }
      MS_EXCEPTION_IF_NULL(tensors_list_[i]);
      offset = std::min(offset, tensors_list_[i]->offset_);
      all_size += tensors_list_[i]->aligned_size_;
    }
    if (all_size > 0) {
      block_list.emplace_back(offset, all_size);
    }
    // for allocator
    tensors_list_[list[0]]->offset_ += communication_gap_size_;
  }

  for (auto &node : execution_nodes) {
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_somas_result = GetNodeOutputSomasResult(node);
    auto workspace_somas_result = GetNodeWorkSpaceSomasResult(node);
    SetSomasResult(std::move(output_somas_result), std::move(workspace_somas_result), node.get());
  }

  std::stack<Block> merged_blocks;
  MergeBlocks(&block_list, &merged_blocks);
  session::SomasInfo *somas_info = graph.MutableSomasInfo();
  somas_info->whole_block_size_ = reused_memory_size_;
  std::vector<std::pair<size_t, size_t>> log_merged_blocks;
  size_t all_block_size = 0;
  while (!merged_blocks.empty()) {
    auto block = merged_blocks.top();
    merged_blocks.pop();
    somas_info->merged_blocks_map_[block.start_offset_] = block.size_;
    dump_merged_blocks_.emplace_back(block.start_offset_, block.size_);
    log_merged_blocks.emplace_back(block.start_offset_, block.size_);
    all_block_size = std::max(block.start_offset_ + block.size_, all_block_size);
  }
  std::sort(log_merged_blocks.begin(), log_merged_blocks.end(),
            [](const std::pair<size_t, size_t> &A, const std::pair<size_t, size_t> &B) { return A.second > B.second; });
  MS_LOG(INFO) << "Merged Block size: " << log_merged_blocks.size();
  for (size_t i = 0; i < std::min(kLogMergedBlockSize, log_merged_blocks.size()); i++) {
    MS_LOG(INFO) << "Merged Block: " << i << ", offset: " << log_merged_blocks[i].first
                 << ", size: " << log_merged_blocks[i].second;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(all_block_size == reused_memory_size_,
                             "All block size and Reused memory size not equal: " + std::to_string(all_block_size) +
                               ", " + std::to_string(reused_memory_size_));
}

bool Somas::LoadSomasResult(const string &filename) {
  std::ifstream somas_json_fs(filename);
  if (!somas_json_fs.is_open()) {
    MS_LOG(INFO) << "Open json file: " << filename << " error, Somas Cache Missed.";
    return false;
  }
  nlohmann::json somas_json;
  try {
    somas_json_fs >> somas_json;
    somas_json_fs.close();
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Parse json file error: " << filename << ", sleep 500ms and retry again.";
    somas_json_fs.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalSeconds));
    std::ifstream retry_tmp(filename);
    if (!retry_tmp.is_open()) {
      MS_LOG(INFO) << "Open json file: " << filename << " error, please check kernel_meta.";
      return false;
    }
    retry_tmp >> somas_json;
    retry_tmp.close();
  }

  auto ret = VerifySomasResult(somas_json);
  if (!ret) {
    MS_LOG(WARNING) << "Verify Somas Result Failed.";
    return false;
  }
  reused_memory_size_ = somas_json[kReused_memory_size];
  ret = UpdateTensorsOffset(somas_json[kTensors]);
  return ret;
}

bool Somas::VerifySomasResult(const nlohmann::json &somas_json) const {
  const auto &hash_id = somas_json[kHashId];
  const auto &node_size = somas_json[kNodeSize];
  const auto &tensor_size = somas_json[kTensorSize];
  const auto &contiguous_size = somas_json[kContiguousSize];
  const auto &ref_node_size = somas_json[kRefNodeSize];
  const auto &stream_size = somas_json[kStreamSize];
  const auto &stream_group_size = somas_json[kStreamGroupSize];

  if (hash_id != hash_id_) {
    MS_LOG(WARNING) << "Mismatch hash id " << hash_id << " vs " << hash_id_;
    return false;
  }

  if (node_size != nodes_list_.size()) {
    MS_LOG(WARNING) << "Mismatch node size " << node_size << " vs " << nodes_list_.size();
    return false;
  }

  if (tensor_size != tensors_list_.size()) {
    MS_LOG(WARNING) << "Mismatch tensor size " << tensor_size << " vs " << tensors_list_.size();
    return false;
  }

  if (contiguous_size != contiguous_tensors_list_.size()) {
    MS_LOG(WARNING) << "Mismatch contiguous size " << contiguous_size << " vs " << contiguous_tensors_list_.size();
    return false;
  }

  if (ref_node_size != union_tensors_list_.size()) {
    MS_LOG(WARNING) << "Mismatch ref node size " << ref_node_size << " vs " << union_tensors_list_.size();
    return false;
  }

  if (stream_size != streams_map_.size()) {
    MS_LOG(WARNING) << "Mismatch stream size " << stream_size << " vs " << streams_map_.size();
    return false;
  }

  if (stream_group_size != streams_groups_.size()) {
    MS_LOG(WARNING) << "Mismatch stream group size " << stream_group_size << " vs " << streams_groups_.size();
    return false;
  }

  const auto &tensors_json = somas_json[kTensors];
  for (const auto &tensor_json : tensors_json) {
    const auto &tensor_id = tensor_json[kTensorId];
    const auto &size = tensor_json[kSize];
    const auto &ori_size = tensor_json[kOriSize];
    const auto &lifelong_value = tensor_json[kLifelongValue];
    const auto &life_start = tensor_json[kLifeStart];
    const auto &life_end = tensor_json[kLifeEnd];
    if (tensor_id < tensors_list_.size()) {
      auto &tensor = tensors_list_[tensor_id];
      MS_EXCEPTION_IF_NULL(tensor);
      if (size != tensor->aligned_size_) {
        MS_LOG(WARNING) << "Mismatch size of tensor " << tensor_id << " " << size << " vs " << tensor->aligned_size_;
        return false;
      }

      if (ori_size != tensor->GetOriginalSize()) {
        MS_LOG(WARNING) << "Mismatch original size of tensor " << tensor_id << " " << ori_size << " vs "
                        << tensor->GetOriginalSize();
        return false;
      }

      if (lifelong_value != tensor->lifelong_value_) {
        MS_LOG(WARNING) << "Mismatch lifelong value of tensor " << tensor_id << " " << lifelong_value << " vs "
                        << tensor->lifelong_value_;
        return false;
      }

      if (life_start != tensor->lifetime_.start_) {
        MS_LOG(WARNING) << "Mismatch life start of tensor " << tensor_id << " " << life_start << " vs "
                        << tensor->lifetime_.start_;
        return false;
      }

      if (life_end != tensor->lifetime_.end_) {
        MS_LOG(WARNING) << "Mismatch life start of tensor " << tensor_id << " " << life_end << " vs "
                        << tensor->lifetime_.end_;
        return false;
      }
    } else {
      MS_LOG(WARNING) << "Can't find tensor " << tensor_id;
      return false;
    }
  }

  return true;
}

bool Somas::UpdateTensorsOffset(const std::vector<nlohmann::json> &tensors_json) {
  bool ret = true;
  for (auto &tensor_json : tensors_json) {
    const auto &tensor_id = tensor_json[kTensorId];
    const auto &size = tensor_json[kSize];
    const auto &offset = tensor_json[kOffset];
    auto &tensor = tensors_list_[tensor_id];
    MS_EXCEPTION_IF_NULL(tensor);
    // update memory offset
    tensor->offset_ = offset;
    tensor->aligned_size_ = size;
  }
  return ret;
}

void Somas::InitSomasModel(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_CHECK_FAIL(InitBasicInfoFromGraph(graph), "Init SOMAS basic info from graph failed.");
#if defined(ENABLE_DUMP_IR) && !defined(ENABLE_SECURITY)
  SubModuleId module = SubModuleId::SM_OPTIMIZER;
  std::string name = device_name_ + "_somas_initial_info." + std::to_string(graph.graph_id());
  (void)mindspore::RDR::RecordString(module, name, SomasInfo());
#endif
  DumpSomasModelInfo("somas_initial_info", graph.graph_id());

  MS_EXCEPTION_IF_CHECK_FAIL(InitDevSpecControlTensors(graph), "Init device special control tensors failed.");
  DumpSomasModelInfo("somas_device_control_info", graph.graph_id());

  MS_EXCEPTION_IF_CHECK_FAIL(CommonSpecNodeProcess(graph), "Common special node process failed.");
  DumpSomasModelInfo("somas_common_spec_node_process", graph.graph_id());

  MS_EXCEPTION_IF_CHECK_FAIL(DevSpecNodeProcess(graph), "Device specify special node process failed.");
  DumpSomasModelInfo("somas_device_spec_node_process", graph.graph_id());

  UnReuseNodeProcess(graph);
  UpdateContiguousTensorList();
  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor from graph " << graph.graph_id();
    return;
  }

  MS_LOG(INFO) << "Created " << streams_map_.size() << " streams (" << streams_groups_.size() << " groups), "
               << nodes_list_.size() << " nodes, " << tensors_list_.size() << " tensors, " << union_tensors_list_.size()
               << " union tensors lists, and " << contiguous_tensors_list_.size() << " contiguous tensors lists";

#if defined(ENABLE_DUMP_IR) && !defined(ENABLE_SECURITY)
  name = device_name_ + "_somas_pre_processed_info." + std::to_string(graph.graph_id());
  (void)mindspore::RDR::RecordString(module, name, SomasInfo());
  name = device_name_ + "_somas_offline_log." + std::to_string(graph.graph_id());
  (void)mindspore::RDR::RecordString(module, name, Offline());
#endif

  DumpSomasModelInfo("somas_pre_processed_info", graph.graph_id());
  if (save_debug_info_) {
    std::string offline_file_path = GetSaveGraphsPathName(
      "/" + device_name_ + "_somas_offline_log_" + std::to_string(graph.graph_id()) + ".ir", debug_info_path_);
    DumpOfflineIR(offline_file_path);
  }
}

void Somas::AddControlTensor(const SomasNodePtr &from, const SomasNodePtr &to) {
  size_t control_tensor_index = control_tensors_list_.size();
  SomasTensorPtr tensor =
    std::make_shared<SomasTensor>(control_tensor_index, from->GetId(), from->GetStreamId(), 0, 0, kLifeLongNone);
  tensor->lifetime_.start_ = from->GetId();
  tensor->lifetime_.end_ = to->GetId();
  tensor->type_ = kControl;
  tensor->destination_nodes_.insert(to->GetId());
  tensor->consumer_list_.emplace_back(to->GetId());
  from->control_output_tensors_.push_back(tensor);
  to->control_input_tensors_.push_back(tensor);
  to->ancestor_nodes_.insert(from);
  control_tensors_list_.push_back(tensor);
}

void Somas::AddControlTensorFromExecOrder() {
  // Loop to add control edges within each stream (node order within stream)
  for (const auto &stream_kv : streams_map_) {
    auto stream = stream_kv.second;
    MS_EXCEPTION_IF_NULL(stream);
    auto &nodes = stream->nodes_;
    std::sort(nodes.begin(), nodes.end(), NodeSort);
    for (size_t i = 1; i < nodes.size(); i++) {
      const auto &previous_node = nodes[i - 1];
      const auto &current_node = nodes[i];
      MS_EXCEPTION_IF_NULL(current_node);
      AddControlTensor(previous_node, current_node);
    }
  }

  // Loop to add control edges from end to beginning of next group
  for (const auto &group : streams_groups_) {
    for (size_t i = 1; i < group.size(); i++) {
      size_t previous_stream = group[i - 1];
      size_t current_stream = group[i];

      auto stream = GetSomasStream(previous_stream);
      if (stream == nullptr) {
        continue;
      }

      auto &last_node_in_prev_stream = stream->nodes_.back();

      stream = GetSomasStream(current_stream);
      if (stream == nullptr) {
        continue;
      }
      auto &first_node_in_cur_stream = stream->nodes_.front();
      AddControlTensor(last_node_in_prev_stream, first_node_in_cur_stream);
    }
  }

  // Loop to compute max destinations in each stream
  mindspore::HashMap<size_t, size_t> stream_max_destination_node;
  // Loop to compute max destinations in each stream
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    stream_max_destination_node.clear();
    for (const auto &node_id : tensor->destination_nodes_) {
      auto node = GetSomasNode(node_id);
      MS_EXCEPTION_IF_NULL(node);
      if (node_id > stream_max_destination_node[node->GetStreamId()]) {
        stream_max_destination_node[node->GetStreamId()] = node_id;
      }
    }

    tensor->consumer_list_.clear();
    for (const auto &dst_map : stream_max_destination_node) {
      tensor->consumer_list_.emplace_back(dst_map.second);
    }
  }
}

void Somas::InitControlTensors() {
  if (depend_exec_order_) {
    AddControlTensorFromExecOrder();
  }
}

bool Somas::CommonSpecNodeProcess(const session::KernelGraph &graph) {
#ifndef ENABLE_SECURITY
  SummaryInputProcess(graph);
#endif
  RefNodeProcess(graph);
  CommunicationNodeProcess();
  return true;
}

void Somas::InitSomasStreamAndNode(const session::KernelGraph &graph) {
  MS_LOG(DEBUG) << "Somas InitSomasStreamAndNode start...";
  streams_map_.clear();
  nodes_list_ = {};
  SomasNodePtr last_node = nullptr;
  size_t last_repeat_node_size = 0;
  auto &kernel_cnodes = (graph.subgraph_multi_call()) ? graph.mem_reuse_exec_order() : graph.execution_order();
  for (size_t i = 0; i < kernel_cnodes.size(); i++) {
    auto kernel = kernel_cnodes[i];
    MS_EXCEPTION_IF_NULL(kernel);
    SomasStreamPtr stream;
    size_t stream_id = i;
    if (depend_exec_order_) {
      stream_id = AnfAlgo::GetStreamId(kernel);
    }
    auto it = streams_map_.find(stream_id);
    if (it == streams_map_.end()) {
      stream = std::make_shared<SomasStream>(stream_id);
      streams_map_[stream_id] = stream;
    } else {
      stream = (*it).second;
    }

    // Node
    NodeType type = kCommonNode;
    if (common::AnfAlgo::IsCommunicationOp(kernel)) {
      type = kCommunicationNode;
    }
    MS_EXCEPTION_IF_NULL(stream);
    auto node = std::make_shared<SomasNode>(kernel->fullname_with_scope(), i, type, stream->GetId());
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_CHECK_FAIL(nodes_list_.size() == i, "node_list_ size error!!!");
    nodes_list_.push_back(node);
    auto key = kernel.get();
    auto &nodes = nodes_map_[key];
    if (nodes.empty()) {
      stream->nodes_.push_back(node);
    }
    nodes.push_back(node);
    if (last_node != nullptr && (last_repeat_node_size > 1 || nodes.size() > 1) && depend_exec_order_) {
      AddControlTensor(last_node, node);
    }
    last_node = node;
    last_repeat_node_size = nodes.size();
  }
}

void Somas::InitSomasOutputAndWorkspaceTensors(const session::KernelGraph &graph) {
  MS_LOG(DEBUG) << "Somas InitSomasOutputAndWorkspaceTensors start...";
  tensors_list_ = {};
  size_t tensor_index = 0;
  auto &kernel_cnodes = graph.execution_order();
  for (const auto &kernel : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto nodes = nodes_map_[kernel.get()];
    auto node = nodes[0];
    MS_EXCEPTION_IF_NULL(node);
    auto stream_id = node->GetStreamId();

    // Output Tensor
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (const auto &size : output_sizes) {
      auto output_tensor_index = tensor_index;
      tensor_index++;
      size_t aligned_size = GetAlignSize(size);
      if (aligned_size == 0) {
        // Device Address still need to be allocated when output_size is 0
        aligned_size = GetAlignSize(kZeroAlignSize);
        MS_LOG(INFO) << "Node output size is zero: " << kernel->fullname_with_scope() << " output size " << size
                     << " align size " << aligned_size;
      }
      auto tensor =
        std::make_shared<SomasTensor>(output_tensor_index, node->GetId(), stream_id, size, aligned_size, kLifeLongNone);
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->lifetime_.start_ = node->GetId();
      tensor->lifetime_.end_ = (nodes.size() > 1) ? nodes.back()->GetId() : node->GetId();
      tensor->type_ = kOutputOnly;

      MS_EXCEPTION_IF_CHECK_FAIL(tensors_list_.size() == output_tensor_index, "tensors_list_ size error!!!");
      tensors_list_.push_back(tensor);
      std::for_each(nodes.begin(), nodes.end(), [tensor](auto &node) {
        MS_EXCEPTION_IF_NULL(node);
        node->output_tensors_.push_back(tensor);
      });
    }

    // WorkSpace Tensor
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (const auto &size : workspace_sizes) {
      auto workspace_tensor_index = tensor_index;
      tensor_index++;
      size_t aligned_size = GetAlignSize(size);
      if (aligned_size == 0) {
        // Device Address still need to be allocated when workspace_size is 0
        aligned_size = GetAlignSize(kZeroAlignSize);
      }
      SomasTensorPtr tensor = std::make_shared<SomasTensor>(workspace_tensor_index, node->GetId(), stream_id, size,
                                                            aligned_size, kLifeLongNone);
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->type_ = kWorkspace;
      tensor->lifetime_.start_ = node->GetId();
      tensor->lifetime_.end_ = (nodes.size() > 1) ? nodes.back()->GetId() : node->GetId();

      MS_EXCEPTION_IF_CHECK_FAIL(tensors_list_.size() == workspace_tensor_index, "tensors_list_ size error!!!");
      tensors_list_.push_back(tensor);
      std::for_each(nodes.begin(), nodes.end(), [tensor](auto &node) {
        MS_EXCEPTION_IF_NULL(node);
        node->workspace_tensors_.push_back(tensor);
      });
    }
  }
}

void Somas::InitSomasInputTensors(const session::KernelGraph &graph) {
  MS_LOG(DEBUG) << "Somas InitSomasInputTensors start...";
  static const auto enable_fusion_clear = (common::GetEnv("ENV_FUSION_CLEAR") == "1");
  auto &kernel_cnodes = graph.execution_order();
  for (const auto &kernel : kernel_cnodes) {
    if (common::AnfAlgo::GetCNodeName(kernel) != kAtomicAddrCleanOpName) {
      InitCommonNodeInputs(kernel);
    } else {
      InitAtomicCleanInputs(enable_fusion_clear, kernel);
    }
  }
}

void Somas::InitCommonNodeInputs(const CNodePtr &kernel) {
  auto nodes = nodes_map_[kernel.get()];
  auto node = nodes[0];
  MS_EXCEPTION_IF_NULL(node);

  // Input Tensor
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  size_t real_input_index = 0;
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto &input_size_list = kernel_mod->GetInputSizeList();
  for (size_t i = 0; i < input_tensor_num; i++) {
    auto input_node = kernel->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    session::KernelWithIndex prenode_index = GetVisitKernelWithReturnType(input_node, 0);
    MS_EXCEPTION_IF_NULL(prenode_index.first);
    if (common::AnfAlgo::CheckPrimitiveType(prenode_index.first, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "Node " << node->scope_full_name_ << "'s input node [" << input_node->DebugString()
                        << "]'s input " << i << " is MakeTuple";
    }
    if (!AnfUtils::IsRealCNodeKernel(prenode_index.first)) {
      auto op_name = common::AnfAlgo::GetCNodeName(kernel);
      TypeId input_origin_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel, i);
      if ((op_name == kDynamicRNNOpName || op_name == kDynamicGRUV2OpName) && input_origin_type == kMetaTypeNone) {
        continue;
      }
      auto index = AnfAlgo::GetInputKernelIdxByGraphIdx(kernel, i);
      size_t input_size = 0;
      if (index >= input_size_list.size()) {
        MS_LOG(INFO) << "Node: " << kernel->fullname_with_scope() << " input idx: " << index
                     << " greater than the size of input_size_list: " << input_size_list.size()
                     << ", so use 0 as parameter size.";
      } else {
        input_size = input_size_list.at(index);
      }
      auto parameter =
        GetSomasParameter(prenode_index.first, prenode_index.second, input_size, kernel->fullname_with_scope());
      node->input_parameters_map_[real_input_index] = parameter;
      real_input_index++;
      MS_LOG(DEBUG) << "Input  [" << prenode_index.first->fullname_with_scope() << "] is not a real cnode kernel.";
      continue;
    }

    auto iter = nodes_map_.find(prenode_index.first.get());
    if (iter == nodes_map_.end()) {
      MS_LOG(EXCEPTION) << "Kernel[" << kernel->fullname_with_scope() << "]'s input " << i << " ["
                        << prenode_index.first->fullname_with_scope() << "] is not init.";
    }
    SomasNodePtr pre_somas_node = iter->second.at(0);
    if (prenode_index.second > pre_somas_node->output_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Output index " << prenode_index.second << " exceed input node ["
                        << prenode_index.first->fullname_with_scope() << "]'s outputs size "
                        << pre_somas_node->output_tensors_.size();
    }
    auto input_somas_tensor = pre_somas_node->output_tensors_[prenode_index.second];
    MS_EXCEPTION_IF_NULL(input_somas_tensor);
    std::for_each(nodes.begin(), nodes.end(),
                  [input_somas_tensor](auto &node) { node->input_tensors_.push_back(input_somas_tensor); });
    real_input_index++;
    if (input_somas_tensor->type_ == kOutputOnly) {
      input_somas_tensor->type_ = kCommon;
    }

    for (auto &repeat_node : nodes) {
      input_somas_tensor->destination_nodes_.insert(repeat_node->GetId());
      input_somas_tensor->consumer_list_.emplace_back(repeat_node->GetId());
      if (input_somas_tensor->lifetime_.end_ < repeat_node->GetId()) {
        input_somas_tensor->lifetime_.end_ = repeat_node->GetId();
      }
    }

    if (node != pre_somas_node) {
      node->ancestor_nodes_.insert(pre_somas_node);
    }
  }
}

void Somas::InitAtomicCleanInputs(bool enable_fusion_clear, const CNodePtr &kernel) {
  auto node = nodes_map_[kernel.get()].at(0);
  MS_EXCEPTION_IF_NULL(node);
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_tensor_num; i++) {
    MS_EXCEPTION_IF_NULL(kernel->inputs()[i + 1]);
    auto pre_node = kernel->input(i + 1)->cast<CNodePtr>();
    auto iter = nodes_map_.find(pre_node.get());
    if (iter == nodes_map_.end()) {
      MS_LOG(EXCEPTION) << "Kernel[" << kernel->fullname_with_scope() << "]'s input ["
                        << pre_node->fullname_with_scope() << "] is not init.";
    }
    auto pre_somas_node = iter->second.at(0);
    MS_EXCEPTION_IF_NULL(pre_somas_node);
    // set clean output tensors
    if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
      auto clean_output_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
      for (auto index : clean_output_indexs) {
        if (index > pre_somas_node->output_tensors_.size()) {
          MS_LOG(EXCEPTION) << "Output index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                            << "]'s outputs size " << pre_somas_node->output_tensors_.size();
        }
        auto input_somas_tensor = pre_somas_node->output_tensors_[index];
        MS_EXCEPTION_IF_NULL(input_somas_tensor);
        node->input_tensors_.push_back(input_somas_tensor);
        if (enable_fusion_clear) {
          input_somas_tensor->lifelong_value_ = kLifeLongGraphStart;
          MS_LOG(INFO) << "Set " << node->scope_full_name_ << "'s Input node " << pre_somas_node->scope_full_name_
                       << " 's output" << index << " to lifelong";
        }
      }
    }
    // set clean workspace tensors
    if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
      auto clean_workspace_indexs =
        common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
      for (const auto &index : clean_workspace_indexs) {
        if (index > pre_somas_node->output_tensors_.size()) {
          MS_LOG(EXCEPTION) << "Workspace index " << index << " exceed input node [" << pre_node->fullname_with_scope()
                            << "]'s Workspace size " << pre_somas_node->workspace_tensors_.size();
        }
        auto input_somas_tensor = pre_somas_node->workspace_tensors_[index];
        MS_EXCEPTION_IF_NULL(input_somas_tensor);
        node->input_tensors_.push_back(input_somas_tensor);
        if (enable_fusion_clear) {
          input_somas_tensor->lifelong_value_ = kLifeLongGraphStart;
          MS_LOG(INFO) << "Set " << node->scope_full_name_ << "'s Input node " << pre_somas_node->scope_full_name_
                       << " 's workspace" << index << " to lifelong";
        }
      }
    }
  }
}

SomasParameterPtr Somas::CreateSomasParameter(const AnfNodePtr &node, size_t index, size_t param_size,
                                              const std::string &kernel_name) {
  MS_EXCEPTION_IF_NULL(node);
  auto id = parameters_list_.size();
  const void *addr = nullptr;
  if (AnfAlgo::OutputAddrExist(node, index)) {
    auto device_addr = AnfAlgo::GetOutputAddr(node, index);
    if (device_addr == nullptr) {
      MS_LOG(EXCEPTION) << "Node " << node->fullname_with_scope() << " has no device address before Somas.";
    }
    addr = device_addr->GetPtr();
    if (device_addr->GetSize() != param_size) {
      MS_LOG(INFO) << "Dev Size and Param Size is not equal, Node: " << node->fullname_with_scope()
                   << " Dev Size: " << device_addr->GetSize() << " param_size: " << param_size
                   << " Kernel: " << kernel_name;
    }
  }

  auto param = std::make_shared<SomasParameter>(id, node->fullname_with_scope(), index, addr, param_size);
  parameters_list_.push_back(param);
  return param;
}

SomasParameterPtr Somas::GetSomasParameter(const AnfNodePtr &node, size_t index, size_t param_size,
                                           const std::string &kernel_name) {
  auto key = node.get();
  auto iter = parameters_map_.find(key);
  if (iter != parameters_map_.end()) {
    auto it = std::find_if(iter->second.begin(), iter->second.end(),
                           [index](const SomasParameterPtr &param) -> bool { return index == param->output_index_; });
    if (it != iter->second.end()) {
      return *it;
    } else {
      auto new_param = CreateSomasParameter(node, index, param_size, kernel_name);
      iter->second.push_back(new_param);
      return new_param;
    }
  } else {
    auto param = CreateSomasParameter(node, index, param_size, kernel_name);
    parameters_map_[key].push_back(param);
    return param;
  }
}

bool Somas::InitBasicInfoFromGraph(const session::KernelGraph &graph) {
  InitSomasStreamAndNode(graph);
  InitSomasOutputAndWorkspaceTensors(graph);
  InitSomasInputTensors(graph);
  InitControlTensors();
  GraphOutputProcess(graph);
  return true;
}

#ifndef ENABLE_SECURITY
void Somas::SummaryInputProcess(const session::KernelGraph &graph) {
  bool summary_exist = graph.summary_node_exist();
  if (!summary_exist) {
    return;
  }

  auto summary_nodes = graph.summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  size_t total_summary_size = 0;
  for (const auto &node_item : summary_nodes) {
    auto origin_node = node_item.second.first;
    size_t origin_index = IntToSize(node_item.second.second);
    auto item_with_index = GetVisitKernelWithReturnType(origin_node, origin_index);
    auto node = item_with_index.first;
    size_t index = item_with_index.second;
    auto iter = nodes_map_.find(node.get());
    if (iter != nodes_map_.end()) {
      auto input_node = iter->second.at(0);
      MS_EXCEPTION_IF_NULL(input_node);
      if (index < input_node->output_tensors_.size()) {
        auto tensor = input_node->output_tensors_[index];
        MS_EXCEPTION_IF_NULL(tensor);
        tensor->lifelong_value_ = kLifeLongGraphEnd;
        tensor->type_ = kSummaryInput;
        total_summary_size += tensor->GetAlignedSize();
        MS_LOG(INFO) << "Set summary node input tensor's lifelong, node: " << node->fullname_with_scope()
                     << " index: " << index;
      } else {
        MS_LOG(WARNING) << "Index exceed size, node " << node->fullname_with_scope() << " index: " << index
                        << " size: " << input_node->output_tensors_.size();
      }
    } else {
      MS_LOG(WARNING) << "Can't find summary input node " << node->fullname_with_scope() << " index: " << index;
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: SummaryNodes: " << total_summary_size;
}
#endif

void Somas::GraphOutputProcess(const session::KernelGraph &graph) {
  size_t count = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph.output());
  for (auto &output : outputs) {
    auto output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto output_kernel = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_kernel);
    while (AnfUtils::IsRealCNodeKernel(output_kernel) && nodes_map_.find(output_kernel.get()) == nodes_map_.end()) {
      auto cnode = output_kernel->cast<CNodePtr>();
      if (!common::AnfAlgo::IsNopNode(cnode)) {
        MS_LOG(EXCEPTION) << "Node[" << cnode->fullname_with_scope()
                          << "] doesn't exist in nodes_map and is not a nop node!!!";
      }
      output_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, false);
      output_kernel = output_with_index.first;
    }

    if (!AnfUtils::IsRealCNodeKernel(output_kernel)) {
      continue;
    }

    auto output_index = output_with_index.second;
    auto iter = nodes_map_.find(output_kernel.get());
    if (iter != nodes_map_.end()) {
      auto &node = iter->second.at(0);
      MS_EXCEPTION_IF_NULL(node);
      if (output_index <= node->output_tensors_.size()) {
        auto &tensor = node->output_tensors_[output_index];
        tensor->aligned_size_ = 0;
        tensor->type_ = kGraphOutput;
        count++;
      } else {
        MS_LOG(EXCEPTION) << "Graph's output node " << output_kernel->fullname_with_scope() << "'s output index"
                          << output_index << " is larger than its output tensor number "
                          << node->output_tensors_.size();
      }
    } else {
      MS_LOG(EXCEPTION) << "Can't find somas node for graph output node " << output_kernel->fullname_with_scope();
    }
  }
  MS_LOG(INFO) << "Set " << count << " graph output tensors' aligned size to 0.";
}

void Somas::RefNodeProcess(const session::KernelGraph &graph) {
  auto &kernel_cnodes = graph.execution_order();
  size_t total_output_size = 0;
  size_t total_input_size = 0;
  for (const auto &kernel : kernel_cnodes) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    if (kernel_mod == nullptr) {
      MS_LOG(WARNING) << "Kernel mode is NULL Of " << kernel->fullname_with_scope();
      continue;
    }
    auto output_sizes = kernel_mod->GetOutputSizeList();
    size_t output_index = 0;
    for (const auto &size : output_sizes) {
      auto out_index = output_index;
      output_index++;
      session::AnfWithOutIndex out_pair(kernel, out_index);
      if (graph.IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph.GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto &node = nodes_map_[kernel.get()].at(0);
        MS_EXCEPTION_IF_NULL(node);
        auto output_tensor = node->output_tensors_[out_index];
        MS_EXCEPTION_IF_NULL(output_tensor);
        output_tensor->type_ = kUnion;
        total_output_size += size;

        if (!AnfUtils::IsRealCNodeKernel(origin_pair.first)) {
          output_tensor->type_ = kGraphInput;
          output_tensor->aligned_size_ = 0;
          continue;
        }

        if (nodes_map_.find(origin_pair.first.get()) == nodes_map_.end()) {
          auto cnode = origin_pair.first->cast<CNodePtr>();
          if (!common::AnfAlgo::IsNopNode(cnode)) {
            MS_LOG(EXCEPTION) << "Node[" << origin_pair.first->fullname_with_scope() << "] find input node["
                              << cnode->fullname_with_scope()
                              << "] doesn't exist in nodes_map and is not a nop node!!!!";
          }
          origin_pair = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, false);
        }
        if (!origin_pair.first->isa<CNode>()) {
          MS_LOG(EXCEPTION) << "The origin_pair.first is not a cnode. Info origin_pair.first: "
                            << origin_pair.first->DebugString();
        }
        auto ori_node = origin_pair.first->cast<CNodePtr>();
        auto ori_index = origin_pair.second;
        if (nodes_map_.find(ori_node.get()) == nodes_map_.end()) {
          MS_LOG(EXCEPTION)
            << "The ori_node is not included in nodes_map_ constructed from exec_order of graph. Info ori_node: "
            << ori_node->DebugString();
        }
        auto &repeat_node = nodes_map_[ori_node.get()].at(0);
        MS_EXCEPTION_IF_NULL(repeat_node);
        auto input_tensor = repeat_node->output_tensors_[ori_index];
        MS_EXCEPTION_IF_NULL(input_tensor);
        input_tensor->type_ = kUnion;
        total_input_size += input_tensor->aligned_size_;
        std::vector<size_t> refnode_input_output;
        refnode_input_output.push_back(input_tensor->GetId());
        refnode_input_output.push_back(output_tensor->GetId());
        union_tensors_list_.push_back(refnode_input_output);
        MS_LOG(INFO) << "RefNode: input " << input_tensor->GetId() << " output " << output_tensor->GetId();
      }
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: RefNode: input " << total_input_size << " output " << total_output_size;
}

void Somas::UnReuseNodeProcess(const session::KernelGraph &graph) {
  std::map<std::string, UnReuseType> full_name_type = {};
  for (const auto &node : un_reuse_node_name_) {
    full_name_type.insert(node);
  }

  auto &kernel_cnodes = graph.execution_order();
  for (const auto &kernel : kernel_cnodes) {
    auto type = common::AnfAlgo::GetCNodeName(kernel);
    auto iter = un_reuse_node_type_.find(type);
    if (iter == un_reuse_node_type_.end()) {
      continue;
    }
    auto full_name = kernel->fullname_with_scope();
    full_name_type[full_name] = iter->second;
  }

  if (full_name_type.empty()) {
    return;
  }

  for (const auto &kernel : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto full_name = kernel->fullname_with_scope();
    auto iter = full_name_type.find(full_name);
    if (iter == full_name_type.end()) {
      continue;
    }
    auto un_reuse_type = iter->second;
    MS_LOG(INFO) << "Set UnReuse Node in somas, Node:" << iter->first << ", UnReuse type:" << un_reuse_type;
    auto key = kernel.get();
    auto somas_node = nodes_map_[key].at(0);
    MS_EXCEPTION_IF_NULL(somas_node);
    // input
    if (un_reuse_type == UnReuseType::kUnReuseAll || un_reuse_type == UnReuseType::kUnReuseInput) {
      auto inputs = somas_node->input_tensors_;
      for (auto &input : inputs) {
        MS_EXCEPTION_IF_NULL(input);
        input->lifelong_value_ = kLifeLongGraphAll;
      }
    }

    // output
    if (un_reuse_type == UnReuseType::kUnReuseAll || un_reuse_type == UnReuseType::kUnReuseOutput) {
      auto outputs = somas_node->output_tensors_;
      MS_LOG(INFO) << "Output size of " << kernel->fullname_with_scope() << " is  " << outputs.size();
      for (auto &output : outputs) {
        MS_EXCEPTION_IF_NULL(output);
        output->lifelong_value_ = kLifeLongGraphAll;
      }
    }

    // workspace
    if (un_reuse_type == UnReuseType::kUnReuseAll || un_reuse_type == UnReuseType::kUnReuseWorkspace) {
      auto workspaces = somas_node->workspace_tensors_;
      for (auto &workspace : workspaces) {
        MS_EXCEPTION_IF_NULL(workspace);
        workspace->lifelong_value_ = kLifeLongGraphAll;
      }
    }
  }
}

void Somas::CommunicationNodeProcess() {
  for (const auto &node : nodes_list_) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->GetType() != kCommunicationNode) {
      continue;
    }

    // Contiguous input
    if ((!node->input_tensors_.empty()) && (!node->input_tensors_[0]->contiguous_)) {
      CommunicationTensorProcess(node->input_tensors_);
      std::vector<size_t> inputs;
      for (const auto &input_tensor : node->input_tensors_) {
        MS_EXCEPTION_IF_NULL(input_tensor);
        comm_input_total_size_ += input_tensor->aligned_size_;
        input_tensor->contiguous_ = true;
        inputs.push_back(input_tensor->GetId());
      }
      if (inputs.size() != (std::set<size_t>(inputs.begin(), inputs.end())).size()) {
        MS_LOG(EXCEPTION) << node->scope_full_name_
                          << " has same input tensors, please double check node input tensors.";
      }
      if (!NeedContiguous(inputs)) {
        continue;
      }
      contiguous_tensors_list_.push_back(inputs);
    }

    // Contiguous output
    if ((!node->output_tensors_.empty()) && (!node->output_tensors_[0]->contiguous_)) {
      CommunicationTensorProcess(node->output_tensors_);
      std::vector<size_t> outputs;
      for (const auto &output_tensor : node->output_tensors_) {
        MS_EXCEPTION_IF_NULL(output_tensor);
        comm_output_total_size_ += output_tensor->aligned_size_;
        output_tensor->contiguous_ = true;
        outputs.push_back(output_tensor->GetId());
      }
      if (outputs.size() != (std::set<size_t>(outputs.begin(), outputs.end())).size()) {
        MS_LOG(EXCEPTION) << node->scope_full_name_
                          << " has same output tensor, please double check node output tensors.";
      }
      if (!NeedContiguous(outputs)) {
        continue;
      }
      contiguous_tensors_list_.push_back(outputs);
    }

    // check the tensors of the list
    std::set<size_t> all_contiguous_tensors_set;
    size_t all_contiguous_tensors_num = 0;
    for (auto &tensors : contiguous_tensors_list_) {
      all_contiguous_tensors_num += tensors.size();
      all_contiguous_tensors_set.insert(tensors.begin(), tensors.end());
    }
    if (all_contiguous_tensors_num != all_contiguous_tensors_set.size()) {
      MS_LOG(EXCEPTION) << "Please check the CommunicationNodes, some tensor are in multiple contiguous list";
    }
  }
}

bool Somas::NodeSort(const SomasNodePtr &node1, const SomasNodePtr &node2) { return node1->GetId() < node2->GetId(); }

void Somas::BuildConflictInfo(const std::shared_ptr<SomasTensor> &tensor, TensorConflictInfo *tensor_conflict_info,
                              std::vector<size_t> *destination_node_list) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(tensor_conflict_info);
  const auto &consumer_list = tensor->consumer_list_;
  tensor_conflict_info->destination_num = consumer_list.size();

  //  the destination_node size of most nodes is small.
  //  in order to have better spatial locality in the loop, when the destination_num is 1 or 2,
  //  the destination node is directly stored in the structure.
  if (tensor_conflict_info->destination_num == kOnlyOneDestinationNode) {
    tensor_conflict_info->l.id = consumer_list.back();
  } else if (tensor_conflict_info->destination_num == kOnlyTwoDestinationNode) {
    tensor_conflict_info->l.id = consumer_list.at(0);
    tensor_conflict_info->r.id = consumer_list.at(1);
  } else {
    tensor_conflict_info->l.index = destination_node_list->size();
    destination_node_list->insert(destination_node_list->cend(), consumer_list.cbegin(), consumer_list.cend());
    tensor_conflict_info->r.index = destination_node_list->size();
  }
}

void Somas::ComputeBasicMatrix() {
  MS_LOG(INFO) << "Start Conflict Computing (Bitset Model)";
  auto start_conflict = std::chrono::system_clock::now();
  std::sort(nodes_list_.begin(), nodes_list_.end(), NodeSort);
  UpdateTensorDestinations();

  MS_LOG(INFO) << "Start Bitset";
  std::vector<DynamicBitSet> nodes_dependency;

  size_t count = nodes_list_.back()->GetId() + 1;
  for (size_t i = 0; i < count; i++) {
    (void)nodes_dependency.emplace_back(count);
  }

  MS_LOG(INFO) << "Start Path Computing";
  // Loop to compute ancestor paths via bitset for time dependence
  for (const auto &node : nodes_list_) {
    for (const auto &ancestor : node->ancestor_nodes_) {
      nodes_dependency[node->GetId()].SetBitTrue(ancestor->GetId());
      Union(&nodes_dependency[node->GetId()], &nodes_dependency[ancestor->GetId()]);
    }
  }
  MS_LOG(INFO) << "End Path Computing";

  MS_LOG(INFO) << "Start Tensor Relation Computing";
  count = tensors_list_.back()->GetId() + 1;
  for (size_t i = 0; i < count; i++) {
    (void)reuse_matrix_.emplace_back(count);
  }

  std::vector<TensorConflictInfo> tensor_conflict_info_list;
  std::vector<size_t> destination_node_list;
  std::vector<SomasTensorPtr> candidate_tensor_list;
  for (const auto &calc_tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(calc_tensor);
    // If the life cycle of the tensor is global, or the tensor does not need to allocate memory, it is not reused
    if (calc_tensor->IsLifelong() || calc_tensor->GetAlignedSize() == 0) {
      continue;
    }
    candidate_tensor_list.emplace_back(calc_tensor);
    tensor_conflict_info_list.emplace_back(calc_tensor->GetId(), calc_tensor->GetSourceNodeId());
    BuildConflictInfo(calc_tensor, &tensor_conflict_info_list.back(), &destination_node_list);
  }
  std::shuffle(candidate_tensor_list.begin(), candidate_tensor_list.end(), std::mt19937(std::random_device()()));

  if (candidate_tensor_list.size() < kParallelComputeSizeThreshold) {
    ComputeMultiTensorConflicts(candidate_tensor_list, tensor_conflict_info_list, destination_node_list,
                                nodes_dependency, &reuse_matrix_);
  } else {
    MS_LOG(INFO) << "Candidate Tensor Num " << candidate_tensor_list.size() << " is larger than "
                 << kParallelComputeSizeThreshold;
    MS_LOG(INFO) << "Enter Multi-Thread Mode...";
    size_t process_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
    if (process_num == IntToSize(0)) {
      MS_LOG(EXCEPTION) << "Threads Num is Zero !!!!!";
    }
    MS_LOG(INFO) << "Threads Num is " << process_num;

    int64_t start_index = 0;
    int64_t total_size = SizeToLong(candidate_tensor_list.size());
    int64_t job_size = total_size / SizeToLong(process_num);
    if (job_size == 0) {
      job_size = total_size;
    }
    std::vector<common::Task> tasks;
    while (start_index < total_size) {
      int64_t end_index = (start_index + job_size) > total_size ? total_size : start_index + job_size;
      auto jobs = std::vector<SomasTensorPtr>(candidate_tensor_list.begin() + start_index,
                                              candidate_tensor_list.begin() + end_index);
      auto task = [this, jobs, &tensor_conflict_info_list, &destination_node_list, &nodes_dependency]() {
        this->ComputeMultiTensorConflicts(jobs, tensor_conflict_info_list, destination_node_list, nodes_dependency,
                                          &reuse_matrix_);
        return common::SUCCESS;
      };
      (void)tasks.emplace_back(task);
      start_index += job_size;
    }

    (void)common::ThreadPool::GetInstance().SyncRun(tasks);
  }

  auto end_conflict = std::chrono::system_clock::now();
  MS_LOG(INFO) << "End Basic Conflict Computing (Bitset Model)(time taken "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end_conflict - start_conflict).count() << "ms)";
}

void Somas::ProcessSemiLifeLongTensor() {
  for (const auto &calc_tensor : tensors_list_) {
    // if the tensor is semi-life long start, it can't reuse with tensor with smaller id.
    // if the tensor is semi-life long end, it can't reuse with tensor with larger id.
    if (!calc_tensor->IsSemiLifelongStart() && !calc_tensor->IsSemiLifelongEnd()) {
      continue;
    }
    for (const auto &target_tensor : tensors_list_) {
      if (calc_tensor == target_tensor) {
        continue;
      }
      if (depend_exec_order_) {
        if ((calc_tensor->IsSemiLifelongStart() && target_tensor->GetId() < calc_tensor->GetId()) ||
            (calc_tensor->IsSemiLifelongEnd() && target_tensor->GetId() > calc_tensor->GetId())) {
          reuse_matrix_[calc_tensor->GetId()].SetBitFalse(target_tensor->GetId());
          reuse_matrix_[target_tensor->GetId()].SetBitFalse(calc_tensor->GetId());
        }
      } else {
        reuse_matrix_[calc_tensor->GetId()].SetBitFalse(target_tensor->GetId());
        reuse_matrix_[target_tensor->GetId()].SetBitFalse(calc_tensor->GetId());
      }
    }
  }
}

void Somas::ComputeConflictMatrix() {
  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor for Conflict computing";
    return;
  }
  ComputeBasicMatrix();
  ProcessSemiLifeLongTensor();
  UpdateUnionTensorsConflict();
}

void Somas::UpdateContiguousTensorList() {
  processed_contiguous_tensors_list_.clear();
  processed_contiguous_tensors_list_.insert(processed_contiguous_tensors_list_.end(), contiguous_tensors_list_.begin(),
                                            contiguous_tensors_list_.end());
  std::set<std::vector<size_t>> contiguous_tensors_list_to_remove;

  GetContiguousListContainUnionTensor();
  for (const auto &ref_list_pair : contiguous_list_with_ref_index_map_) {
    contiguous_tensors_list_to_remove.insert(contiguous_tensors_list_[ref_list_pair.second]);
  }

  // remove the contiguous list which all tensors' align size is 0
  for (const auto &contiguous_list : contiguous_tensors_list_) {
    bool all_outputs = true;
    for (auto tensor_id : contiguous_list) {
      auto tensor = tensors_list_[tensor_id];
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->aligned_size_ != 0) {
        all_outputs = false;
        break;
      }
    }

    if (all_outputs) {
      contiguous_tensors_list_to_remove.insert(contiguous_list);
    }
  }

  for (const auto &contiguous_list : contiguous_tensors_list_to_remove) {
    auto iterator =
      std::find(processed_contiguous_tensors_list_.begin(), processed_contiguous_tensors_list_.end(), contiguous_list);
    if (iterator != processed_contiguous_tensors_list_.end()) {
      processed_contiguous_tensors_list_.erase(iterator);
    } else {
      MS_LOG(WARNING) << "Could not find contiguous list to remove for ref";
    }
  }
}

void Somas::UpdateTensorDestinations() {
  // Loop to avoid tensors with empty destinations (add itself)
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->destination_nodes_.empty()) {
      tensor->destination_nodes_.insert(tensor->GetSourceNodeId());
      tensor->consumer_list_.emplace_back(tensor->GetSourceNodeId());
    }
  }
}

void Somas::ComputeMultiTensorConflicts(const std::vector<SomasTensorPtr> &target_tensors_list,
                                        const std::vector<TensorConflictInfo> &tensor_conflict_info_list,
                                        const std::vector<size_t> &destination_node_list,
                                        const vector<DynamicBitSet> &nodes_dependency,
                                        std::vector<DynamicBitSet> *tensor_relation) const {
  auto start = std::chrono::system_clock::now();
  MS_LOG(INFO) << "Start Computing Conflicts Pairs, tensors list size is " << target_tensors_list.size();
  for (const auto &target_tensor : target_tensors_list) {
    MS_EXCEPTION_IF_NULL(target_tensor);
    ComputeOneTensorConflicts(target_tensor, tensor_conflict_info_list, destination_node_list, nodes_dependency,
                              tensor_relation);
  }
  auto end = std::chrono::system_clock::now();
  MS_LOG(INFO) << "End Computing Conflicts Pairs (time taken "
               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms)";
}

bool Somas::CheckIsDependency(const TensorConflictInfo &tensor_conflict_info, const size_t &src_node_id,
                              const vector<DynamicBitSet> &nodes_dependency,
                              const std::vector<size_t> &destination_node_list) {
  // check calc_tensor's all consumers is target_tensor's source node's dependency or not
  if (tensor_conflict_info.destination_num == kOnlyOneDestinationNode) {
    // calc_tensor's consumer is not in target_tensor's source node's dependency, not sure this consumer is done or
    // not when target_tensor produced
    // calc_tensor is target_tensor's source node's input, can't reuse
    if (!nodes_dependency[src_node_id].IsBitTrue(tensor_conflict_info.l.id) ||
        src_node_id == tensor_conflict_info.l.id) {
      return false;
    }
  } else if (tensor_conflict_info.destination_num == kOnlyTwoDestinationNode) {
    if (!nodes_dependency[src_node_id].IsBitTrue(tensor_conflict_info.l.id) ||
        !nodes_dependency[src_node_id].IsBitTrue(tensor_conflict_info.r.id) ||
        src_node_id == tensor_conflict_info.l.id || src_node_id == tensor_conflict_info.r.id) {
      return false;
    }
  } else {
    for (size_t i = tensor_conflict_info.l.index; i < tensor_conflict_info.r.index; i++) {
      const auto &dst_node_id = destination_node_list[i];
      if (!nodes_dependency[src_node_id].IsBitTrue(dst_node_id) || src_node_id == dst_node_id) {
        return false;
      }
    }
  }
  // calc_tensor's consumer is in target_tensor's source node's dependency, this consumer is done when
  // target_tensor produced
  return true;
}

void Somas::ComputeOneTensorConflicts(const std::shared_ptr<SomasTensor> &target_tensor,
                                      const std::vector<TensorConflictInfo> &tensor_conflict_info_list,
                                      const std::vector<size_t> &destination_node_list,
                                      const vector<DynamicBitSet> &nodes_dependency,
                                      std::vector<DynamicBitSet> *tensor_relation) {
  MS_EXCEPTION_IF_NULL(target_tensor);
  auto target_tensor_id = target_tensor->GetId();
  auto target_src_node_id = target_tensor->GetSourceNodeId();

  std::vector<size_t> target_destination_node_list;
  TensorConflictInfo target_info(target_tensor->GetId(), target_tensor->GetSourceNodeId());
  BuildConflictInfo(target_tensor, &target_info, &target_destination_node_list);

  //  the conflict info of per calc_tensor
  for (const auto &tensor_conflict_info : tensor_conflict_info_list) {
    if (tensor_conflict_info.tensor_id == target_tensor_id || tensor_conflict_info.src_node_id == target_src_node_id) {
      continue;
    }

    if (CheckIsDependency(tensor_conflict_info, target_src_node_id, nodes_dependency, destination_node_list) ||
        CheckIsDependency(target_info, tensor_conflict_info.src_node_id, nodes_dependency,
                          target_destination_node_list)) {
      // calc_tensor and target_tensor have dependencies so they can reuse each other
      (*tensor_relation)[target_tensor_id].SetBitTrue(tensor_conflict_info.tensor_id);
    }
  }
}

void Somas::Solve(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "Somas Assign start...";
  if (tensors_list_.empty()) {
    MS_LOG(INFO) << "No Tensor for Assigner";
    return;
  }

  // Compute number of constraints for each tensor which will used in solver
  auto tensors_num = tensors_list_.size();
  for (const auto &tensor : tensors_list_) {
    auto ones_num = reuse_matrix_[tensor->GetId()].CountOnesNum();
    tensor->num_constraints_ = tensors_num - ones_num;
  }

  // Prepare solver info
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->GetSolverTensorDesc() != nullptr) {
      SomasSolverTensorDescPtr pSolverTensor = tensor->GetSolverTensorDesc();
      (void)solver_tensor_desc_map_.emplace(pSolverTensor->index_, pSolverTensor);
    }
  }

  MS_LOG(INFO) << "Start Solving";
  if (solver_tensor_desc_map_.empty()) {
    MS_LOG(INFO) << "solver_tensor_desc_list is empty.";
    return;
  }

  somas_solver_ = std::make_shared<SomasSolverPre>();
  auto status =
    somas_solver_->Solving(graph, &solver_tensor_desc_map_, &reuse_matrix_, processed_contiguous_tensors_list_, false);
  MS_LOG(INFO) << "End Solving";

  GenGraphStatisticInfo();

  if (status != SUCCESS) {
    MS_LOG(EXCEPTION) << "SOMAS Solving Failed.";
  }

  // Update solver_tensor_desc offset to tensors list
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->SetOffset();
  }

  UpdateUnionTensorsOffset();
  UpdateContiguousTensorsOffset(contiguous_list_with_ref_index_map_);

  reused_memory_size_ = static_cast<size_t>(somas_solver_->GetMaxOffset());

  MS_LOG(INFO) << "Somas Assign end.";
}

std::map<size_t, std::map<size_t, std::set<size_t>>> Somas::GetContiguousRefListErrorCheckMap() {
  std::map<size_t, std::map<size_t, std::set<size_t>>> contiguous_ref_list_error_check_map;
  std::map<size_t, size_t> ref_tensors_in_contiguous_map = GetRefTensorsInContiguousList();
  for (const auto &ref_pair : ref_tensors_in_contiguous_map) {
    size_t ref_first = ref_pair.first;
    size_t ref_second = ref_pair.second;
    bool found_first = false;
    bool found_second = false;
    size_t index_first = 0;
    size_t index_second = 0;
    size_t index_in_list_first = 0;
    size_t index_in_list_second = 0;
    for (size_t index = 0; index < contiguous_tensors_list_.size() && (!found_first || !found_second); index++) {
      if (!found_first) {
        auto iterator_first =
          std::find(contiguous_tensors_list_[index].begin(), contiguous_tensors_list_[index].end(), ref_first);
        if (iterator_first != contiguous_tensors_list_[index].end()) {
          index_first = index;
          index_in_list_first = IntToSize(iterator_first - contiguous_tensors_list_[index].begin());
          found_first = true;
        }
      }
      if (!found_second) {
        auto iterator_second =
          std::find(contiguous_tensors_list_[index].begin(), contiguous_tensors_list_[index].end(), ref_second);
        if (iterator_second != contiguous_tensors_list_[index].end()) {
          index_second = index;
          index_in_list_second = IntToSize(iterator_second - contiguous_tensors_list_[index].begin());
          found_second = true;
        }
      }
    }

    if (!found_first) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_first << " not found in any contiguous list";
    }
    if (!found_second) {
      MS_LOG(WARNING) << "Contiguous ref tensor " << ref_second << " not found in any contiguous list";
    }
    if (contiguous_list_with_ref_index_map_.find(index_first) == contiguous_list_with_ref_index_map_.end() ||
        contiguous_list_with_ref_index_map_[index_first] == index_second) {
      contiguous_list_with_ref_index_map_[index_first] = index_second;
      // Checking for error cases
      if (index_in_list_first != index_in_list_second) {
        MS_LOG(WARNING) << "Inconsistency in contiguous ref: tensor " << ref_first << " in position "
                        << index_in_list_first << " of contiguous list " << index_first << " and tensor " << ref_second
                        << " in position " << index_in_list_second << " of contiguous list " << index_second;
      }
      (void)contiguous_ref_list_error_check_map[index_first][index_second].insert(index_in_list_first);
    } else {
      MS_LOG(WARNING) << "Contiguous list " << index_first << " associated (ref node) with two other contiguous lists: "
                      << contiguous_list_with_ref_index_map_[index_first] << " and " << index_second;
    }
  }
  return contiguous_ref_list_error_check_map;
}

void Somas::GetContiguousListContainUnionTensor() {
  // key: contiguous list index with first union tensor; value: contiguous list index with other union tensor
  contiguous_list_with_ref_index_map_.clear();
  std::map<size_t, std::map<size_t, std::set<size_t>>> contiguous_ref_list_error_check_map =
    GetContiguousRefListErrorCheckMap();
  for (const auto &check_list_pair : contiguous_ref_list_error_check_map) {
    auto first_list = check_list_pair.first;
    auto index_set_map = check_list_pair.second;
    for (const auto &index_set : index_set_map) {
      auto second_list = index_set.first;
      if (contiguous_tensors_list_[first_list].size() != contiguous_tensors_list_[second_list].size()) {
        MS_LOG(WARNING) << "Contiguous lists " << first_list << " and " << second_list
                        << " considered in ref do not have the same size";
      }
      for (size_t x = 0; x < contiguous_tensors_list_[second_list].size(); x++) {
        if (contiguous_ref_list_error_check_map[first_list][second_list].count(x) == 0) {
          MS_LOG(WARNING) << "Contiguous lists " << first_list << " and " << second_list
                          << " considered in ref: ref pair at in-lists index " << x << " has not been considered";
        }
      }
    }
  }
}

std::map<size_t, size_t> Somas::GetRefTensorsInContiguousList() {
  // key: refnode input value: refnode output
  std::map<size_t, size_t> ref_tensors_in_contiguous_map;
  for (auto ref_node_list : union_tensors_list_) {
    // Count contiguous tensors in ref list
    auto contiguous_in_ref_list = std::count_if(ref_node_list.begin(), ref_node_list.end(),
                                                [this](size_t tid) { return tensors_list_[tid]->contiguous_; });
    // Keep info about contiguous and check for errors
    if (ref_node_list.size() > kRefNodeTensorNum && contiguous_in_ref_list > 0) {
      MS_LOG(WARNING) << "Ref node of size greater than two with at least one contiguous tensor in";
    }
    if (ref_node_list.size() == kRefNodeTensorNum && contiguous_in_ref_list == 1) {
      MS_LOG(WARNING) << "Ref node of size two with only one contiguous tensor" << ref_node_list[0] << ":"
                      << tensors_list_[ref_node_list[0]]->contiguous_ << ", " << ref_node_list[1] << ":"
                      << tensors_list_[ref_node_list[1]]->contiguous_;
    }
    if (ref_node_list.size() == kRefNodeTensorNum && LongToSize(contiguous_in_ref_list) == kRefNodeTensorNum) {
      ref_tensors_in_contiguous_map[ref_node_list[0]] = ref_node_list[1];
    }
  }
  return ref_tensors_in_contiguous_map;
}

void Somas::UpdateContiguousTensorsOffset(const std::map<size_t, size_t> &contiguous_ref_list_map) {
  // Handle contiguous ref node
  for (auto ref_list_pair : contiguous_ref_list_map) {
    size_t index_first = ref_list_pair.first;
    size_t index_second = ref_list_pair.second;
    for (size_t x = 0; x < contiguous_tensors_list_[index_second].size(); x++) {
      tensors_list_[contiguous_tensors_list_[index_second][x]]->offset_ =
        tensors_list_[contiguous_tensors_list_[index_first][x]]->offset_;
      tensors_list_[contiguous_tensors_list_[index_second][x]]->aligned_size_ =
        tensors_list_[contiguous_tensors_list_[index_first][x]]->aligned_size_;
    }
  }
}

void Somas::UpdateUnionTensorsOffset() {
  // Set offset for rest of ref node list (ignored by solver due to ref node preprocessing)
  for (auto ref_node_list : union_tensors_list_) {
    for (size_t i = 1; i < ref_node_list.size(); ++i) {
      tensors_list_[ref_node_list[i]]->offset_ = tensors_list_[ref_node_list[0]]->offset_;
      tensors_list_[ref_node_list[i]]->aligned_size_ = tensors_list_[ref_node_list[0]]->aligned_size_;
    }
  }
}

// Disjoint-set
size_t find_father(std::vector<size_t> *father, size_t x) {
  MS_EXCEPTION_IF_NULL(father);
  if (x >= father->size()) {
    MS_LOG(EXCEPTION) << "Index " << x << " out of range " << father->size();
  }
  if (x == (*father)[x]) {
    return x;
  }
  (*father)[x] = find_father(father, (*father)[x]);
  return (*father)[x];
}

void Somas::UpdateUnionTensorsConflict() {
  // Keep all constraints for first tensor in list
  MS_EXCEPTION_IF_NULL(tensors_list_.back());
  size_t cnt = tensors_list_.back()->GetId() + 1;
  std::vector<size_t> father;
  for (size_t i = 0; i < cnt; i++) {
    father.push_back(i);
  }
  for (auto union_node_list : union_tensors_list_) {
    if (union_node_list.empty()) {
      MS_LOG(EXCEPTION) << "union node list is empty.";
    }
    size_t tid_0 = union_node_list[0];
    for (size_t i = 1; i < union_node_list.size(); ++i) {
      size_t tid_1 = union_node_list[i];
      father[find_father(&father, tid_1)] = find_father(&father, tid_0);
    }
  }

  std::map<size_t, size_t> kv;
  std::vector<vector<size_t>> tmp_union;
  for (const auto &union_node_list : union_tensors_list_) {
    for (size_t tid : union_node_list) {
      size_t fa = find_father(&father, tid);
      if (kv.find(fa) == kv.end()) {
        tmp_union.emplace_back();
        kv.emplace(fa, tmp_union.size() - 1);
      }
      tmp_union[kv.at(fa)].push_back(tid);
    }
  }

  union_tensors_list_ = tmp_union;

  for (auto union_node_list : union_tensors_list_) {
    size_t tid_0 = union_node_list[0];
    for (const SomasTensorPtr &tensor : tensors_list_) {
      for (size_t tid : union_node_list) {
        if (!reuse_matrix_[tid].IsBitTrue(tensor->GetId())) {
          reuse_matrix_[tid_0].SetBitFalse(tensor->GetId());
          reuse_matrix_[tensor->GetId()].SetBitFalse(tid_0);
          break;
        }
      }
    }
    // if union_tensors_list has a zero, when need set all union_tensors in this list is zero
    bool zero_flag = std::any_of(union_node_list.begin(), union_node_list.end(),
                                 [this](size_t i) { return tensors_list_[i]->aligned_size_ == 0; });
    if (zero_flag) {
      for (size_t i : union_node_list) {
        tensors_list_[i]->aligned_size_ = 0;
      }
    }
    // Set rest to size 0, so that solver ignores them (if not contiguous)
    for (size_t i = 1; i < union_node_list.size(); ++i) {
      if (!tensors_list_[union_node_list[i]]->contiguous_) {
        if (tensors_list_[union_node_list[i]]->aligned_size_ > tensors_list_[union_node_list[0]]->aligned_size_) {
          MS_LOG(WARNING) << "The aligned_size of union tensor " << tensors_list_[union_node_list[i]]->GetId()
                          << " is bigger than the aligned_size of union tensor "
                          << tensors_list_[union_node_list[0]]->GetId();
          tensors_list_[union_node_list[0]]->aligned_size_ = tensors_list_[union_node_list[i]]->aligned_size_;
        }
        tensors_list_[union_node_list[i]]->aligned_size_ = 0;
      }
    }
  }

  // solver should ignore union contiguous tensors.
  for (auto ref_list_pair : contiguous_list_with_ref_index_map_) {
    size_t index_second = ref_list_pair.second;
    for (size_t x : contiguous_tensors_list_.at(index_second)) {
      MS_EXCEPTION_IF_NULL(tensors_list_.at(x));
      tensors_list_.at(x)->aligned_size_ = 0;
    }
  }
}

std::string Somas::GetSplitName(const std::string &scope_name) {
  auto index = scope_name.rfind('/');
  if (index == std::string::npos) {
    return scope_name;
  } else {
    if (index < scope_name.size() - 1) {
      auto split_name = scope_name.substr(index + 1);
      return split_name;
    }
    return scope_name;
  }
}

std::string Somas::SomasInfo(bool calc_hash) const {
  std::ostringstream oss;
  if (!calc_hash) {
    DumpParameters(oss);
  }
  DumpTensors(oss);
  DumpNodes(oss);

  oss << "\n\nAll Union Tensors Info:\n\n";
  for (const auto &ref_in_out : union_tensors_list_) {
    oss << "union tensors: [";
    for (const auto &item : ref_in_out) {
      oss << "%" << item << "T ";
    }
    oss << "]\n";
  }

  oss << "\n\nAll Original Contiguous Tensors Info:\n\n";
  for (const auto &contiguous : contiguous_tensors_list_) {
    oss << "contiguous tensors: [";
    for (const auto &item : contiguous) {
      oss << "%" << item << "T ";
    }
    oss << "]\n";
  }

  oss << "\n\nAll Processed Contiguous Tensors Info:\n\n";
  for (const auto &contiguous : processed_contiguous_tensors_list_) {
    oss << "contiguous tensors: [";
    for (const auto &item : contiguous) {
      oss << "%" << item << "T ";
    }
    oss << "]\n";
  }

  oss << "\n\nAll Stream Groups:\n\n";
  for (const auto &stream_group : streams_groups_) {
    for (const auto &stream : stream_group) {
      oss << "stm" << stream << " ";
    }
    oss << "\n";
  }

  oss << "\n\nAll Merged Blocks:\n\n";
  oss << "start_offset:"
      << "\tsize:\n";
  for (const auto &merged_block : dump_merged_blocks_) {
    oss << merged_block.first << "\t" << merged_block.second << "\n";
  }
  oss << "\nTotal Memory Size after reused:" << reused_memory_size_;
  return oss.str();
}

void Somas::DumpNodes(std::ostringstream &oss) const {
  oss << "\n\nAll Nodes:\n\n";
  for (const auto &node : nodes_list_) {
    MS_EXCEPTION_IF_NULL(node);
    auto scope_name = node->scope_full_name_;
    std::string split_name = GetSplitName(scope_name);
    oss << "$" << node->GetId() << "\t" << split_name << "\t" << static_cast<int>(node->GetType()) << "\t";
    auto input_num = node->input_tensors_.size() + node->input_parameters_map_.size();
    oss << "inputs[";
    size_t tensor_index = 0;
    for (size_t input_index = 0; input_index < input_num; input_index++) {
      auto iter = node->input_parameters_map_.find(input_index);
      if (iter != node->input_parameters_map_.end()) {
        oss << "%" << iter->second->id_ << "P"
            << ", ";
      } else {
        oss << "%" << node->input_tensors_[tensor_index]->GetId() << "T"
            << ", ";
        tensor_index++;
      }
    }
    oss << "]";

    oss << "\toutputs[";
    for (const auto &out : node->output_tensors_) {
      MS_EXCEPTION_IF_NULL(out);
      oss << "%" << out->GetId() << "T"
          << ", ";
    }
    oss << "]";

    oss << "\tworkspace[";
    for (const auto &wk : node->workspace_tensors_) {
      MS_EXCEPTION_IF_NULL(wk);
      oss << "%" << wk->GetId() << "T"
          << ", ";
    }
    oss << "]";

    oss << "\tctrl_inputs[";
    for (const auto &ctrl_in : node->control_input_tensors_) {
      MS_EXCEPTION_IF_NULL(ctrl_in);
      oss << "%" << ctrl_in->GetId() << "CT"
          << ", ";
    }
    oss << "]";

    oss << "\tctrl_outputs[";
    for (const auto &ctrl_out : node->control_output_tensors_) {
      MS_EXCEPTION_IF_NULL(ctrl_out);
      oss << "%" << ctrl_out->GetId() << "CT"
          << ", ";
    }
    oss << "]";

    oss << "\tstreamID["
        << "@" << node->GetStreamId() << "]\n";
  }
}

void Somas::DumpTensors(std::ostringstream &oss) const {
  oss << "\n\nAll Tensors:\n\n";
  oss << "index:"
      << "\taligned_size:"
      << "\toriginal_size:"
      << "\toffset:"
      << "\ttype:"
      << "\tlifelong:"
      << "\tlife_start:"
      << "\tlife_end:"
      << "\tsource node name:\n";
  std::vector<SomasTensorPtr> dump_tensors_list;
  dump_tensors_list.insert(dump_tensors_list.end(), tensors_list_.begin(), tensors_list_.end());
  dump_tensors_list.insert(dump_tensors_list.end(), control_tensors_list_.begin(), control_tensors_list_.end());
  for (const auto &tensor : dump_tensors_list) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto node = GetSomasNode(tensor->GetSourceNodeId());
    MS_EXCEPTION_IF_NULL(node);
    auto scope_name = node->scope_full_name_;
    std::string split_name = GetSplitName(scope_name);
    std::string dump_tensor_str = "T";
    if (tensor->GetTypeString() == "Control") {
      dump_tensor_str = "CT";
    }
    oss << "%" << tensor->GetId() << dump_tensor_str << "\t"
        << "#" << tensor->GetAlignedSize() << "S"
        << "\t"
        << "#" << tensor->GetOriginalSize() << "S"
        << "\t"
        << "&" << tensor->GetOffset() << ""
        << "\t" << tensor->GetTypeString() << "\t" << tensor->GetLifelongString() << "\t" << tensor->lifetime_.start_
        << "\t" << tensor->lifetime_.end_ << "\t" << split_name << "\n";
  }
}

void Somas::DumpParameters(std::ostringstream &oss) const {
  oss << "All Parameters:\n\n";
  oss << "index:"
      << "\tsize:"
      << "\tsource node name:"
      << "\tnode out index:\n";

  for (const auto &param : parameters_list_) {
    MS_EXCEPTION_IF_NULL(param);
    oss << "%" << param->id_ << "P"
        << "\t"
        << "#" << param->size_ << "S"
        << "\t" << param->source_node_name_ << "\t" << param->output_index_ << "\n";
  }
}

void Somas::DumpSomasModelInfo(const string &tag, uint32_t graph_id) const {
#ifndef ENABLE_SECURITY
  if (save_debug_info_) {
    std::string file_path =
      GetSaveGraphsPathName("/" + device_name_ + "_" + tag + "_" + std::to_string(graph_id) + ".ir", debug_info_path_);
    (void)Common::SaveStringToFile(file_path, SomasInfo());
  }
#endif
}

std::string Somas::Offline() const {
  std::ostringstream oss;

  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->IsOutputOnly() || tensor->type_ == TensorType::kUnion) {
      oss << "Somas EDGE ERROR src=n" << tensor->GetSourceNodeId() << ", srcstm=" << tensor->GetSourceStreamId()
          << ", dst=nc"
          << ", dststm=nc"
          << ", workspace=0, size=" << tensor->GetOriginalSize()
          << ", lifelong=" << static_cast<int>(tensor->lifelong_value_) << ", tid=" << tensor->GetId()
          << ", start=" << tensor->lifetime_.start_ << ", end=" << tensor->lifetime_.end_ << std::endl;
    } else {
      std::map<size_t, size_t> dest_node_streams;
      for (const auto &dest_node : tensor->destination_nodes_) {
        auto node = GetSomasNode(dest_node);
        MS_EXCEPTION_IF_NULL(node);
        (void)dest_node_streams.emplace(dest_node, node->GetStreamId());
      }

      for (const auto &dest_info : dest_node_streams) {
        oss << "Somas EDGE src=n" << tensor->GetSourceNodeId() << ", srcstm=" << tensor->GetSourceStreamId()
            << ", dst=n" << dest_info.first << ", dststm=" << dest_info.second
            << ", workspace=" << static_cast<int>(tensor->type_ == kWorkspace) << ", size=" << tensor->GetOriginalSize()
            << ", lifelong=" << static_cast<int>(tensor->lifelong_value_) << ", tid=" << tensor->GetId()
            << ", start=" << tensor->lifetime_.start_ << ", end=" << tensor->lifetime_.end_ << std::endl;
      }
    }
  }
  for (const vector<size_t> &tList : contiguous_tensors_list_) {
    oss << "Somas CONTIGUOUS";
    for (size_t tid : tList) {
      oss << " " << tid;
    }
    oss << std::endl;
  }
  for (const auto &group : streams_groups_) {
    oss << "Somas GROUP";
    for (int64_t sid : group) {
      oss << " " << sid;
    }
    oss << std::endl;
  }
  return oss.str();
}

void Somas::DumpOfflineIR(const string &filename) const {
  MS_LOG(INFO) << "Printing somas-log-from-graph log: " << filename;
  (void)Common::SaveStringToFile(filename, Offline());
}

size_t Somas::CalcLowerBound() const {
  size_t max_node_id = std::accumulate(tensors_list_.begin(), tensors_list_.end(), 0, [](size_t max_id, auto tensor) {
    return std::max(max_id, tensor->lifetime_.end_);
  });

  std::map<size_t, size_t> lifetime_lb;
  for (size_t time = 0; time <= max_node_id; time++) {
    lifetime_lb[time] = 0;
  }

  size_t lower;
  size_t upper;
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor->lifelong_value_ == kLifeLongGraphAll) {
      lower = 0;
      upper = max_node_id;
    } else {
      lower = tensor->lifetime_.start_;
      upper = tensor->lifetime_.end_;
    }

    for (size_t time = lower; time <= upper; time++) {
      lifetime_lb[time] += tensor->GetAlignedSize();
    }
  }

  size_t max_lifetime = 0;
  for (size_t time = 0; time <= max_node_id; time++) {
    if (max_lifetime < lifetime_lb[time]) {
      max_lifetime = lifetime_lb[time];
    }
  }
  return max_lifetime;
}

void Somas::GenGraphStatisticInfo() {
  lower_bound_ = CalcLowerBound();
  for (const auto &tensor : tensors_list_) {
    MS_EXCEPTION_IF_NULL(tensor);
    upper_bound_ += tensor->aligned_size_;
    if (tensor->type_ == kWorkspace) {
      workspace_total_size_ += tensor->aligned_size_;
    }
    if (tensor->lifelong_value_ == kLifeLongGraphAll) {
      lifelong_all_total_size_ += tensor->aligned_size_;
    } else if (tensor->lifelong_value_ == kLifeLongGraphStart) {
      lifelong_start_total_size_ += tensor->aligned_size_;
    } else if (tensor->lifelong_value_ == kLifeLongGraphEnd) {
      lifelong_end_total_size_ += tensor->aligned_size_;
    }
  }

  const double giga = 1024. * 1024. * 1024.;
  MS_LOG(INFO) << "Lower Bound: " << lower_bound_ << " (" << static_cast<double>(lower_bound_) / giga
               << " GB), Upper Bound: " << upper_bound_ << " (" << static_cast<double>(upper_bound_) / giga << " GB)";

  MS_LOG(INFO) << "\nTotal Dynamic Size (Upper Bound):\t" << upper_bound_ << "\n"
               << "Theoretical Optimal Size (Lower Bound):\t" << lower_bound_ << "\n"
               << "Total Workspace Size:\t" << workspace_total_size_ << "\n"
               << "Total Communication Input Tensor Size:\t" << comm_input_total_size_ << "\n"
               << "Total Communication Output Tensor Size:\t" << comm_output_total_size_ << "\n"
               << "Total LifeLong All Tensor Size:\t" << lifelong_all_total_size_ << "\n"
               << "Total LifeLong Start Tensor Size:\t" << lifelong_start_total_size_ << "\n"
               << "Total LifeLong End Tensor Size:\t" << lifelong_end_total_size_ << "\n"
               << "Reused Size(Allocate Size):\t" << reused_memory_size_ << "\n\n\n";
}

std::vector<std::pair<size_t, size_t>> Somas::GetNodeOutputSomasResult(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto key = node.get();
  auto iter = nodes_map_.find(key);
  std::vector<std::pair<size_t, size_t>> output_somas_result;
  if (iter != nodes_map_.end()) {
    auto &somas_node = iter->second.at(0);
    MS_EXCEPTION_IF_NULL(somas_node);
    std::transform(somas_node->output_tensors_.cbegin(), somas_node->output_tensors_.cend(),
                   std::back_inserter(output_somas_result),
                   [](const SomasTensorPtr &tensor) { return std::make_pair(tensor->offset_, tensor->aligned_size_); });
  } else {
    MS_LOG(EXCEPTION) << "node [" << common::AnfAlgo::GetCNodeName(node) << "] don't exist in nodes_map";
  }
  return output_somas_result;
}

std::vector<std::pair<size_t, size_t>> Somas::GetNodeWorkSpaceSomasResult(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto key = node.get();
  auto iter = nodes_map_.find(key);
  std::vector<std::pair<size_t, size_t>> workspace_somas_result;
  if (iter != nodes_map_.end()) {
    auto &somas_node = iter->second.at(0);
    MS_EXCEPTION_IF_NULL(somas_node);
    std::transform(somas_node->workspace_tensors_.cbegin(), somas_node->workspace_tensors_.cend(),
                   std::back_inserter(workspace_somas_result),
                   [](const SomasTensorPtr &tensor) { return std::make_pair(tensor->offset_, tensor->aligned_size_); });
  } else {
    MS_LOG(EXCEPTION) << "node [" << common::AnfAlgo::GetCNodeName(node) << "] don't exist in nodes_map";
  }
  return workspace_somas_result;
}

SomasStreamPtr Somas::GetSomasStream(size_t stream_id) const {
  auto it = streams_map_.find(stream_id);
  if (it != streams_map_.end()) {
    return (*it).second;
  } else {
    MS_LOG(ERROR) << "Can't find somas stream for stream " << stream_id;
    return nullptr;
  }
}

SomasNodePtr Somas::GetSomasNode(size_t node_id) const {
  if (node_id >= nodes_list_.size()) {
    return nullptr;
  } else {
    return nodes_list_[node_id];
  }
}

common::KernelWithIndex Somas::GetVisitKernelWithReturnType(const AnfNodePtr &ori_node, size_t ori_index) {
  auto prenode = common::AnfAlgo::VisitKernelWithReturnType(ori_node, ori_index, false);
  while (prenode.first->isa<CNode>() && nodes_map_.find(prenode.first.get()) == nodes_map_.end()) {
    auto anf_node = prenode.first;
    auto cnode = anf_node->cast<CNodePtr>();
    if (!common::AnfAlgo::IsNopNode(cnode)) {
      MS_LOG(EXCEPTION) << "Node[" << ori_node->fullname_with_scope() << "] find input node["
                        << cnode->fullname_with_scope() << "] doesn't exist in nodes_map and is not a nop node!!!!";
    }
    prenode = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, false);
  }
  return prenode;
}

SomasManager &SomasManager::Instance() {
  static SomasManager instance{};
  return instance;
}

void SomasManager::Clear() { base_map_.clear(); }
}  // namespace somas
}  // namespace mindspore
