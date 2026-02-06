/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "include/frontend/parallel/parallel_strategy_checkpoint.h"

#include <pybind11/operators.h>

#include <algorithm>
#include <functional>
#include <fstream>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "mindspore/core/include/utils/distributed_meta.h"
#include "ir/dtype/ref.h"
#include "ir/dtype/tensor_type.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/utils/common.h"
#include "utils/convert_utils.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"

namespace py = pybind11;
namespace mindspore {
namespace parallel {
const uint32_t JSON_SUFFIX_LENGTH = 5;
StrategyCheckpoint &StrategyCheckpoint::GetInstance() {
  static StrategyCheckpoint instance = StrategyCheckpoint();
  if (ParallelContext::GetInstance() != nullptr) {
    instance.load_file_ = ParallelContext::GetInstance()->strategy_ckpt_load_file();
    instance.load_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_load_file().empty();
    instance.save_file_ = ParallelContext::GetInstance()->strategy_ckpt_save_file();
    instance.save_checkpoint_on_ = !ParallelContext::GetInstance()->strategy_ckpt_save_file().empty();
    instance.group_info_save_file_ = ParallelContext::GetInstance()->group_ckpt_save_file();
    instance.group_info_save_on_ = !ParallelContext::GetInstance()->group_ckpt_save_file().empty();
    instance.load_format_json_ = instance.load_file_.size() >= JSON_SUFFIX_LENGTH &&
                                 instance.load_file_.substr(instance.load_file_.size() - JSON_SUFFIX_LENGTH) == ".json";
    instance.save_format_json_ = instance.save_file_.size() >= JSON_SUFFIX_LENGTH &&
                                 instance.save_file_.substr(instance.save_file_.size() - JSON_SUFFIX_LENGTH) == ".json";
    instance.auto_op_strategy_file_ = ParallelContext::GetInstance()->strategy_json_config_file_path();
    instance.auto_op_strategy_file_type_ = ParallelContext::GetInstance()->strategy_json_config_file_type();
    instance.load_auto_op_strategy_on_ = (!ParallelContext::GetInstance()->strategy_json_config_file_path().empty()) &&
                                         (instance.auto_op_strategy_file_type_.compare("LOAD") == 0);
    instance.save_auto_op_strategy_on_ = (!ParallelContext::GetInstance()->strategy_json_config_file_path().empty()) &&
                                         (instance.auto_op_strategy_file_type_.compare("SAVE") == 0);
  }
  return instance;
}

bool StrategyCheckpoint::CheckPath(const std::string path) const {
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "The checkpoit path " << path << " is too long";
    return false;
  }
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return false;
  }
  return true;
}

bool StrategyCheckpoint::CheckPointExit(const std::string path) const {
  std::ifstream fin(path);
  if (fin) {
    return true;
  }
  return false;
}

Status StrategyCheckpoint::LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const {
  MS_EXCEPTION_IF_NULL(group_info_map);
  if (!CheckPath(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  if (!CheckPointExit(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is not found";
  }
  straspb::ParallelGroupMap parallel_group_map;
  std::fstream input(file, std::ios::in | std::ios::binary);
  if (!parallel_group_map.ParseFromIstream(&input)) {
    MS_LOG(ERROR) << "Load strategy file failed";
    return FAILED;
  }
  input.close();

  size_t group_num = LongToSize(parallel_group_map.parallel_group_item_size());
  for (size_t i = 0; i < group_num; ++i) {
    straspb::ParallelGroupItem parallel_group_item = parallel_group_map.parallel_group_item(SizeToInt(i));
    std::string group_name = parallel_group_item.group_name();

    straspb::ParallelGroupRanks parallel_group_ranks = parallel_group_item.parallel_group_ranks();
    size_t rank_num = LongToSize(parallel_group_ranks.dim_size());
    std::vector<uint32_t> ranks;
    for (size_t j = 0; j < rank_num; ++j) {
      uint32_t rank = parallel_group_ranks.dim(SizeToInt(j));
      ranks.push_back(rank);
    }

    std::pair<std::string, std::vector<uint32_t>> group = std::make_pair(group_name, ranks);
    group_info_map->push_back(group);
  }

  return SUCCESS;
}

Status StrategyCheckpoint::Load(StrategyMap *strategy_map) {
  if (strategy_map == nullptr) {
    MS_LOG(EXCEPTION) << "Failure:strategy_map is nullptr";
  }
  if (!CheckPath(load_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  if (!CheckPointExit(load_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is not found";
  }
  if (load_format_json_) {
    std::fstream input(load_file_, std::ios::in);
    nlohmann::json stra_ckpt_info_j;
    input >> stra_ckpt_info_j;
    strategy_checkpoint_info_.FromJson(stra_ckpt_info_j);
  } else {
    straspb::ParallelStrategyMap parallel_strategy_map;
    std::fstream input(load_file_, std::ios::in | std::ios::binary);
    if (!parallel_strategy_map.ParseFromIstream(&input)) {
      MS_LOG(ERROR) << "Load strategy file failed";
      return FAILED;
    }
    input.close();
    strategy_checkpoint_info_.from_protobuf(parallel_strategy_map);
  }
  *strategy_map = strategy_checkpoint_info_.strategy_map();
  current_stage_ = SizeToLong(strategy_checkpoint_info_.current_stage());
  return SUCCESS;
}

std::string StrategyInfo::ToString() const {
  std::ostringstream oss;
  oss << "dev_matrix = " << VectorUtils::PrintVector(dev_matrix_)
      << ", tensor_map = " << VectorUtils::PrintNestedVector(tensor_map_)
      << ", global_tensor_shape = " << VectorUtils::PrintVector(tensor_shape_) << ", tensor_type = " << tensor_type_
      << ", field = " << field_ << ", opt_weight_shard_step = " << opt_weight_shard_step_
      << ", opt_weight_shard_size = " << opt_weight_shard_size_
      << ", param_split_shape = " << VectorUtils::PrintVector(param_split_shape_)
      << ", indices_offset = " << VectorUtils::PrintVector(indices_offset_) << ", stage_id = " << stage_id_
      << ", pipeline_stages = " << pipeline_stages_ << ", rank_list = " << VectorUtils::PrintVector(rank_list_);
  return oss.str();
}

std::shared_ptr<StrategyLayout> StrategyLayout::GetInstance() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (layout_instance_ == nullptr) {
      MS_LOG(INFO) << "Create global network parameter strategy.";
      layout_instance_ = std::make_shared<StrategyLayout>();
    }
  });
  return layout_instance_;
}

void StrategyLayout::SetParamStageIdRanks(const std::string &param_name, const int64_t &stage_id,
                                          const std::vector<int64_t> &rank_list) {
  auto &pair = parameter_stage_ranks_map_[param_name];
  pair.first = stage_id;
  pair.second = rank_list;
}

mindspore::HashMap<std::string, std::pair<int64_t, std::vector<int64_t>>> StrategyLayout::ParamStageIdRankId() const {
  return parameter_stage_ranks_map_;
}

void StrategyLayout::SetParamGlobalShape(const AnfNodePtr &parameter) {
  std::string param_name = parameter->ToString();
  const auto &param_shape = parameter->Shape();
  if (param_shape == nullptr) {
    MS_LOG(WARNING) << "Shape is nullptr for parameter: " << param_name;
    return;
  }
  auto real_shape = std::dynamic_pointer_cast<abstract::Shape>(param_shape);
  if (real_shape == nullptr) {
    MS_LOG(WARNING) << "Unsupported shape type for parameter: " << param_name;
    return;
  }
  std::vector<int64_t> origin_shape(real_shape->GetShapeVector().begin(), real_shape->GetShapeVector().end());
  param_shape_map_[param_name] = origin_shape;
}

std::vector<int64_t> StrategyLayout::ParamGlobalShape(const std::string &param_name) const {
  const auto &it = param_shape_map_.find(param_name);
  if (it == param_shape_map_.end()) {
    MS_LOG(EXCEPTION) << "Param shape not found for: " << param_name;
  }
  return it->second;
}

void StrategyLayout::SetParamType(const AnfNodePtr &parameter) {
  std::string param_name = parameter->ToString();
  auto type = parameter->Type();
  if (type == nullptr) {
    MS_LOG(WARNING) << "Parameter type is null: " << param_name;
    return;
  }
  if (type->isa<RefType>()) {
    type = type->cast<RefTypePtr>();
  }
  if (type->isa<TensorType>()) {
    type = type->cast<TensorTypePtr>()->element();
  }
  std::string param_type = type->ToReprString();
  param_type_map_[param_name] = param_type;
}

std::string StrategyLayout::ParamType(const std::string &param_name) const {
  const auto &it = param_type_map_.find(param_name);
  if (it == param_type_map_.end()) {
    MS_LOG(EXCEPTION) << "Param type not found for: " << param_name;
  }
  return it->second;
}

void StrategyLayout::SetNetworkLayoutSaved(const std::string &name) { network_state_map_[name] = true; }

bool StrategyLayout::NetworkLayoutSaved(const std::string &name) const {
  const auto &it = network_state_map_.find(name);
  if (it == network_state_map_.end()) {
    MS_LOG(INFO) << "Strategy has not been saved: " << name;
    return false;
  }
  return it->second;
}

void StrategyLayout::SaveParamStraInfo(uint32_t rank_id, const ParamStrategyMap &param_map) {
  auto &entry = global_rank_stra_map_[rank_id];
  for (const auto &kv : param_map) {
    entry[kv.first] = kv.second;
  }
  layout_sorted_ = false;
}

SortedParamVec StrategyLayout::SortParamStrategy(const ParamStrategyMap &param_stra_map) const {
  SortedParamVec sorted_params;
  for (const auto &kv : param_stra_map) {
    (void)sorted_params.emplace_back(kv.first, kv.second);
  }
  enum class ParamCat { Origin, AdamM, AdamV, Moments, AccuGrads };
  auto make_sort_key = [](const std::string &key) -> std::tuple<std::string_view, ParamCat> {
    ParamCat cat = ParamCat::Origin;
    std::string_view base = key;
    constexpr std::string_view kAdamM = "adam_m.";
    constexpr std::string_view kAdamV = "adam_v.";
    constexpr std::string_view kMom = "moments.";
    constexpr std::string_view kGrads = "accu_grads.";
    if (base.rfind(kAdamM, 0) == 0) {
      cat = ParamCat::AdamM;
      base.remove_prefix(kAdamM.size());
    }
    if (base.rfind(kAdamV, 0) == 0) {
      cat = ParamCat::AdamV;
      base.remove_prefix(kAdamV.size());
    }
    if (base.rfind(kMom, 0) == 0) {
      cat = ParamCat::Moments;
      base.remove_prefix(kMom.size());
    }
    if (base.rfind(kGrads, 0) == 0) {
      cat = ParamCat::AccuGrads;
      base.remove_prefix(kGrads.size());
    }
    return {base, cat};
  };
  std::sort(sorted_params.begin(), sorted_params.end(),
            [&](const auto &lhs, const auto &rhs) { return make_sort_key(lhs.first) < make_sort_key(rhs.first); });
  return sorted_params;
}

void StrategyLayout::SortCurNetGlobalLayout() {
  if (layout_sorted_ == true) {
    return;
  }

  if (global_rank_stra_map_.empty()) {
    MS_LOG(DEBUG) << "No valid strategy.";
    layout_sorted_ = true;
    return;
  }

  std::vector<std::pair<int64_t, SortedParamVec>> sorted_items;
  for (const auto &items : global_rank_stra_map_) {
    SortedParamVec inner_list = SortParamStrategy(items.second);
    (void)sorted_items.emplace_back(items.first, inner_list);
  }
  std::sort(sorted_items.begin(), sorted_items.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
  global_layout_list_.reserve(sorted_items.size());
  for (auto &item : sorted_items) {
    (void)global_layout_list_.emplace_back(std::move(item.second));
  }
  const uint32_t rank_id = DistributedMeta::GetInstance()->global_rank_id();
  (void)local_layout_list_.emplace_back(global_layout_list_[rank_id]);
  layout_sorted_ = true;
}

void StrategyLayout::ClearCurNet() {
  parameter_stage_ranks_map_.clear();
  param_shape_map_.clear();
  param_type_map_.clear();
  global_layout_list_.clear();
  local_layout_list_.clear();
  global_rank_stra_map_.clear();
  layout_sorted_ = false;
  compile_phase_.clear();
}

void StrategyLayout::SaveNetworkGlobalLayout() {
  SortCurNetGlobalLayout();
  std::string phase = CellPhase();
  if (!phase.empty()) {
    (void)global_network_rank_stra_list_.emplace_back(phase, global_layout_list_);
    (void)local_network_rank_stra_list_.emplace_back(phase, local_layout_list_);
  }
  SetNetworkLayoutSaved(phase);
  MS_LOG(INFO) << "Successfully saved strategy information of " << phase;
  ClearCurNet();
}

void StrategyLayout::EnsureSorted() const {
  if (!layout_sorted_) {
    const_cast<StrategyLayout *>(this)->SortCurNetGlobalLayout();
  }
}

std::string StrategyLayout::DebugString(const SortedRankParamVec &layout) const {
  std::stringstream ss;
  ss << "\n";
  for (size_t i = 0; i < layout.size(); ++i) {
    ss << "  rank " << i << ": {\n";
    const auto inner_list = layout.at(i);
    for (const auto &item : inner_list) {
      ss << "    " << item.first << ": { " << item.second.ToString() << " };\n";
    }
    ss << "  }\n\n";
  }
  return ss.str();
}

std::string StrategyLayout::CurNetGlobalStraInfo() const {
  EnsureSorted();
  return DebugString(global_layout_list_);
}

std::string StrategyLayout::CurNetLocalStraInfo() const {
  EnsureSorted();
  return DebugString(local_layout_list_);
}

py::dict StrategyLayout::ConvertNetStraToPyDict(const SortedNetRankParamVec &layout) const {
  py::dict result;
  for (auto it = layout.begin(); it != layout.end(); ++it) {
    const auto &compile_phase = it->first;
    const auto &cur_net_stra = it->second;
    py::dict cur_net_layout;
    uint32_t rank_id = 0;
    for (const auto &inner_list : cur_net_stra) {
      py::dict param_dict;
      for (const auto &param_entry : inner_list) {
        param_dict[param_entry.first.c_str()] = param_entry.second;
      }
      cur_net_layout[py::int_(rank_id++)] = param_dict;
    }
    result[compile_phase.c_str()] = cur_net_layout;
  }
  return result;
}

py::dict StrategyLayout::global_network_layout() const {
  return ConvertNetStraToPyDict(global_network_rank_stra_list_);
}

py::dict StrategyLayout::local_network_layout() const { return ConvertNetStraToPyDict(local_network_rank_stra_list_); }

void StrategyLayout::clear_strategy_metadata() {
  ClearCurNet();
  global_network_rank_stra_list_.clear();
  local_network_rank_stra_list_.clear();
  network_state_map_.clear();
}

static inline std::string_view StripKnownPrefix(std::string_view name) {
  constexpr std::string_view kAdamM = "adam_m.";
  constexpr std::string_view kAdamV = "adam_v.";
  constexpr std::string_view kMom = "moments.";
  constexpr std::string_view kAccu = "accu_grads.";
  if (name.rfind(kAdamM, 0) == 0) {
    return name.substr(kAdamM.size());
  }
  if (name.rfind(kAdamV, 0) == 0) {
    return name.substr(kAdamV.size());
  }
  if (name.rfind(kMom, 0) == 0) {
    return name.substr(kMom.size());
  }
  if (name.rfind(kAccu, 0) == 0) {
    return name.substr(kAccu.size());
  }
  return name;
}

std::optional<std::reference_wrapper<const std::pair<int64_t, std::vector<int64_t>>>> MatchParamBySuffix(
  const std::string &param_full_name,
  const mindspore::HashMap<std::string, std::pair<int64_t, std::vector<int64_t>>> &param_map) {
  std::string_view stripped = param_full_name;
  const auto &is_pp_interleave = ParallelContext::GetInstance()->pipeline_interleave();
  if (is_pp_interleave) {
    stripped = StripKnownPrefix(param_full_name);
  }
  for (const auto &kv : param_map) {
    const std::string &key = kv.first;
    if (stripped == key) {
      return kv.second;
    }
  }
  return std::nullopt;
}

void StrategyCheckpoint::HandleEmptyParallelLayout() {
  MS_LOG(WARNING)
    << "The network parameter is None. Possibly due to empty net or 'stand_alone'/'data_parallel' mode. Current mode: "
    << parallel::ParallelContext::GetInstance()->parallel_mode();
  StrategyLayout::GetInstance()->SaveNetworkGlobalLayout();
}

bool StrategyCheckpoint::PipelineNotSupported() {
  auto parallel_context = ParallelContext::GetInstance();
  if (!parallel_context->pipeline_interleave() && parallel_context->pipeline_stage_split_num() > 1) {
    MS_LOG(WARNING) << "Training lacks @lazy_inline or prediction mode: pipeline parallel strategy info not available.";
    StrategyLayout::GetInstance()->SaveNetworkGlobalLayout();
    return true;
  }
  return false;
}

StrategyInfo StrategyCheckpoint::BuildStrategyInfo(const std::string &param_name,
                                                   const straspb::ParallelLayouts &layouts) const {
  StrategyInfo stra;
  for (const auto &dev_dim : layouts.dev_matrix(0).dim()) {
    stra.set_dev_matrix(dev_dim);
  }

  if (layouts.tensor_map_size() == 1) {
    const auto &tensor_map_vec = layouts.tensor_map(0);
    for (int i = 0; i < tensor_map_vec.dim_size(); ++i) {
      stra.set_tensor_map({LongToInt(tensor_map_vec.dim(i))});
    }
  } else {
    for (const auto &tensor_map_vec : layouts.tensor_map()) {
      std::vector<int64_t> one_map(tensor_map_vec.dim().size());
      std::transform(tensor_map_vec.dim().begin(), tensor_map_vec.dim().end(), one_map.begin(), LongToInt);
      stra.set_tensor_map(one_map);
    }
  }

  stra.set_tensor_shape(StrategyLayout::GetInstance()->ParamGlobalShape(param_name));
  stra.set_tensor_type(StrategyLayout::GetInstance()->ParamType(param_name));
  stra.set_field(layouts.field());

  const bool isgrad = param_name.rfind("accu_grads.", 0) == 0;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  const bool isZero1 = parallel_context->enable_parallel_optimizer() && !parallel_context->grad_accumulation_shard() &&
                       !parallel_context->zero3();
  if (isgrad && isZero1) {
    stra.set_opt_weight_shard_step(0);
    stra.set_opt_weight_shard_size(0);
  } else {
    stra.set_opt_weight_shard_step(layouts.opt_weight_shard_step());
    stra.set_opt_weight_shard_size(layouts.opt_weight_shard_size());
  }

  if (layouts.param_split_shape_size() > 0 && layouts.indices_offset_size() > 0) {
    for (const auto &dim : layouts.param_split_shape(0).dim()) {
      stra.set_param_split_shape(LongToInt(dim));
    }
    for (const auto &dim : layouts.indices_offset(0).dim()) {
      stra.set_indices_offset(LongToInt(dim));
    }
    if (stra.param_split_shape().size() != stra.indices_offset().size()) {
      MS_LOG(EXCEPTION) << "param_split_shape size != indices_offset size for " << param_name;
    }
  }

  return stra;
}

void StrategyCheckpoint::SaveStrategyParamLayout() {
  const auto &parallel_strategy_map = strategy_checkpoint_info_.to_protobuf();
  if (parallel_strategy_map.parallel_layout_item().empty()) {
    HandleEmptyParallelLayout();
    return;
  }

  if (PipelineNotSupported()) {
    return;
  }

  const auto &param_stage_rank_map = StrategyLayout::GetInstance()->ParamStageIdRankId();
  for (const auto &item : parallel_strategy_map.parallel_layout_item()) {
    const std::string &param_name = item.param_name();
    const auto &layouts = item.parallel_layouts();
    auto param_pair = MatchParamBySuffix(param_name, param_stage_rank_map);
    if (!param_pair.has_value()) {
      MS_LOG(EXCEPTION) << "No matching parameter name found for " << param_name;
    }
    const auto &[stage_id, rank_list] = param_pair.value().get();
    StrategyInfo stra = BuildStrategyInfo(param_name, layouts);
    stra.set_stage_id(stage_id);
    stra.set_pipeline_stages(ParallelContext::GetInstance()->pipeline_stage_split_num());
    stra.set_rank_list(rank_list);
    for (const auto &dev_id : rank_list) {
      StrategyLayout::GetInstance()->SaveParamStraInfo(dev_id, {{param_name, stra}});
    }
    if (stra.tensor_map().size() > 1) {
      MS_LOG(DEBUG) << "[Multi-slice] Param: " << param_name
                    << ", tensor_map = " << VectorUtils::PrintNestedVector(stra.tensor_map());
    }
  }

  MS_LOG(INFO) << "Cur network global layout: " << StrategyLayout::GetInstance()->CurNetGlobalStraInfo();
  MS_LOG(INFO) << "Cur network local layout: " << StrategyLayout::GetInstance()->CurNetLocalStraInfo();
  StrategyLayout::GetInstance()->SaveNetworkGlobalLayout();
}

Status StrategyCheckpoint::SaveOnline(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                      const ManualShapeMap &manual_shape_map) {
  strategy_checkpoint_info_.Init(strategy_map, tensor_info_map, manual_shape_map, ++current_stage_);
  const auto &compile_phase = StrategyLayout::GetInstance()->CellPhase();
  bool isValidPhase = !compile_phase.empty();
  bool isInit = StrategyLayout::GetInstance()->NetworkLayoutSaved(compile_phase);
  bool enable_save = StrategyLayout::GetInstance()->save_strategy_online();
  if (isValidPhase && !isInit && enable_save) {
    StrategyCheckpoint::GetInstance().SaveStrategyParamLayout();
  }
  return SUCCESS;
}

Status StrategyCheckpoint::Save(const StrategyMap &strategy_map, const TensorInfoMap &tensor_info_map,
                                const ManualShapeMap &manual_shape_map) {
  if (!CheckPath(save_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }

  strategy_checkpoint_info_.Init(strategy_map, tensor_info_map, manual_shape_map, ++current_stage_);
  if (save_format_json_) {
    auto stra_ckpt_info_j = strategy_checkpoint_info_.to_json();
    std::fstream output(save_file_, std::ios::out);
    stra_ckpt_info_j >> output;
    output.close();
  } else {
    auto parallel_strategy_map = strategy_checkpoint_info_.to_protobuf();
    std::fstream output(save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!parallel_strategy_map.SerializeToOstream(&output)) {
      MS_LOG(ERROR) << "Save strategy file failed";
      return FAILED;
    }
    output.close();
  }

  ChangeFileMode(save_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}

Status StrategyCheckpoint::SaveGroupInfo(const GroupInfoMap &group_info_map, const RankList &restore_rank_list) {
  straspb::ParallelGroupMap parallel_group_map;
  for (auto &group : group_info_map) {
    straspb::ParallelGroupItem *parallel_group_item = parallel_group_map.add_parallel_group_item();
    MS_EXCEPTION_IF_NULL(parallel_group_item);
    parallel_group_item->set_group_name(group.first);
    straspb::ParallelGroupRanks *parallel_group_ranks = parallel_group_item->mutable_parallel_group_ranks();
    MS_EXCEPTION_IF_NULL(parallel_group_ranks);
    for (auto &rank : group.second) {
      parallel_group_ranks->add_dim(rank);
    }
  }
  straspb::ParallelGroupRanks *ckpt_restore_rank_list = parallel_group_map.mutable_ckpt_restore_rank_list();
  for (auto &restore_rank : restore_rank_list) {
    ckpt_restore_rank_list->add_dim(static_cast<uint32_t>(LongToSize(restore_rank)));
  }

  if (!CheckPath(group_info_save_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  std::fstream output(group_info_save_file_, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!parallel_group_map.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save strategy file failed";
    return FAILED;
  }
  output.close();
  ChangeFileMode(group_info_save_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}

Status StrategyCheckpoint::LoadAutoOpStrategy(StrategyMap *strategy_map, StrategyMap *out_strategy_map,
                                              TensorLayoutValueMap *tensor_layout_map,
                                              TensorLayoutValueMap *out_tensor_layout_map,
                                              TensorLayoutValueMap *tensor_layout_newshape_map,
                                              TensorLayoutValueMap *out_tensor_layout_newshape_map) {
  if (strategy_map == nullptr) {
    MS_LOG(WARNING) << "Failure:strategy_map is nullptr";
    return FAILED;
  }
  if (!CheckPath(auto_op_strategy_file_)) {
    MS_LOG(WARNING) << "Op strategy json path is invalid";
    return FAILED;
  }
  if (!CheckPointExit(auto_op_strategy_file_)) {
    MS_LOG(WARNING) << "Op strategy json file is not found";
    return FAILED;
  }
  std::fstream input(auto_op_strategy_file_, std::ios::in);
  nlohmann::json stra_ckpt_info_j;
  input >> stra_ckpt_info_j;
  strategy_json_info_.FromJson(stra_ckpt_info_j);
  *strategy_map = strategy_json_info_.strategy_map();
  *out_strategy_map = strategy_json_info_.out_strategy_map();
  *tensor_layout_map = strategy_json_info_.tensor_layout_map();
  *out_tensor_layout_map = strategy_json_info_.out_tensor_layout_map();
  *tensor_layout_newshape_map = strategy_json_info_.tensor_layout_newshape_map();
  *out_tensor_layout_newshape_map = strategy_json_info_.out_tensor_layout_newshape_map();
  return SUCCESS;
}

Status StrategyCheckpoint::SaveAutoOpStrategy(const StrategyMap &strategy_map, const StrategyMap &out_strategy_map,
                                              const TensorLayoutValueMap &tensor_layout_map,
                                              const TensorLayoutValueMap &out_tensor_layout_map,
                                              const TensorLayoutValueMap &tensor_layout_newshape_map,
                                              const TensorLayoutValueMap &out_tensor_layout_newshape_map) {
  if (!CheckPath(auto_op_strategy_file_)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is invalid";
  }
  strategy_json_info_.Init(strategy_map, out_strategy_map, tensor_layout_map, out_tensor_layout_map,
                           tensor_layout_newshape_map, out_tensor_layout_newshape_map, 0);
  auto stra_ckpt_info_j = strategy_json_info_.to_json();
  std::fstream output(auto_op_strategy_file_, std::ios::out);
  stra_ckpt_info_j >> output;
  output.close();

  ChangeFileMode(auto_op_strategy_file_, S_IRUSR | S_IWUSR);
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
