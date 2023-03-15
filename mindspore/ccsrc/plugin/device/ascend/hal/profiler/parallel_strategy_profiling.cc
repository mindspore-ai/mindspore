/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"
#include <fstream>
#include "sys/stat.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/ascend/hal/profiler/options.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/utils/file_utils.h"
#include "google/protobuf/util/json_util.h"
#include "nlohmann/json.hpp"
#include "proto/profiling_parallel.pb.h"

#ifdef WITH_BACKEND
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/ps/util.h"
#endif

namespace mindspore {
namespace profiler {
namespace ascend {
std::shared_ptr<ParallelStrategy> ParallelStrategy::parallel_strategy_inst_ = std::make_shared<ParallelStrategy>();

std::shared_ptr<ParallelStrategy> &ParallelStrategy::GetInstance() {
  MS_EXCEPTION_IF_NULL(parallel_strategy_inst_);
  return parallel_strategy_inst_;
}

bool ParallelStrategy::IsProfilingParallelStrategyEnabled() {
  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->IsInitialized() || !ascend_profiler->GetParallelStrategyEnableFlag()) {
    MS_LOG(INFO) << "Profiling parallel strategy is disabled.";
    return false;
  }

#ifdef WITH_BACKEND
  if (ps::PSContext::instance()->is_server() || ps::PSContext::instance()->is_scheduler()) {
    MS_LOG(INFO) << "Current is ps server or ps scheduler, profiling parallel "
                    "strategy is disabled.";
    return false;
  }
#endif

  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if ((parallel_mode == parallel::kAutoParallel) || (parallel_mode == parallel::kSemiAutoParallel) ||
      (parallel_mode == parallel::kDataParallel)) {
    return true;
  }

  MS_LOG(INFO) << "Profiling parallel strategy is disabled, current parallel mode is " << parallel_mode;
  return false;
}

bool ParallelStrategy::StringToInt(std::string *str, int32_t *value) {
  try {
    *value = stoi(*str);
  } catch (std::invalid_argument &) {
    MS_LOG(ERROR) << "Catch invalid_argument, invalid of digit string: " << *str;
    return false;
  }
  return true;
}

std::shared_ptr<irpb::ProfilingParallel> ParallelStrategy::GetProfilingParallel() {
  std::shared_ptr<irpb::ProfilingParallel> profiling_parallel = std::make_shared<irpb::ProfilingParallel>();

  // set parallel model
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  std::string parallel_mode = parallel_context->parallel_mode();
  irpb::Config *config = profiling_parallel->mutable_config();
  MS_EXCEPTION_IF_NULL(config);
  config->set_parallel_type(parallel_mode);

  if (parallel_context->parallel_mode() == parallel::kAutoParallel ||
      parallel_context->parallel_mode() == parallel::kSemiAutoParallel) {
    uint32_t rank_id = CommManager::GetInstance().GetRank();
    uint32_t rank_size = 0;
    bool ret = CommManager::GetInstance().GetRankSize(kHcclWorldGroup, &rank_size);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Get rank size failed.";
    }
    int64_t stage_num = parallel_context->pipeline_stage_split_num();
    if (static_cast<int64_t>(rank_size) % stage_num != 0) {
      MS_LOG(EXCEPTION) << "Invalid stage num " << stage_num << " is not divisible by rank size " << rank_size;
    }
    int64_t device_per_stage = static_cast<int64_t>(rank_size) / stage_num;
    int64_t stage_id = static_cast<int64_t>(rank_id) / device_per_stage;
    config->set_rank_id(rank_id);
    config->set_stage_id(IntToUint(LongToInt(stage_id)));
    int64_t device = 0;
    for (int64_t i = 0; i < stage_num; ++i) {
      irpb::TensorShapeProto *stage_devices = config->add_stage_devices();
      MS_EXCEPTION_IF_NULL(stage_devices);
      for (int64_t j = 0; j < device_per_stage && device < static_cast<int64_t>(rank_size); ++j, ++device) {
        stage_devices->add_dim()->set_size(device);
      }
    }
  } else {
    auto rank_id = common::GetEnv("RANK_ID");
    // If RANK_ID is not set, default value is 0
    if (rank_id.empty()) {
      rank_id = "0";
      MS_LOG(WARNING) << "(Can not find RANK_ID in environment, "
                      << "This affects profiling to collect rank ID data and parallel strategy data. "
                      << "Please execute 'export RANK_ID=RANK_ID' in environment.)";
    }
    int32_t rank_id_int = 0;
    bool ret = StringToInt(&rank_id, &rank_id_int);
    if (!ret) {
      MS_LOG(EXCEPTION) << "The given RANK_ID is an invalid digit string.";
    }
    config->set_rank_id(rank_id_int);
  }

  has_got_parallel_strategy_data_ = true;
  return profiling_parallel;
}

void ParallelStrategy::DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {
  if (has_save_parallel_strategy_ || !IsProfilingParallelStrategyEnabled()) {
    return;
  }

  MS_LOG(INFO) << "Start to DumpProfileParallelStrategy.";

  cache_profiling_parallel_pb_ = GetProfilingParallel();
  graph_proto_str_ = GetFuncGraphProtoJsonString(func_graph);

  auto ascend_profiler = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->GetEnableFlag()) {
    MS_LOG(INFO) << "Profiling parallel strategy has not started.";
    return;
  }

  SaveParallelStrategyToFile();
}

void ParallelStrategy::SaveParallelStrategyToFile() {
  if (has_save_parallel_strategy_ || !has_got_parallel_strategy_data_) {
    return;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string dir = GetOutputPath();
  auto rank_id = common::GetEnv("RANK_ID");
  // If RANK_ID is not set, default value is 0
  if (rank_id.empty()) {
    rank_id = "0";
  }
  std::string parallel_str;
  (void)google::protobuf::util::MessageToJsonString(*cache_profiling_parallel_pb_, &parallel_str);
  std::string parallel_file = std::string("parallel_strategy_") + std::string(rank_id) + std::string(".json");
  std::string parallel_path = dir + "/" + parallel_file;
  MS_LOG(INFO) << "Start to write parallel strategy string, file path is " << parallel_path;
  std::ofstream ofs(parallel_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << parallel_path << "' failed!"
                  << " Errno:" << errno << " ErrInfo:" << strerror(errno);
    return;
  }

  ofs << parallel_str.substr(0, parallel_str.length() - 1) << ",\"graph\":" << graph_proto_str_ << "}";
  ofs.close();

  ChangeFileMode(parallel_path, S_IRUSR | S_IWUSR);

  has_save_parallel_strategy_ = true;

  MS_LOG(INFO) << "Save profile parallel strategy success.";
}

std::string ParallelStrategy::GetParallelStrategyForReport() {
  bool parallel_data_save_status = has_got_parallel_strategy_data_;
  std::string report_data;
  std::shared_ptr<irpb::ProfilingParallel> profiling_parallel;
  if (has_got_parallel_strategy_data_) {
    profiling_parallel = cache_profiling_parallel_pb_;
  } else {
    profiling_parallel = GetProfilingParallel();
  }

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  (void)google::protobuf::util::MessageToJsonString(*profiling_parallel, &report_data);
  try {
    nlohmann::json report_content = nlohmann::json::parse(report_data);
    report_content["config"]["ai_framework_type"] = "MindSpore";
    report_content["config"]["stage_num"] = parallel_context->pipeline_stage_split_num();
    report_data = report_content.dump();
  } catch (nlohmann::json::exception &e) {
    MS_LOG(ERROR) << e.what();
    report_data = "";
  }

  has_got_parallel_strategy_data_ = parallel_data_save_status;
  return report_data;
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
