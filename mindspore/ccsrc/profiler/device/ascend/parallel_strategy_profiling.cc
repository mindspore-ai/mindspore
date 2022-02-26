/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "profiler/device/ascend/parallel_strategy_profiling.h"

#include <vector>
#include "sys/stat.h"

#include "debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "profiler/device/ascend/options.h"
#include "profiler/device/ascend/ascend_profiling.h"
#include "proto/profiling_parallel.pb.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

#include "google/protobuf/util/json_util.h"

#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/ps_context.h"
#include "ps/util.h"
#endif

namespace mindspore {
namespace profiler {
namespace ascend {
bool has_save_parallel_strategy = false;
bool has_got_parallel_strategy_data = false;
irpb::ProfilingParallel cache_profiling_parallel_pb;

bool IsProfilingParallelStrategyEnabled() {
  auto ascend_profiler = AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->IsInitialized()) {
    MS_LOG(INFO) << "Profiling parallel strategy is disabled.";
    return false;
  }

#if ((defined ENABLE_CPU) && (!defined _WIN32))
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

bool StringToInt(std::string *str, int32_t *value) {
  try {
    *value = stoi(*str);
  } catch (std::invalid_argument &) {
    MS_LOG(ERROR) << "Catch invalid_argument, invalid of digit string: " << *str;
    return false;
  }
  return true;
}

irpb::ProfilingParallel GetProfilingParallel(const FuncGraphPtr &func_graph) {
  irpb::ProfilingParallel profiling_parallel;
  irpb::GraphProto *graph_proto = profiling_parallel.mutable_graph();
  MS_EXCEPTION_IF_NULL(graph_proto);
  GetFuncGraphProto(func_graph, graph_proto);

  // set parallel model
  std::string parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  irpb::Config *config = profiling_parallel.mutable_config();
  MS_EXCEPTION_IF_NULL(config);
  config->set_parallel_type(parallel_mode);

  // Note: Only parallel mode is AUTO_PARALLEL or SEMI_AUTO_PARALLEL, the
  // g_device_manager is not nullptr;
  if (parallel::g_device_manager != nullptr) {
    auto rank_id = parallel::g_device_manager->global_rank();
    auto stage_id = parallel::g_device_manager->stage_id();
    config->set_rank_id(rank_id);
    config->set_stage_id(stage_id);

    // set stage_devices
    for (std::vector<int64_t> devices : parallel::g_device_manager->stage_devices()) {
      irpb::TensorShapeProto *stage_devices = config->add_stage_devices();
      MS_EXCEPTION_IF_NULL(stage_devices);
      for (int64_t device : devices) {
        stage_devices->add_dim()->set_size(device);
      }
    }
  } else {
    auto rank_id = common::GetEnv("RANK_ID");
    // If RANK_ID is not set, default value is 0
    if (rank_id.empty()) {
      rank_id = "0";
      MS_LOG(WARNING) << R"(Can not find RANK_ID in environment, This affects profiling to "
                         "collect rank ID data and parallel strategy data. please execute "
                         "'export RANK_ID=RANK_ID' in environment.)";
    }
    int32_t rank_id_int = 0;
    bool ret = StringToInt(&rank_id, &rank_id_int);
    if (!ret) {
      MS_LOG(EXCEPTION) << "The given RANK_ID is an invalid digit string.";
    }
    config->set_rank_id(rank_id_int);
  }

  has_got_parallel_strategy_data = true;
  return profiling_parallel;
}

void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph) {
  if (!IsProfilingParallelStrategyEnabled()) {
    return;
  }

  MS_LOG(INFO) << "Start to DumpProfileParallelStrategy.";

  cache_profiling_parallel_pb = GetProfilingParallel(func_graph);

  auto ascend_profiler = AscendProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_profiler);
  if (!ascend_profiler->GetProfilingEnableFlag()) {
    MS_LOG(INFO) << "Profiling parallel strategy has not started.";
    return;
  }

  SaveParallelStrategyToFile();
}

void SaveParallelStrategyToFile() {
  if (has_save_parallel_strategy || !has_got_parallel_strategy_data) {
    return;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string dir_path = GetOutputPath();
  auto rank_id = common::GetEnv("RANK_ID");
  // If RANK_ID is not set, default value is 0
  if (rank_id.empty()) {
    rank_id = "0";
  }
  std::string file_path = dir_path + std::string("/parallel_strategy_") + std::string(rank_id) + std::string(".json");

  MS_LOG(INFO) << "Start to write parallel strategy string, file path is " << file_path;
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << file_path << "' failed!"
                  << " Errno:" << errno << " ErrInfo:" << strerror(errno);
    return;
  }

  std::string profiling_parallel_str;
  google::protobuf::util::MessageToJsonString(cache_profiling_parallel_pb, &profiling_parallel_str);
  ofs << profiling_parallel_str;
  ofs.close();

  ChangeFileMode(file_path, S_IRUSR | S_IWUSR);

  has_save_parallel_strategy = true;

  MS_LOG(INFO) << "Save profile parallel strategy success.";
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
