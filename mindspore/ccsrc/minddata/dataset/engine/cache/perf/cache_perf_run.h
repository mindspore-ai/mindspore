/**
 * Copyright 2020 Huawei Technologies Co., Ltd

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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_RUN_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_RUN_H_

#include <getopt.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/cache/perf/cache_msg.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {

constexpr int32_t kDftNumOfPipelines = 8;
constexpr int32_t kDftNumberOfEpochs = 10;
constexpr int32_t kDftCacheSize = 0;
constexpr bool kDftShuffle = false;
constexpr bool kDftSpill = false;

class CachePerfRun {
 public:
  static const char kCachePipelineBinary[];
  CachePerfRun();
  ~CachePerfRun();
  void PrintHelp();
  int32_t ProcessArgs(int argc, char **argv);

  void Print(std::ostream &out) const {
    out << "Number of pipelines: " << num_pipelines_ << "\n"
        << "Number of epochs: " << num_epoches_ << "\n"
        << "Sample size: " << num_rows_ << "\n"
        << "Average row size: " << row_size_ << "\n"
        << "Shuffle: " << std::boolalpha << shuffle_;
  }

  friend std::ostream &operator<<(std::ostream &out, const CachePerfRun &cp) {
    cp.Print(out);
    return out;
  }

  Status Run();

 private:
  std::mutex mux_;
  int32_t my_pipeline_;
  int32_t num_pipelines_;
  int32_t num_epoches_;
  int64_t num_rows_;
  int32_t row_size_;
  bool shuffle_;
  CacheClient::Builder cache_builder_;
  session_id_type session_;
  int32_t crc_;
  std::vector<int32_t> pid_lists_;
  std::vector<int32_t> msg_send_lists_;
  std::vector<int32_t> msg_recv_lists_;
  TaskGroup vg_;
  std::atomic<int32_t> epoch_sync_cnt_;
  WaitPost pipeline_wp_;
  std::map<std::pair<int32_t, int32_t>, PipelineWorkerEpochSummary> epoch_results_;
  ConfigManager cfg_;
  std::shared_ptr<CacheClient> cc_;

  Status GetSession();
  Status ListenToPipeline(int32_t workerId);
  void PrintEpochSummary() const;
  Status StartPipelines();
  Status Cleanup();
  int32_t SanityCheck(std::map<int32_t, int32_t> seen_opts);
  int32_t ProcessArgsHelper(int32_t opt);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PERF_RUN_H_
