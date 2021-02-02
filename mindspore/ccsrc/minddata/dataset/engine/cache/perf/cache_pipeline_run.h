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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PIPELINE_RUN_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PIPELINE_RUN_H_

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

class CachePipelineRun {
 public:
  CachePipelineRun();
  ~CachePipelineRun();
  static void PrintHelp();
  int32_t ProcessArgs(int argc, char **argv);
  int32_t ProcessPipelineArgs(char *argv);
  int32_t ProcessClientArgs(char *argv);

  void Print(std::ostream &out) const {
    out << "Number of pipelines: " << num_pipelines_ << "\n"
        << "Number of epochs: " << num_epoches_ << "\n"
        << "Sample size: " << num_rows_ << "\n"
        << "Average row size: " << row_size_ << "\n"
        << "Shuffle: " << std::boolalpha << shuffle_;
  }

  friend std::ostream &operator<<(std::ostream &out, const CachePipelineRun &cp) {
    cp.Print(out);
    return out;
  }

  Status Run();

  template <typename T>
  Status SendMessage(CachePerfMsg *msg, CachePerfMsg::MessageType type, T *proto) {
    RETURN_UNEXPECTED_IF_NULL(msg);
    msg->SetType(type);
    if (proto != nullptr) {
      auto size_needed = proto->ByteSizeLong();
      CHECK_FAIL_RETURN_UNEXPECTED(
        size_needed <= kSharedMessageSize,
        "Shared message set too small. Suggest to increase to " + std::to_string(size_needed));
      CHECK_FAIL_RETURN_UNEXPECTED(proto->SerializeToArray(msg->GetMutableBuffer(), kSharedMessageSize),
                                   "Serialization fails");
      msg->SetProtoBufSz(size_needed);
    }
    RETURN_IF_NOT_OK(msg->Send(send_id_));
    return Status::OK();
  }

 private:
  int32_t my_pipeline_;
  int32_t num_pipelines_;
  int32_t num_epoches_;
  int64_t num_rows_;
  int32_t row_size_;
  bool shuffle_;
  CacheClient::Builder cache_builder_;
  session_id_type session_;
  int32_t crc_;
  TaskGroup vg_;
  WaitPost pipeline_wp_;
  int32_t send_id_;
  int32_t recv_id_;
  row_id_type start_row_;
  row_id_type end_row_;
  ConfigManager cfg_;
  QueueList<std::unique_ptr<IOBlock>> io_block_queues_;  // queues of IOBlocks
  std::shared_ptr<CacheClient> cc_;

  Status ListenToParent();
  Status RunFirstEpoch();
  Status RunReadEpoch();
  Status WriterWorkerEntry(int32_t worker_id);
  Status ReaderWorkerEntry(int32_t worker_id);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_PIPELINE_RUN_H_
