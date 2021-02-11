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

#include "minddata/dataset/engine/cache/perf/cache_pipeline_run.h"
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/services.h"

namespace mindspore {
namespace dataset {
void CachePipelineRun::PrintHelp() { std::cout << "Please run the executable cache_perf instead." << std::endl; }

int32_t CachePipelineRun::ProcessPipelineArgs(char *argv) {
  try {
    std::stringstream cfg_ss(argv);
    std::string s;
    int32_t numArgs = 0;
    while (std::getline(cfg_ss, s, ',')) {
      if (numArgs == 0) {
        my_pipeline_ = std::stoi(s);
      } else if (numArgs == 1) {
        session_ = std::stoul(s);
        cache_builder_.SetSessionId(session_);
      } else if (numArgs == 2) {
        crc_ = std::stoi(s);
      } else if (numArgs == 3) {
        recv_id_ = std::stoi(s);
      } else if (numArgs == 4) {
        send_id_ = std::stoi(s);
      } else if (numArgs == 5) {
        num_pipelines_ = std::stoi(s);
      } else if (numArgs == 6) {
        num_epoches_ = std::stoi(s);
      } else if (numArgs == 7) {
        num_rows_ = std::stol(s);
      } else if (numArgs == 8) {
        row_size_ = std::stoi(s);
      } else if (numArgs == 9) {
        cfg_.set_num_parallel_workers(std::stol(s));
      } else if (numArgs == 10) {
        shuffle_ = strcmp(s.data(), "true") == 0;
      }
      ++numArgs;
    }
    if (numArgs != 11) {
      std::cerr << "Incomplete arguments. Expect 11. But get " << numArgs << std::endl;
      return -1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Parse error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}

int32_t CachePipelineRun::ProcessClientArgs(char *argv) {
  try {
    std::stringstream client_ss(argv);
    std::string s;
    int32_t numArgs = 0;
    while (std::getline(client_ss, s, ',')) {
      if (numArgs == 0) {
        cache_builder_.SetHostname(s);
      } else if (numArgs == 1) {
        cache_builder_.SetPort(std::stoi(s));
      } else if (numArgs == 2) {
        cache_builder_.SetPrefetchSize(std::stoi(s));
      } else if (numArgs == 3) {
        cache_builder_.SetCacheMemSz(std::stoi(s));
      } else if (numArgs == 4) {
        cache_builder_.SetNumConnections(std::stoi(s));
      } else if (numArgs == 5) {
        cache_builder_.SetSpill(strcmp(s.data(), "true") == 0);
      }
      ++numArgs;
    }
    if (numArgs != 6) {
      std::cerr << "Incomplete arguments. Expect 6. But get " << numArgs << std::endl;
      return -1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Parse error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}

int32_t CachePipelineRun::ProcessArgs(int argc, char **argv) {
  if (argc != 3) {
    PrintHelp();
    return -1;
  }
  int32_t rc = ProcessPipelineArgs(argv[1]);
  if (rc < 0) return rc;
  rc = ProcessClientArgs(argv[2]);
  return rc;
}

CachePipelineRun::CachePipelineRun()
    : my_pipeline_(-1),
      num_pipelines_(kDftNumOfPipelines),
      num_epoches_(kDftNumberOfEpochs),
      num_rows_(0),
      row_size_(0),
      shuffle_(kDftShuffle),
      session_(0),
      crc_(0),
      send_id_(-1),
      recv_id_(-1),
      start_row_(-1),
      end_row_(-1) {
  cache_builder_.SetSpill(kDftSpill).SetCacheMemSz(kDftCacheSize);
}

CachePipelineRun::~CachePipelineRun() {
  CachePerfMsg msg;
  (void)SendMessage<ErrorMsg>(&msg, CachePerfMsg::MessageType::kInterrupt, nullptr);
}

Status CachePipelineRun::ListenToParent() {
  TaskManager::FindMe()->Post();
  do {
    RETURN_IF_INTERRUPTED();
    CachePerfMsg msg;
    RETURN_IF_NOT_OK(msg.Receive(recv_id_));
    // Decode the messages.
    auto type = msg.GetType();
    switch (type) {
      case CachePerfMsg::MessageType::kInterrupt: {
        TaskManager::WakeUpWatchDog();
        return Status::OK();
      }
      case CachePerfMsg::MessageType::kEpochStart: {
        pipeline_wp_.Set();
        break;
      }
      default:
        std::string errMsg = "Unknown request type: " + std::to_string(type);
        MS_LOG(ERROR) << errMsg;
        RETURN_STATUS_UNEXPECTED(errMsg);
        break;
    }
  } while (true);

  return Status::OK();
}

Status CachePipelineRun::Run() {
  RETURN_IF_NOT_OK(cache_builder_.Build(&cc_));
  RETURN_IF_NOT_OK(vg_.ServiceStart());

  auto num_workers = cfg_.num_parallel_workers();
  io_block_queues_.Init(num_workers, cfg_.op_connector_size());

  RETURN_IF_NOT_OK(io_block_queues_.Register(&vg_));

  Status rc = cc_->CreateCache(crc_, false);
  // Duplicate key is fine.
  if (rc.IsError() && rc != StatusCode::kMDDuplicateKey) {
    return rc;
  }

  // Log a warning level message so we can see it in the log file when it starts.
  MS_LOG(WARNING) << "Pipeline number " << my_pipeline_ + 1 << " successfully creating cache service." << std::endl;

  // Spawn a thread to listen to the parent process
  RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Queue listener", std::bind(&CachePipelineRun::ListenToParent, this)));

  RETURN_IF_NOT_OK(RunFirstEpoch());

  // The rest of the epochs are just fetching.
  auto remaining_epochs = num_epoches_ - 1;
  while (remaining_epochs > 0) {
    // Wait for parent process signal to start
    pipeline_wp_.Wait();
    pipeline_wp_.Clear();
    RETURN_IF_NOT_OK(RunReadEpoch());
    --remaining_epochs;
  }

  // The listener thread is blocked on a shared message queue. It will be waken up by
  // the parent process which will send an interrupt message to it, and this program
  // will exit.
  RETURN_IF_NOT_OK(vg_.join_all());
  return Status::OK();
}

Status CachePipelineRun::RunFirstEpoch() {
  auto sz = num_rows_ / num_pipelines_;
  start_row_ = my_pipeline_ * sz;
  end_row_ = (my_pipeline_ + 1) * sz - 1;
  if (my_pipeline_ + 1 == num_pipelines_) {
    end_row_ = num_rows_ - 1;
  }
  std::cout << "Pipeline number " << my_pipeline_ + 1 << " row id range: [" << start_row_ << "," << end_row_ << "]"
            << std::endl;

  // Spawn the worker threads.
  auto f = std::bind(&CachePipelineRun::WriterWorkerEntry, this, std::placeholders::_1);
  std::vector<Task *> worker_threads;
  auto num_workers = cfg_.num_parallel_workers();
  worker_threads.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    Task *pTask;
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Parallel worker", std::bind(f, i), &pTask));
    worker_threads.push_back(pTask);
  }

  std::vector<row_id_type> keys;
  auto rows_per_buffer = cfg_.rows_per_buffer();
  keys.reserve(rows_per_buffer);
  int32_t worker_id = 0;
  for (auto i = start_row_; i <= end_row_; ++i) {
    keys.push_back(i);
    if (keys.size() == rows_per_buffer) {
      auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
      RETURN_IF_NOT_OK(io_block_queues_[worker_id++ % num_workers]->Add(std::move(blk)));
      keys.clear();
    }
  }
  if (!keys.empty()) {
    auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
    RETURN_IF_NOT_OK(io_block_queues_[worker_id++ % num_workers]->Add(std::move(blk)));
    keys.clear();
  }

  // Shutdown threads and wait for them to come back.
  for (int32_t i = 0; i < num_workers; i++) {
    RETURN_IF_NOT_OK(
      io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
  }
  for (auto *pTask : worker_threads) {
    RETURN_IF_NOT_OK(pTask->Join(Task::WaitFlag::kBlocking));
  }

  // Final flush
  cc_->FlushAsyncWriteBuffer();

  // Send a message saying epoch one done for this pipeline.
  EpochDone proto;
  proto.set_pipeline(my_pipeline_);
  CachePerfMsg msg;
  RETURN_IF_NOT_OK(SendMessage(&msg, CachePerfMsg::MessageType::kEpochEnd, &proto));

  return Status::OK();
}

Status CachePipelineRun::WriterWorkerEntry(int32_t worker_id) {
  Status rc;
  TaskManager::FindMe()->Post();
  int64_t min_val = std::numeric_limits<int64_t>::max();
  int64_t max_val = 0;
  int64_t total_val = 0;
  int64_t cnt = 0;
  std::vector<int64_t> duration;
  duration.reserve(num_rows_ / num_pipelines_ / cfg_.num_parallel_workers());
  bool resource_err = false;
  auto col_desc = std::make_unique<ColDescriptor>("int64", DataType(DataType::DE_INT64), TensorImpl::kFlexible, 1);
  auto num_elements = row_size_ / sizeof(int64_t);
  TensorShape shape(std::vector<dsize_t>(1, num_elements));
  std::unique_ptr<IOBlock> blk;
  auto epoch_start = std::chrono::steady_clock::now();
  do {
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&blk));
    std::vector<int64_t> keys;
    RETURN_IF_NOT_OK(blk->GetKeys(&keys));
    if (keys.empty()) {
      // empty key is a quit signal for workers
      break;
    }
    // Once we hit resource error, we drain the io block. No point to send anything to the server.
    if (!resource_err) {
      auto buffer = std::make_unique<DataBuffer>(cnt++, DataBuffer::kDeBFlagNone);
      auto tensor_table = std::make_unique<TensorQTable>();
      for (auto id : keys) {
        TensorRow row;
        std::shared_ptr<Tensor> element;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, col_desc->type(), &element));
        row.setId(id);
        // CreateEmpty allocates the memory but in virtual address. Let's commit the memory
        // so we can get an accurate timing.
        auto it = element->begin<int64_t>();
        for (auto i = 0; i < num_elements; ++i, ++it) {
          *it = i;
        }
        row.push_back(std::move(element));
        tensor_table->push_back(std::move(row));
      }
      buffer->set_tensor_table(std::move(tensor_table));
      // Measure the time to call WriteBuffer
      auto start_tick = std::chrono::steady_clock::now();
      rc = cc_->AsyncWriteBuffer(std::move(buffer));
      auto end_tick = std::chrono::steady_clock::now();
      if (rc.IsError()) {
        if (rc == StatusCode::kMDOutOfMemory || rc == StatusCode::kMDNoSpace) {
          MS_LOG(WARNING) << "Pipeline number " << my_pipeline_ + 1 << " worker id " << worker_id << ": "
                          << rc.ToString();
          resource_err = true;
          cc_->ServerRunningOutOfResources();
          continue;
        } else {
          return rc;
        }
      } else {
        int64_t ms = std::chrono::duration_cast<std::chrono::microseconds>(end_tick - start_tick).count();
        min_val = std::min(min_val, ms);
        max_val = std::max(max_val, ms);
        duration.push_back(ms);
        total_val += ms;
      }
    }
  } while (true);

  auto epoch_end = std::chrono::steady_clock::now();
  int64_t elapse_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

  PipelineWorkerEpochSummary proto;
  proto.set_pipeline(my_pipeline_);
  proto.set_worker(worker_id);
  proto.set_min(min_val);
  proto.set_max(max_val);
  proto.set_elapse(elapse_time);
  auto sz = duration.size();
  proto.set_cnt(sz);
  if (sz > 0) {
    // median
    auto n = sz / 2;
    std::nth_element(duration.begin(), duration.begin() + n, duration.end());
    auto median = duration[n];
    proto.set_med(median);
    // average
    int64_t avg = total_val / sz;
    proto.set_avg(avg);
  }
  CachePerfMsg msg;
  RETURN_IF_NOT_OK(SendMessage(&msg, CachePerfMsg::MessageType::kEpochResult, &proto));
  return Status::OK();
}

Status CachePipelineRun::RunReadEpoch() {
  std::vector<row_id_type> keys;
  auto rows_per_buffer = cc_->GetPrefetchSize();  // We will use prefetch size to read.
  auto num_workers = cfg_.num_parallel_workers();
  keys.reserve(rows_per_buffer);
  // Spawn workers
  auto f = std::bind(&CachePipelineRun::ReaderWorkerEntry, this, std::placeholders::_1);
  std::vector<Task *> worker_threads;
  worker_threads.reserve(num_workers);
  for (int32_t i = 0; i < num_workers; ++i) {
    Task *pTask;
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Parallel worker", std::bind(f, i), &pTask));
    worker_threads.push_back(pTask);
  }

  std::vector<row_id_type> all_keys;
  all_keys.reserve(end_row_ - start_row_ + 1);
  for (auto i = start_row_; i <= end_row_; ++i) {
    all_keys.push_back((i));
  }
  // If we need to shuffle the keys
  if (shuffle_) {
    std::shuffle(all_keys.begin(), all_keys.end(), GetRandomDevice());
  }

  int32_t worker_id = 0;
  for (auto id : all_keys) {
    keys.push_back(id);
    if (keys.size() == rows_per_buffer) {
      auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
      RETURN_IF_NOT_OK(io_block_queues_[worker_id++ % num_workers]->Add(std::move(blk)));
      keys.clear();
    }
  }
  if (!keys.empty()) {
    auto blk = std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone));
    RETURN_IF_NOT_OK(io_block_queues_[worker_id++ % num_workers]->Add(std::move(blk)));
    keys.clear();
  }

  // Shutdown threads and wait for them to come back.
  for (int32_t i = 0; i < num_workers; i++) {
    RETURN_IF_NOT_OK(
      io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
  }
  for (auto *pTask : worker_threads) {
    RETURN_IF_NOT_OK(pTask->Join(Task::WaitFlag::kBlocking));
  }

  // Send a message saying epoch one done for this pipeline.
  EpochDone proto;
  proto.set_pipeline(my_pipeline_);
  CachePerfMsg msg;
  RETURN_IF_NOT_OK(SendMessage(&msg, CachePerfMsg::MessageType::kEpochEnd, &proto));
  return Status::OK();
}

Status CachePipelineRun::ReaderWorkerEntry(int32_t worker_id) {
  Status rc;
  TaskManager::FindMe()->Post();
  int64_t min_val = std::numeric_limits<int64_t>::max();
  int64_t max_val = 0;
  int64_t total_val = 0;
  int64_t cnt = 0;
  std::vector<int64_t> duration;
  duration.reserve(num_rows_ / num_pipelines_ / cfg_.num_parallel_workers());
  std::unique_ptr<IOBlock> blk;
  auto epoch_start = std::chrono::steady_clock::now();
  do {
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&blk));
    std::vector<int64_t> keys;
    RETURN_IF_NOT_OK(blk->GetKeys(&keys));
    if (keys.empty()) {
      // empty key is a quit signal for workers
      break;
    }
    std::vector<row_id_type> prefetch_keys;
    prefetch_keys.reserve(keys.size());

    // Filter out all those keys that unlikely we will find at the server
    for (auto row_id : keys) {
      if (!cc_->KeyIsCacheMiss(row_id)) {
        prefetch_keys.push_back(row_id);
      }
    }
    // Early exit if nothing to fetch
    if (prefetch_keys.empty()) {
      continue;
    }
    // Get the rows from the server
    TensorTable ttbl;
    // Measure how long it takes for the row to come back.
    auto start_tick = std::chrono::steady_clock::now();
    RETURN_IF_NOT_OK(cc_->GetRows(prefetch_keys, &ttbl));
    auto end_tick = std::chrono::steady_clock::now();
    int64_t ms = std::chrono::duration_cast<std::chrono::microseconds>(end_tick - start_tick).count();
    min_val = std::min(min_val, ms);
    max_val = std::max(max_val, ms);
    duration.push_back(ms);
    total_val += ms;
    ++cnt;
  } while (true);

  auto epoch_end = std::chrono::steady_clock::now();
  int64_t elapse_time = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

  PipelineWorkerEpochSummary proto;
  proto.set_pipeline(my_pipeline_);
  proto.set_worker(worker_id);
  proto.set_min(min_val);
  proto.set_max(max_val);
  proto.set_elapse(elapse_time);
  auto sz = duration.size();
  proto.set_cnt(sz);
  if (sz > 0) {
    // median
    auto n = sz / 2;
    std::nth_element(duration.begin(), duration.begin() + n, duration.end());
    auto median = duration[n];
    proto.set_med(median);
    // average
    int64_t avg = total_val / sz;
    proto.set_avg(avg);
  }
  CachePerfMsg msg;
  RETURN_IF_NOT_OK(SendMessage(&msg, CachePerfMsg::MessageType::kEpochResult, &proto));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
