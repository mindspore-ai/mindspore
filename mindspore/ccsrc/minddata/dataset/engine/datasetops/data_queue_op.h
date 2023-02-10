/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATA_QUEUE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATA_QUEUE_OP_H_

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/dataset_iterator.h"

#include "minddata/dataset/engine/perf/device_queue_tracing.h"
#include "minddata/dataset/util/status.h"
#ifdef ENABLE_DUMP_IR
#include "minddata/dataset/util/rdr.h"
#endif
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/circular_pool.h"
#include "mindspore/ccsrc/include/backend/data_queue/data_queue.h"

namespace mindspore {
namespace dataset {
class GpuConnector;
using DATA_INFO = std::vector<std::pair<DataType, TensorShape>>;
using DATA_INFO_QUEUE = Queue<DATA_INFO>;
using mindspore::device::DataQueueItem;
using mindspore::device::DataQueueStatus;
constexpr int32_t kTimeOutMilliSeconds = 25000;
const int kDataInfoQueueCapacity = 128;

class DataQueueOp : public PipelineOp {
 public:
  static const uint32_t INVALID_HANDLE = 0xffffffffUL;
  const uint32_t WAIT_TIME = 5;

  enum class DeviceType { Ascend = 0, GPU = 1, CPU = 2 };

  //  Name: constructor
  //  Description
  DataQueueOp(const std::string channel_name, DeviceType device_type, int32_t device_id, bool send_epoch_end,
              int32_t total_batch, bool create_data_info_queue);

  //  Name: destructor
  //  Description
  ~DataQueueOp();

  /// \brief Getter function
  /// \return connector size of current op
  int32_t ConnectorSize() const { return ChildOpConnectorSize(); }

  Status EoeReceived(int32_t worker_id) override;

  void StopSend() {
    stop_send_ = true;
    send_finished_ = true;
  }

  void ContinueSend() {
    MS_LOG(INFO) << "continue send at the beginning of the epoch";
    stop_send_ = false;
  }

  void StopWaiting() { ascend_keep_waiting_ = false; }

  Status ClearDevice();

  Status GetDataInfo(DATA_INFO *data_info);

  // Name: Print()
  // Description: A function that prints info about the node
  void Print(std::ostream &out,              // In: The output stream to print to
             bool show_all) const override;  // In: T/F if it should print everything

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const DataQueueOp &to) {
    to.Print(out, false);
    return out;
  }

  Status operator()() override;
#ifndef ENABLE_SECURITY
  // Record the pipeline profiling info
  void ProfilingRecorder(bool is_profiling_enable, const std::shared_ptr<DeviceQueueTracing> &profiling_node,
                         int64_t send_batch, int32_t tdt_cost, uint64_t *batch_start_time, uint64_t *end_time,
                         int32_t connector_capacity, int32_t connector_size) const;

#endif
  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kDeviceQueueOp; }

 private:
  // Name: FilterMetadata(TensorRow *);
  // Description: Auto filter metadata column before sending to device.
  Status FilterMetadata(TensorRow *row) const;

  // Name: CheckExceptions(TensorRow);
  // Description: Check whether the TensorRow meets the condition for performing DataQueueOp
  Status CheckExceptions(const TensorRow &row) const;

  // Name: PrintBeginInfoWhenFirstBatch(bool)
  // Description: Print info when first batch begin to send in sink_mode
  void PrintBeginInfoWhenFirstBatch(const bool &first_push_flag) const;

  // Name: PrintEndInfoWhenFirstBatch(bool)
  // Description: Print info when first batch send successful in sink_mode
  void PrintEndInfoWhenFirstBatch(bool *first_push_flag) const;
  Status RetryPushData(const std::vector<DataQueueItem> &data, bool profiling, uint64_t *push_time);
  bool NoExceptionRaised() const;
  Status SendDataToAscendDynamic();

  void WaitContinueSignal() const;
  Status SendDataToAscend();
  Status SendEpochEndToAscend(const TensorRow &curr_row, const bool &is_profiling_enable, int32_t *tdt_cost,
                              bool *is_break_loop);
  void LimitSendingBatches(int64_t send_batch, int64_t *sending_num, const std::shared_ptr<ConfigManager> &cfg) const;
  Status SendRowToTdt(TensorRow curr_row, bool is_profiling_enable, int32_t *tdt_cost);
  // check status that push data into device
  Status CheckPushStatus(DataQueueStatus status, bool stop_send, bool *send_finished, bool *is_break_loop);
  bool ascend_keep_waiting_;

  Status SendDataToGPU();
  Status MallocForGPUData(std::vector<device::DataQueueItem> *items, const TensorRow &curr_row,
                          const int32_t &worker_id);
  void ReleaseData(void *addr, int32_t worker_id);
  Status LaunchParallelCopyThread();
  Status PushDataToGPU();
  Status WorkerEntry(int32_t worker_id);
  Status SetThreadDevice();
  Status CreateDynamicDataQueue();
  double CalMbufQueueMemory(size_t realtime_queue_size);
  void RecordProfilingData(bool is_profiling_enable, bool end_of_epoch, int32_t *connector_size,
                           int32_t *connector_capacity, int64_t *send_batch);

  QueueList<TensorRow> receive_queues_;
  std::vector<std::shared_ptr<MemoryPool>> pool_;
  std::unique_ptr<GpuConnector> gpu_connector_;
  const uint32_t kDeviceQueGpuNumThreads = 2;
  const uint32_t kDeviceQueGpuQueueCapacity = 8;
  const int32_t kDeviceQueGpuThreadMemory = 1024;
  const uint32_t kDynamicHostQueueCapacity = 2;
  uint32_t num_workers_;
  uint32_t queue_capacity_;

  Status SendDataToCPU();
#ifndef ENABLE_SECURITY
  // Create async thread to detect whether it takes too long and unable to fetch first batch
  Status DetectFirstBatch();

  // Detect the cost time of each batch, present alarm message if cost too long
  void DetectPerBatchTime(const uint64_t *start_time, uint64_t *end_time) const;
#endif

  std::unique_ptr<ChildIterator> child_iterator_;
  std::string channel_name_;
  DeviceType device_type_;
  const int32_t device_id_;
  const bool send_epoch_end_;
  bool stop_send_;
  bool send_finished_;
  int32_t total_batch_;
  bool create_data_info_queue_;
  std::unique_ptr<DATA_INFO_QUEUE> data_info_queue_ptr_;
  std::atomic<bool> first_fetch_flag_;
  std::mutex data_info_mutex_;
  bool first_push_flag_;  // default: false, when first push, it will be true
  bool dynamic_shape_{false};
  std::deque<double> memory_per_batch_;
  std::shared_ptr<device::DataQueue> ascend_data_queue_;

#ifdef ENABLE_DUMP_IR
  std::shared_ptr<MDChannelInfo> md_channel_info_;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DATA_QUEUE_OP_H_
