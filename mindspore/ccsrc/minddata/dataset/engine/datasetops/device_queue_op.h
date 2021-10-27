/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_

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
#include "debug/rdr/running_data_recorder.h"
#include "minddata/dataset/util/rdr.h"
#endif

#ifdef ENABLE_TDTQUE
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/engine/tdt/tdt_plugin.h"
#endif

#ifdef ENABLE_GPUQUE
#include "minddata/dataset/engine/gpu_item_connector.h"
#include "minddata/dataset/util/circular_pool.h"
#include "runtime/device/gpu/gpu_buffer_mgr.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
using mindspore::device::BlockQueueStatus_T;
using mindspore::device::GpuBufferMgr;
#endif

namespace mindspore {
namespace dataset {
using DATA_INFO = std::vector<std::pair<DataType, TensorShape>>;
using DATA_INFO_QUEUE = Queue<DATA_INFO>;

constexpr int32_t kTimeOutMilliSeconds = 25000;
const int kDataInfoQueueCapacity = 128;

class DeviceQueueOp : public PipelineOp {
 public:
  static const uint32_t INVALID_HANDLE = 0xffffffffUL;
  static const uint32_t WAIT_TIME = 5;

  enum class DeviceType { Ascend = 0, GPU = 1, CPU = 2 };

  //  Name: constructor
  //  Description
  DeviceQueueOp(std::string channel_name, DeviceType device_type, int32_t device_id, int32_t prefetch_size,
                bool send_epoch_end, int32_t total_batch, bool create_data_info_queue);

  //  Name: destructor
  //  Description
  ~DeviceQueueOp();

  /// \brief Getter function
  /// \return connector size of current op
  int32_t ConnectorSize() const { return ChildOpConnectorSize(); }

  Status EoeReceived(int32_t worker_id) override;

  const int32_t GetPrefetchSize() { return prefetch_size_; }

  void StopSend() { stop_send_ = true; }

  void ContinueSend() {
    MS_LOG(INFO) << "continue send at the beginning of the epoch";
    stop_send_ = false;
  }

#ifdef ENABLE_TDTQUE
  void StopWaiting() { ascend_keep_waiting_ = false; }
#endif

  Status GetDataInfo(DATA_INFO *data_info);

  // Name: Print()
  // Description: A function that prints info about the node
  void Print(std::ostream &out,              // In: The output stream to print to
             bool show_all) const override;  // In: T/F if it should print everything

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const DeviceQueueOp &to) {
    to.Print(out, false);
    return out;
  }

  Status operator()() override;
#ifndef ENABLE_SECURITY
  // Record the pipeline profiling info
  void ProfilingRecorder(bool is_profiling_enable, std::shared_ptr<DeviceQueueTracing> profiling_node,
                         int64_t send_batch, int32_t tdt_cost, uint64_t *batch_start_time, uint64_t *end_time,
                         int32_t connector_capacity, int32_t connector_size);

#endif
  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kDeviceQueueOp; }

 private:
  // Name: FilterMetadata(TensorRow *);
  // Description: Auto filter metadata column before sending to device.
  Status FilterMetadata(TensorRow *row);

  // Name: CheckExceptions(TensorRow);
  // Description: Check whether the TensorRow meets the condition for performing DeviceQueueOp
  Status CheckExceptions(const TensorRow &row) const;

  // Name: PrintBeginInfoWhenFirstBatch(bool)
  // Description: Print info when first batch begin to send in sink_mode
  void PrintBeginInfoWhenFirstBatch(const bool &first_push_flag);

  // Name: PrintEndInfoWhenFirstBatch(bool)
  // Description: Print info when first batch send successful in sink_mode
  void PrintEndInfoWhenFirstBatch(bool *first_push_flag);

 private:
#ifdef ENABLE_TDTQUE
  void WaitContinueSignal() const;
  Status SendDataToAscend();
  void LimitSendingBatches(int64_t send_batch, int64_t *sending_num, std::shared_ptr<ConfigManager> cfg);
  Status SendRowToTdt(TensorRow curr_row, bool is_profiling_enable, int32_t *tdt_cost);
  bool ascend_keep_waiting_;
#endif

#ifdef ENABLE_GPUQUE
  Status SendDataToGPU();
  Status MallocForGPUData(std::vector<device::DataItemGpu> *items, const TensorRow &curr_row, const int32_t &worker_id);
  Status RetryPushData(unsigned int handle, const std::vector<DataItemGpu> &data);
  void ReleaseData(void *addr, int32_t worker_id);
  Status LaunchParallelCopyThread();
  Status PushDataToGPU();
  Status WorkerEntry(int32_t worker_id);
  Status SetThreadDevice();

  QueueList<TensorRow> receive_queues_;
  std::vector<std::shared_ptr<MemoryPool>> pool_;
  std::unique_ptr<GpuItemConnector> gpu_item_connector_;
  const uint32_t kDeviceQueGpuNumThreads = 2;
  const uint32_t kDeviceQueGpuQueueCapacity = 8;
  const uint32_t kDeviceQueGpuThreadMemory = 1024;
  uint32_t num_workers_;
  uint32_t queue_capacity_;
  // This rank_id is for device_queue, one process work with only one rank_id,
  // for standalone scenario, this rank_id may come from env 'CUDA_VISIBLE_DEVICES',
  // but for distribute scenario, this rank_id come from _get_global_rank() in python
  uint32_t rank_id_;
#endif

  Status SendDataToCPU();
#ifndef ENABLE_SECURITY
  // Create async thread to detect whether it takes too long and unable to fetch first batch
  Status DetectFirstBatch();

  // Detect the cost time of each batch, present alarm message if cost too long
  void DetectPerBatchTime(const uint64_t *start_time, uint64_t *end_time);
#endif

  std::unique_ptr<ChildIterator> child_iterator_;
  std::string channel_name_;
  DeviceType device_type_;
  const int32_t device_id_;
  const int32_t prefetch_size_;
  const bool send_epoch_end_;
  bool stop_send_;
  bool send_finished_;
  int32_t total_batch_;
  bool create_data_info_queue_;
  std::unique_ptr<DATA_INFO_QUEUE> data_info_queue_ptr_;
  std::atomic<bool> first_fetch_flag_;
  std::mutex data_info_mutex_;
  bool first_push_flag_;  // default: false, when first push, it will be true

#ifdef ENABLE_TDTQUE
  std::shared_ptr<TdtPlugin> tdtInstancePtr;
#endif
#ifdef ENABLE_DUMP_IR
  std::shared_ptr<MDChannelInfo> md_channel_info_;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_
