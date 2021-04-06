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
const int kDataInfoQueueCapacity = 128;

class DeviceQueueOp : public PipelineOp {
 public:
  static const uint32_t INVALID_HANDLE = 0xffffffffUL;
  static const uint32_t WAIT_TIME = 5;

  enum class DeviceType { Ascend = 0, GPU = 1, CPU = 2 };

  //  The nested builder class inside of the DeviceQueueOp is used to help manage all of
  //  the arguments for constructing it.  Use the builder by setting each argument
  //  with the provided set methods, and then finally call the build method to execute
  //  the actual construction.
  class Builder {
   public:
    explicit Builder(int32_t prefetch_size);

    // Default destructor
    ~Builder() = default;

    Builder &SetPrefetchSize(int32_t prefetch_size) {
      builder_prefetch_size_ = prefetch_size;
      return *this;
    }

    Builder &SetChannelName(const std::string &channel_name) {
      builder_channel_name_ = channel_name;
      return *this;
    }

    Builder &SetDeviceType(const std::string &device_type) {
      if (device_type == "Ascend") {
        builder_device_type_ = DeviceType::Ascend;
      } else if (device_type == "GPU") {
        builder_device_type_ = DeviceType::GPU;
      } else if (device_type == "CPU") {
        builder_device_type_ = DeviceType::CPU;
      }
      return *this;
    }

    Builder &SetDeviceId(int32_t device_id) {
      builder_device_id_ = device_id;
      return *this;
    }

    Builder &SetSendEpochEnd(bool send_epoch_end) {
      builder_send_epoch_end_ = send_epoch_end;
      return *this;
    }

    Builder &SetTotalBatch(int32_t total_batch) {
      builder_total_batch_ = total_batch;
      return *this;
    }

    Builder &SetCreateDataInfoQueue(bool create_data_info_queue) {
      builder_create_data_info_queue_ = create_data_info_queue;
      return *this;
    }
    //  Name: Build()
    //  Description: The final step for building a DeviceQueueOp via the Builder is
    //              to call this Build() method.  It will instantiate the DeviceQueueOp
    //              and return it to caller as a shared pointer.
    Status Build(std::shared_ptr<DeviceQueueOp> *ptr) {
      *ptr = std::make_shared<DeviceQueueOp>(builder_channel_name_, builder_device_type_, builder_device_id_,
                                             builder_prefetch_size_, builder_send_epoch_end_, builder_total_batch_,
                                             builder_create_data_info_queue_);
      return Status::OK();
    }

   private:
    int32_t builder_prefetch_size_;
    int32_t builder_device_id_;
    DeviceType builder_device_type_;
    std::string builder_channel_name_;
    bool builder_send_epoch_end_;
    int32_t builder_total_batch_;
    bool builder_create_data_info_queue_;
  };

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

  const int32_t get_prefetch_size() { return prefetch_size_; }

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

  // Record the pipeline profiling info
  void ProfilingRecorder(bool isProfilingEnable, std::shared_ptr<DeviceQueueTracing> profiling_node, int64_t send_batch,
                         int32_t tdt_cost, uint64_t *batch_start_time, uint64_t *end_time, int32_t connector_capacity,
                         int32_t connector_size);

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kDeviceQueueOp; }

 private:
  //  Name: checkExceptions(DataBuffer);
  //  Description: Check whether the dataBuffer meets the condition for performing DeviceQueueOp
  Status CheckExceptions(const std::unique_ptr<DataBuffer> &buffer) const;

 private:
#ifdef ENABLE_TDTQUE
  void WaitContinueSignal() const;
  Status SendDataToAscend();
  void LimitSendingBatches(int64_t send_batch, int64_t *sending_num, std::shared_ptr<ConfigManager> cfg);
  Status SendRowToTdt(TensorRow currRow, bool isProfilingEnable, int32_t *tdt_cost);
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

  QueueList<std::unique_ptr<DataBuffer>> receive_queues_;
  std::vector<std::shared_ptr<MemoryPool>> pool_;
  std::unique_ptr<GpuItemConnector> gpu_item_connector_;
  uint32_t num_workers_;
  uint32_t queue_capacity_;
  // This rank_id is for device_queue, one process work with only one rank_id,
  // for standalone scenario, this rank_id may come from env 'CUDA_VISIBLE_DEVICES',
  // but for distribute scenario, this rank_id come from _get_global_rank() in python
  uint32_t rank_id_;
#endif

  Status SendDataToCPU();
  std::string channel_name_;
  DeviceType device_type_;
  const int32_t device_id_;
  const int32_t prefetch_size_;
  const bool send_epoch_end_;
  bool stop_send_;
  int32_t total_batch_;
  bool create_data_info_queue_;
  std::unique_ptr<DATA_INFO_QUEUE> data_info_queue_ptr_;
  std::mutex data_info_mutex_;
  bool send_finished_;
#ifdef ENABLE_DUMP_IR
  std::shared_ptr<MDChannelInfo> md_channel_info_;
#endif

#ifdef ENABLE_TDTQUE
  std::shared_ptr<TdtPlugin> tdtInstancePtr;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_
