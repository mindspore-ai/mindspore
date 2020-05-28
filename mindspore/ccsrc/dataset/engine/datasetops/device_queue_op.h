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
#ifndef DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_
#define DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "dataset/engine/datasetops/pipeline_op.h"
#include "dataset/util/status.h"

#ifdef ENABLE_TDTQUE
#include "dataset/engine/tdt/tdt_plugin.h"

#endif

#ifdef ENABLE_GPUQUE
#include "device/gpu/gpu_buffer_mgr.h"
using mindspore::device::GpuBufferMgr;
#endif

namespace mindspore {
namespace dataset {
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

    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
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

    Builder &SetNumBatch(int64_t num_batch) {
      builder_num_batch_ = num_batch;
      return *this;
    }

    //  Name: Build()
    //  Description: The final step for building a DeviceQueueOp via the Builder is
    //              to call this Build() method.  It will instantiate the DeviceQueueOp
    //              and return it to caller as a shared pointer.
    Status Build(std::shared_ptr<DeviceQueueOp> *ptr) {
      *ptr = std::make_shared<DeviceQueueOp>(builder_channel_name_, builder_device_type_, builder_device_id_,
                                             builder_prefetch_size_, builder_op_connector_size_, builder_num_batch_);
      return Status::OK();
    }

   private:
    int32_t builder_prefetch_size_;
    int32_t builder_device_id_;
    DeviceType builder_device_type_;
    std::string builder_channel_name_;
    int64_t builder_num_batch_;
    int32_t builder_op_connector_size_;
  };

  //  Name: constructor
  //  Description
  DeviceQueueOp(std::string channel_name, DeviceType device_type, int32_t device_id, int32_t prefetch_size,
                int32_t op_connector_size, int64_t num_batch);

  //  Name: destructor
  //  Description
  ~DeviceQueueOp();

  Status EoeReceived(int32_t worker_id) override;

  const int32_t get_prefetch_size() { return prefetch_size_; }

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

  // Base-class override for NodePass visitor acceptor.
  // @param p - Pointer to the NodePass to be accepted.
  // @param modified - Whether this node visit modified the pipeline.
  // @return - Status of the node visit.
  Status Accept(NodePass *p, bool *modified) override;

 private:
  //  Name: checkExceptions(DataBuffer);
  //  Description: Check whether the dataBuffer meets the condition for performing DeviceQueueOp
  Status CheckExceptions(const std::unique_ptr<DataBuffer> &buffer) const;

#ifdef ENABLE_TDTQUE
  Status SendDataToAscend();
#endif

#ifdef ENABLE_GPUQUE
  Status SendDataToGPU();
  Status RetryPushGPUData(const std::vector<size_t> &data_size, const TensorRow &curr_row, uint32_t handle);
  Status MallocForGPUData(std::vector<device::DataItemGpu> *items, const TensorRow &curr_row);
#endif

  Status SendDataToCPU();
  std::string channel_name_;
  DeviceType device_type_;
  const int32_t device_id_;
  const int32_t prefetch_size_;
  const int64_t num_batch_;

#ifdef ENABLE_TDTQUE
  std::shared_ptr<TdtPlugin> tdtInstancePtr;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_DATASETOPS_DEVICE_QUEUE_OP_H_
