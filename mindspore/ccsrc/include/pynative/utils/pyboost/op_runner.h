/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_RUNNER_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_RUNNER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include "ir/scalar.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "ir/tensor.h"
#include "include/backend/visible.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/infer_info/infer_info_utils.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/pynative/utils/pyboost/comm_handle.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore {
namespace tensor {
using TensorPtr = tensor::TensorPtr;
}
namespace kernel {
namespace pyboost {
using TensorPtr = tensor::TensorPtr;
// OpRunner is a base class for operators.
// OpRunner records the operator's input abstract,
// output abstract and output Tensors for grad,
// and it also contains several functional methods for the operator to run.
class PYBOOST_API OpRunner : public std::enable_shared_from_this<OpRunner> {
 public:
  OpRunner(PrimitivePtr primitive, const DeviceContext *device_context)
      : primitive_(std::move(primitive)), device_context_(device_context) {}
  virtual ~OpRunner() = default;

  // For users to implement custom call functions in the "customize" directory.
  std::shared_ptr<OpRunner> get_op() { return shared_from_this(); }

  // set and get methods for class member variables.
  void set_primitive(const PrimitivePtr &primitive) { primitive_ = primitive; }
  const PrimitivePtr &primitive() const { return primitive_; }
  const std::vector<AbstractBasePtr> &input_abs() const { return input_abs_; }
  const AbstractBasePtr &output_abs() const { return output_abs_; }
  const DeviceContext *device_context() const { return device_context_; }
  const std::vector<pynative::DeviceAddressPromisePtr> &device_sync_promises() const { return device_sync_promises_; }
  const std::vector<tensor::TensorPtr> &outputs() const { return outputs_; }
  void set_outputs(const std::vector<tensor::TensorPtr> &outputs) { outputs_ = outputs; }
  void set_stream_id(size_t stream_id) { stream_id_ = stream_id; }
  void set_clone_tensor(const tensor::TensorPtr &clone_tensor) { clone_tensor_ = clone_tensor; }
  const tensor::TensorPtr &clone_tensor() { return clone_tensor_; }
  virtual bool output_is_tuple() const { return false; }

  void set_comm_handle(const CommHandlePtr &comm_handle) { comm_handle_ = comm_handle; }
  CommHandlePtr comm_handle() const { return comm_handle_; }

  size_t stream_id() const { return stream_id_; }
  ValueSimpleInfoPtr output_value_simple_info() const { return output_value_simple_info_; }

  const tensor::TensorPtr &output(const size_t &idx) {
    if (idx >= outputs_.size()) {
      MS_LOG(EXCEPTION) << "idx is out of bounds, idx:" << idx << ", outputs_.size():" << outputs_.size();
    }
    return outputs_[idx];
  }

  template <typename... T>
  void GenerateInputAbstract(T &...args) {
    input_abs_.clear();
    (input_abs_.emplace_back(kAbstractConverter.ConvertAbstract(args)), ...);
  }

  // Member function for Infer and creating output tensors.
  template <typename... T>
  void InferOutput(T &...args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostInferOutput,
                                       primitive_->name(), false);
    if (output_value_simple_info_ = ops::DoGeneralInfer(primitive_, args...); output_value_simple_info_ != nullptr) {
      MS_LOG(DEBUG) << "Op " << primitive_->name() << " infer by infer_info, get output "
                    << ValueSimpleInfoToString(*output_value_simple_info_);
      PyBoostUtils::CreateOutputTensor(output_value_simple_info_, &outputs_);
      return;
    } else if (output_value_simple_info_ = ops::InferBySimple(primitive_, args...);
               output_value_simple_info_ != nullptr) {
      MS_LOG(DEBUG) << "Op " << primitive_->name() << " infer by simple, get output "
                    << ValueSimpleInfoToString(*output_value_simple_info_);
      PyBoostUtils::CreateOutputTensor(output_value_simple_info_, &outputs_);
      return;
    }

    GenerateInputAbstract(args...);
    output_abs_ = PyBoostUtils::InferByOpDef(primitive_, input_abs_);
    MS_EXCEPTION_IF_NULL(output_abs_);
    MS_LOG(DEBUG) << "PyBoost infer by abstract, get output " << output_abs_->ToString();
    PyBoostUtils::CreateOutputTensor(output_abs_, &outputs_);
    kAbstractConverter.CacheAbstract(output_abs_);
  }

  // A static function used for the "customize" operator to generate the operator's output Tensor.
  template <typename... T>
  static void InferOpOutput(const std::shared_ptr<OpRunner> &op, T &...args) {
    op->InferOutput(args...);
  }

  void ProfileTrackerTask() {
    static bool enable_trace_mem = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
    if (MS_UNLIKELY(enable_trace_mem || device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info())) {
      PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([primitive = primitive_]() {
        // wait for event
        runtime::Pipeline::Get().launch_stage()->Wait();
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddNestedTask, "PyNative", primitive->name(), "");
      }));
    }
    if (MS_LIKELY(!enable_trace_mem)) {
      return;
    }
    skip_tracker_ = false;
  }

  template <typename... Args>
  void ProfileTrackerInput(const Args &...args) {
    if (MS_UNLIKELY(mindspore::runtime::ProfilerAnalyzer::GetInstance().profiler_enable())) {
      static auto ascend_profiler = mindspore::profiler::Profiler::GetInstance(kAscendDevice);
      if (ascend_profiler != nullptr && ascend_profiler->EnableRecordShapes()) {
        std::vector<tensor::TensorPtr> tensors;
        (CollectTrackerTensor(args, &tensors), ...);
        std::vector<ShapeVector> input_shapes;
        std::vector<std::string> input_types;
        for (const auto &tensor : tensors) {
          input_shapes.emplace_back(tensor->shape_c());
          input_types.emplace_back(tensor->Dtype()->ToString());
        }
        mindspore::runtime::ProfilerAnalyzer::GetInstance().RecordShapesData(primitive_->name(), input_shapes,
                                                                             input_types);
      }
    }
    if (MS_LIKELY(skip_tracker_)) {
      return;
    }
    std::vector<tensor::TensorPtr> tensors;
    (CollectTrackerTensor(args, &tensors), ...);

    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([primitive = primitive_, tensors]() {
      for (const auto &tensor : tensors) {
        MS_EXCEPTION_IF_NULL(tensor);
        const auto &device_sync = tensor->device_address();
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(device_sync);
        if (device_address == nullptr) {
          MS_LOG(WARNING) << "Tracker: input tensor device address is nullptr, primitive: " << primitive->name();
          continue;
        }
        MS_EXCEPTION_IF_NULL(device_address);
        if (device_address->GetPtr() == nullptr) {
          MS_LOG(WARNING) << "Tracker: input tensor device ptr is nullptr, primitive: " << primitive->name();
          continue;
        }
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
          MarkTensorAsInput, "PyNative", device::GetDeviceNameByType(device_address->GetDeviceType()),
          device_address->GetPtr(), tensor->data_type(), tensor->shape(), tensor->storage_info());
      }
    }));
  }

  template <typename... Args>
  void ProfileTrackerOutput(const std::tuple<Args...> &tuple) {
    if (MS_LIKELY(skip_tracker_)) {
      return;
    }
    std::vector<tensor::TensorPtr> tensors;
    std::apply([this, &tensors](const Args &...args) { (CollectTrackerTensor(args, &tensors), ...); }, tuple);
    TrackerOutputTensors(tensors);
  }

  template <typename T>
  void ProfileTrackerOutput(const std::vector<T> &vals) {
    if (MS_LIKELY(skip_tracker_)) {
      return;
    }
    std::vector<tensor::TensorPtr> tensors;
    for (const auto &val : vals) {
      CollectTrackerTensor(val, &tensors);
    }
    TrackerOutputTensors(tensors);
  }

  void ProfileTrackerOutput(const ValuePtr &val) {
    if (MS_LIKELY(skip_tracker_)) {
      return;
    }
    std::vector<tensor::TensorPtr> tensors;
    CollectTrackerTensor(val, &tensors);
    TrackerOutputTensors(tensors);
  }

  void UpdateOutputShape(const TensorPtr &tensor, const ShapeVector &shape) { tensor->set_shape(shape); }

  void CreateOutputSimpleInfo() {
    if (output_value_simple_info_ != nullptr) {
      MS_LOG(DEBUG) << "op:" << primitive_->name() << ", already have simple-info";
      return;
    }
    output_value_simple_info_ = std::make_shared<ValueSimpleInfo>();
    output_value_simple_info_->is_tuple_output_ = output_is_tuple();
    output_value_simple_info_->size_ = outputs_.size();
    for (size_t i = 0; i < outputs_.size(); ++i) {
      output_value_simple_info_->shape_vector_.emplace_back(outputs_[i]->shape());
      output_value_simple_info_->dtype_vector_.emplace_back(outputs_[i]->Dtype());
    }
  }

 protected:
  // Op primitive, may delete latter.
  PrimitivePtr primitive_{nullptr};
  // Input and output abstracts for grad.
  std::vector<AbstractBasePtr> input_abs_{};
  AbstractBasePtr output_abs_{nullptr};
  CommHandlePtr comm_handle_{nullptr};
  // Forward output for grad.
  std::vector<tensor::TensorPtr> outputs_{};
  // clone inplace tensor, temp method, later we will erease it.
  tensor::TensorPtr clone_tensor_{nullptr};
  const DeviceContext *device_context_{nullptr};
  // Device address promise for multi-stage pipeline.
  std::vector<pynative::DeviceAddressPromisePtr> device_sync_promises_;
  // Op stream id
  size_t stream_id_{kDefaultStreamIndex};
  ValueSimpleInfoPtr output_value_simple_info_;
  inline static pynative::AbstractConverter kAbstractConverter;
  bool skip_tracker_{true};

 private:
  void CollectTrackerTensor(const ValuePtr &val, std::vector<tensor::TensorPtr> *tensors) {
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(val);
    if (tensor != nullptr) {
      tensors->emplace_back(tensor);
    }
  }

  void CollectTrackerTensor(const tensor::TensorPtr &tensor, std::vector<tensor::TensorPtr> *tensors) {
    if (tensor != nullptr) {
      tensors->emplace_back(tensor);
    }
  }

  void CollectTrackerTensor(const ValueTuplePtr &tensor_tuple, std::vector<tensor::TensorPtr> *tensors) {
    for (const auto &val : tensor_tuple->value()) {
      CollectTrackerTensor(val, tensors);
    }
  }

  template <typename T>
  void CollectTrackerTensor(const std::optional<T> &opt, std::vector<tensor::TensorPtr> *tensors) {
    if (opt.has_value()) {
      CollectTrackerTensor(opt.value(), tensors);
    }
  }

  template <typename T>
  void CollectTrackerTensor(const T &opt, std::vector<tensor::TensorPtr> *tensors) {
    return;
  }

  void TrackerOutputTensors(const std::vector<tensor::TensorPtr> &tensors) {
    PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([primitive = primitive_, tensors]() {
      for (const auto &tensor : tensors) {
        MS_EXCEPTION_IF_NULL(tensor);
        const auto &device_address = tensor->device_address();
        if (device_address == nullptr) {
          MS_LOG(WARNING) << "Tracker: input tensor device address is nullptr, primitive: " << primitive->name();
          continue;
        }
        if (device_address->GetPtr() == nullptr) {
          MS_LOG(WARNING) << "Tracker: input tensor device ptr is nullptr, primitive: " << primitive->name();
          continue;
        }
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
          MarkTensorAsOutput, "PyNative", device::GetDeviceNameByType(device_address->GetDeviceType()),
          device_address->GetPtr(), tensor->data_type(), tensor->shape(), tensor->storage_info());
      }
      device::tracker::CALL_MEMORY_TRACKER(DelNestedTask);
    }));
  }
};
using OpPtr = std::shared_ptr<OpRunner>;
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_RUNNER_H_
