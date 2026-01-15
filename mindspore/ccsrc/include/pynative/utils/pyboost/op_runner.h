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

// Pre-hook function type for tracing operator execution (called before op execution).
// Parameters: op_name (string), input values (vector of ValuePtr)
using OpRunnerPreHookFunc = std::function<void(const std::string &, const std::vector<ValuePtr> &)>;

// Post-hook function type for tracing operator execution (called after op execution).
// Parameters: op_name (string), input values (vector of ValuePtr), output value (ValuePtr)
using OpRunnerPostHookFunc = std::function<void(const std::string &, const std::vector<ValuePtr> &, const ValuePtr &)>;

// Hook entry with fast check function to avoid unnecessary conversions.
struct HookEntry {
  std::function<bool()> is_enabled;  // Fast check: returns true if hook needs to execute
};

// Pre-hook entry.
struct PreHookEntry : HookEntry {
  OpRunnerPreHookFunc handler;  // Actual execution function
};

// Post-hook entry.
struct PostHookEntry : HookEntry {
  OpRunnerPostHookFunc handler;  // Actual execution function
};

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

  // Register a pre-hook with is_enabled check function and handler function.
  // The is_enabled function is called first to avoid unnecessary conversions.
  static void RegisterPreHook(std::function<bool()> is_enabled, const OpRunnerPreHookFunc &handler) {
    pre_hooks_.emplace_back(PreHookEntry{{std::move(is_enabled)}, handler});
  }

  // Register a post-hook with is_enabled check function and handler function.
  // The is_enabled function is called first to avoid unnecessary conversions.
  static void RegisterPostHook(std::function<bool()> is_enabled, const OpRunnerPostHookFunc &handler) {
    post_hooks_.emplace_back(PostHookEntry{{std::move(is_enabled)}, handler});
  }

  // Clear all registered hooks.
  static void ClearPreHook() { pre_hooks_.clear(); }
  static void ClearPostHook() { post_hooks_.clear(); }
  static void ClearAllHooks() {
    pre_hooks_.clear();
    post_hooks_.clear();
  }

  // Check if any hook is registered.
  static bool HasPreHook() { return !pre_hooks_.empty(); }
  static bool HasPostHook() { return !post_hooks_.empty(); }

  // Call all registered pre-hook functions with op_name and input arguments.
  // First checks is_enabled for all hooks, only converts if any hook needs to execute.
  template <typename... Args>
  static void CallPreHook(const std::string &op_name, const Args &...args) {
    if (!HasPreHook()) {
      return;
    }
    // Skip if already inside a PostHook execution (prevents nested calls from internal ops).
    if (PostHookGuard::IsInPostHook()) {
      return;
    }
    // Fast path: check if any hook needs to execute
    bool any_need_execute = false;
    for (const auto &hook : pre_hooks_) {
      if (hook.is_enabled()) {
        any_need_execute = true;
        break;
      }
    }
    if (!any_need_execute) {
      return;
    }

    // Convert inputs only when needed
    std::vector<ValuePtr> inputs;
    inputs.reserve(sizeof...(args));
    CollectHookValues(&inputs, args...);

    // Execute hooks that need to run
    for (const auto &hook : pre_hooks_) {
      if (hook.is_enabled()) {
        hook.handler(op_name, inputs);
      }
    }
  }

  // Call all registered post-hook functions with op_name, input arguments, and output.
  // First checks is_enabled for all hooks, only converts if any hook needs to execute.
  // Skips execution if already inside a PostHook (prevents nested calls from internal ops).
  template <typename OutputType, typename... Args>
  static void CallPostHook(const std::string &op_name, const OutputType &output, const Args &...args) {
    if (!HasPostHook()) {
      return;
    }

    // Skip if already inside a PostHook execution (prevents nested calls from internal ops).
    if (PostHookGuard::IsInPostHook()) {
      return;
    }

    // Fast path: check if any hook needs to execute
    bool any_need_execute = false;
    for (const auto &hook : post_hooks_) {
      if (hook.is_enabled()) {
        any_need_execute = true;
        break;
      }
    }
    if (!any_need_execute) {
      return;
    }

    // Convert inputs and output only when needed
    std::vector<ValuePtr> inputs;
    inputs.reserve(sizeof...(args));
    CollectHookValues(&inputs, args...);
    ValuePtr output_value = ConvertOutputToValue(output);

    // Use PostHookGuard to mark we're inside PostHook execution.
    // This prevents internal ops from triggering nested PostHook calls.
    PostHookGuard guard;

    // Execute hooks that need to run
    for (const auto &hook : post_hooks_) {
      if (hook.is_enabled()) {
        hook.handler(op_name, inputs, output_value);
      }
    }
  }

  // Convert output to ValuePtr (handles single tensor, tuple of tensors, vector of tensors, etc.)
  static ValuePtr ConvertOutputToValue(const tensor::TensorPtr &tensor) { return tensor; }

  static ValuePtr ConvertOutputToValue(const ValuePtr &value) { return value; }

  template <typename... Args>
  static ValuePtr ConvertOutputToValue(const std::tuple<Args...> &tuple) {
    std::vector<ValuePtr> values;
    values.reserve(sizeof...(Args));
    std::apply([&values](const Args &...args) { (CollectSingleHookValue(&values, args), ...); }, tuple);
    return std::make_shared<ValueTuple>(values);
  }

  template <typename T>
  static ValuePtr ConvertOutputToValue(const std::vector<T> &vec) {
    std::vector<ValuePtr> values;
    values.reserve(vec.size());
    for (const auto &item : vec) {
      CollectSingleHookValue(&values, item);
    }
    return std::make_shared<ValueTuple>(values);
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
  // Static hook functions for hyper offload tracing (supports multiple hooks).
  inline static std::vector<PreHookEntry> pre_hooks_;
  inline static std::vector<PostHookEntry> post_hooks_;

 private:
  // Helper functions for collecting hook arguments into ValuePtr vector (shared by pre/post hooks).
  static void CollectHookValues(std::vector<ValuePtr> *values) {}

  template <typename T, typename... Rest>
  static void CollectHookValues(std::vector<ValuePtr> *values, const T &first, const Rest &...rest) {
    CollectSingleHookValue(values, first);
    CollectHookValues(values, rest...);
  }

  // Overload for shared_ptr<Value-derived> (e.g., BoolImmPtr)
  template <typename T>
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const std::shared_ptr<T> &val,
                                     typename std::enable_if<std::is_base_of_v<Value, T>>::type * = nullptr) {
    if (val != nullptr) {
      values->emplace_back(std::static_pointer_cast<Value>(val));
    }
  }

  // Overload for ValuePtr
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const ValuePtr &val) {
    if (val != nullptr) {
      values->emplace_back(val);
    }
  }

  // Overload for TensorPtr
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const tensor::TensorPtr &tensor) {
    if (tensor != nullptr) {
      values->emplace_back(std::static_pointer_cast<Value>(tensor));
    }
  }

  // Overload for optional types
  template <typename T>
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const std::optional<T> &opt) {
    if (opt.has_value()) {
      CollectSingleHookValue(values, opt.value());
    } else {
      values->emplace_back(kNone);
    }
  }

  // Overload for vector types
  template <typename T>
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const std::vector<T> &vec) {
    for (const auto &item : vec) {
      CollectSingleHookValue(values, item);
    }
  }

  // Overload for arithmetic types (int, int64_t, float, double, etc.)
  // This handles view ops that take raw numeric arguments like split_size, dim
  template <typename T>
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, T val,
                                     typename std::enable_if<std::is_arithmetic_v<T>>::type * = nullptr) {
    values->emplace_back(MakeValue(val));
  }

  // Overload for std::string
  static void CollectSingleHookValue(std::vector<ValuePtr> *values, const std::string &val) {
    values->emplace_back(MakeValue(val));
  }

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
