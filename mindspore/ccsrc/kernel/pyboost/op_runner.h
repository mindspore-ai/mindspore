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
#include "ir/scalar.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "ir/tensor.h"
#include "include/backend/visible.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
// OpRunner is a base class for operators.
// OpRunner records the operator's input abstract,
// output abstract and output Tensors for grad,
// and it also contains several functional methods for the operator to run.
class BACKEND_EXPORT OpRunner : public std::enable_shared_from_this<OpRunner> {
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
  void set_input_abs(const std::vector<AbstractBasePtr> &input_abs) { input_abs_ = input_abs; }
  const AbstractBasePtr &output_abs() const { return output_abs_; }
  void set_output_abs(const AbstractBasePtr &output_abs) { output_abs_ = output_abs; }
  const DeviceContext *device_context() const { return device_context_; }
  const std::vector<pynative::DeviceAddressPromisePtr> &device_sync_promises() const { return device_sync_promises_; }
  const std::vector<tensor::TensorPtr> &outputs() const { return outputs_; }
  void set_outputs(const std::vector<tensor::TensorPtr> &outputs) { outputs_ = outputs; }
  void set_stream_id(size_t stream_id) { stream_id_ = stream_id; }
  size_t stream_id() const { return stream_id_; }
  ValueSimpleInfoPtr output_value_simple_info() const { return output_value_simple_info_; }

  const tensor::TensorPtr &output(const size_t &idx) {
    if (idx >= outputs_.size()) {
      MS_LOG(EXCEPTION) << "idx is out of bounds, idx:" << idx << ", outputs_.size():" << outputs_.size();
    }
    return outputs_[idx];
  }

  // For view op used
  void SetOutputAbstract() { output_abs_ = AbstractConvertFunc::ConvertAbstract(output(kIndex0)); }

  // Member function for Infer and creating output tensors.
  template <typename... T>
  void InferOutput(T &... args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostInferOutput,
                                       primitive_->name(), false);
    if (output_value_simple_info_ = ops::InferBySimple(primitive_, args...); output_value_simple_info_ != nullptr) {
      MS_LOG(DEBUG) << "Op " << primitive_->name() << " infer by simple, get output "
                    << ValueSimpleInfoToString(*output_value_simple_info_);
      PyBoostUtils::CreateOutputTensor(output_value_simple_info_, &outputs_);
      return;
    }

    (input_abs_.emplace_back(AbstractConvertFunc::ConvertAbstract(args)), ...);
    output_abs_ = PyBoostUtils::InferByOpDef(primitive_, input_abs_);
    MS_EXCEPTION_IF_NULL(output_abs_);
    MS_LOG(DEBUG) << "PyBoost infer by abstract, get output " << output_abs_->ToString();
    PyBoostUtils::CreateOutputTensor(output_abs_, &outputs_);
    AbstractConvertFunc::CacheAbstract(output_abs_);
  }

  // A static function used for the "customize" operator to generate the operator's output Tensor.
  template <typename... T>
  static void InferOpOutput(const std::shared_ptr<OpRunner> &op, T &... args) {
    op->InferOutput(args...);
  }

 protected:
  // Op primitive, may delete latter.
  PrimitivePtr primitive_{nullptr};
  // Input and output abstracts for grad.
  std::vector<AbstractBasePtr> input_abs_{};
  AbstractBasePtr output_abs_{nullptr};
  // Forward output for grad.
  std::vector<tensor::TensorPtr> outputs_{};
  const DeviceContext *device_context_{nullptr};
  // Device address promise for multi-stage pipeline.
  std::vector<pynative::DeviceAddressPromisePtr> device_sync_promises_;
  // Op stream id
  size_t stream_id_{kDefaultStreamIndex};
  ValueSimpleInfoPtr output_value_simple_info_;
};
using OpPtr = std::shared_ptr<OpRunner>;
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_RUNNER_H_
