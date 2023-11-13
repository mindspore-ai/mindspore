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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BASE_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BASE_H_

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
#include "kernel/pyboost/py_boost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using GradFunc = std::function<void()>;

class BACKEND_EXPORT Op : public std::enable_shared_from_this<Op> {
 public:
  Op() = default;
  virtual ~Op() = default;

  void set_grad_func(GradFunc &&grad_func) { grad_func_ = std::move(grad_func); }

  std::shared_ptr<Op> get_op() { return shared_from_this(); }

  void set_primitive(const PrimitivePtr &primitive) { primitive_ = primitive; }
  const PrimitivePtr &primitive() const { return primitive_; }

  const std::vector<tensor::TensorPtr> &outputs() const { return outputs_; }

  const std::vector<AbstractBasePtr> &input_abs() const { return input_abs_; }
  const AbstractBasePtr &output_abs() const { return output_abs_; }
  void set_device_context(DeviceContext *device_context) { device_context_ = device_context; }
  DeviceContext *device_context() const { return device_context_; }

  void DoGrad() {
    MS_EXCEPTION_IF_NULL(grad_func_);
    grad_func_();
  }

  const tensor::TensorPtr &output(const size_t &idx) {
    if (idx >= outputs_.size()) {
      MS_LOG(EXCEPTION) << "idx is out of bounds, idx:" << idx << ", outputs_.size():" << outputs_.size();
    }
    return outputs_[idx];
  }

  template <typename T>
  static AbstractBasePtr ConvertAbstract(const std::optional<T> &t) {
    if (!t.has_value()) {
      return kNone->ToAbstract();
    }
    return t.value()->ToAbstract();
  }

  static AbstractBasePtr ConvertAbstract(const ValuePtr &t) { return t->ToAbstract(); }

  template <typename... T>
  inline void InferOutput(T &... args) {
    input_abs_.clear();
    (input_abs_.emplace_back(ConvertAbstract(args)), ...);
    output_abs_ = PyBoostUtils::InferByOpDef(primitive_, input_abs_);
    MS_EXCEPTION_IF_NULL(output_abs_);
    MS_LOG(DEBUG) << "PyBoost infer output " << output_abs_->ToString();
    outputs_.clear();
    PyBoostUtils::CreateOutputTensor(output_abs_, &outputs_, &device_sync_promises_);
  }

  template <typename... T>
  static void InferOpOutput(const std::shared_ptr<Op> &op, T &... args) {
    (op->input_abs_.emplace_back(ConvertAbstract(args)), ...);
    op->output_abs_ = PyBoostUtils::InferByOpDef(op->primitive(), op->input_abs_);
    PyBoostUtils::CreateOutputTensor(op->output_abs_, &op->outputs_, &op->device_sync_promises_);
  }

  tensor::TensorPtr Contiguous(const tensor::TensorPtr &input_tensor) { return ContiguousTensor(input_tensor); }

  template <typename... T>
  void DeviceMalloc(T &... args) {
    PrepareOpInputs(device_context_, args...);
    PrepareOpOutputs(device_context_, outputs_);
  }

  const std::vector<pynative::DeviceAddressPromisePtr> &device_sync_promises() const { return device_sync_promises_; }

 protected:
  std::vector<tensor::TensorPtr> outputs_{};
  GradFunc grad_func_;
  PrimitivePtr primitive_;
  // Save abstract for grad.
  std::vector<AbstractBasePtr> input_abs_{};
  AbstractBasePtr output_abs_{nullptr};
  DeviceContext *device_context_{nullptr};
  std::vector<pynative::DeviceAddressPromisePtr> device_sync_promises_;
};
using OpPtr = std::shared_ptr<Op>;

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_OP_BASE_H_
