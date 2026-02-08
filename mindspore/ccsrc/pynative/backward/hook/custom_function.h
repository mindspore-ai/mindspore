/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_

#include <string>
#include <utility>
#include <memory>
#include <vector>
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "include/utils/pynative/variable.h"
#include "pynative/backward/grad_utils.h"
#include "pynative/backward/saved_tensor.h"

namespace mindspore {
namespace pynative {
namespace autograd {
struct CustomContext {
  // Custom cell name
  std::string cell_name;
  // Cell inputs
  ValuePtrList inputs;
  // Cell output
  ValuePtr output;
  // Input grad type
  std::vector<InputType> input_value_grad_type;
  // Custom bprop function
  py::function bprop_fn;
  // Recompute weight size
  size_t weight_size{0};
  // Whether the cell is recompute cell
  bool is_recompute;

  std::unordered_set<int64_t> used_inputs_set;
  ~CustomContext() {
    py::gil_scoped_acquire gil_acquire;
    bprop_fn = py::object();
  }
};

class CustomBackward : public BackwardNode {
 public:
  CustomBackward(string name, py::function bprop_fn, abstract::AbstractBasePtr out_abstract, bool is_recompute = false,
                 size_t output_size = 1)
      : BackwardNode(std::move(name), output_size),
        bprop_fn_(std::move(bprop_fn)),
        out_abstract_(std::move(out_abstract)),
        is_recompute_(is_recompute) {}
  ~CustomBackward() override;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;
  void Release() override;
  void SetSavedValues(const ValuePtrList &saved_values) { saved_values_ = saved_values; }

 private:
  py::function bprop_fn_;
  ValuePtrList saved_values_;
  abstract::AbstractBasePtr out_abstract_;
  bool is_recompute_{false};
};

class PyBackwardNode : public BackwardNode {
 public:
  PyBackwardNode(string name, py::object backward_fn, py::object obj, size_t output_size = 1)
      : BackwardNode(std::move(name), output_size), backward_fn_(std::move(backward_fn)), obj_(std::move(obj)) {}
  ~PyBackwardNode() override;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;
  void Release() override;
  void SetOutAbstract(abstract::AbstractBasePtr out_abstract) { out_abstract_ = std::move(out_abstract); }
  void SetOutputSize(size_t output_size) { output_size_ = output_size; }
  void SetSavedTensors(SavedTensorPtrList saved_tensors) { saved_tensors_ = std::move(saved_tensors); }
  const SavedTensorPtrList &GetSavedTensors() { return saved_tensors_; }
  py::object obj() { return obj_; }

 private:
  py::object backward_fn_;
  py::object obj_;
  abstract::AbstractBasePtr out_abstract_;
  SavedTensorPtrList saved_tensors_;
};
using PyBackwardNodePtr = std::shared_ptr<PyBackwardNode>;

}  // namespace autograd
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_CUSTOM_FUNCTION_H_
