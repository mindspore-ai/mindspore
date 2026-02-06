/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_

#include <vector>
#include <string>
#include <utility>

#include "tools/profiler/profiler.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "pynative/backward/grad_utils.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore::pynative::autograd {
template <typename Func>
void DoViewGrad(const TensorPtr &input_tensor, const TensorPtr &output_tensor, const Func &make_func,
                bool is_safe = true) {
  static const std::string kDoGradName = "DoViewGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);
  const bool requires_grad = kernel::pyboost::OpRunStatus::Get().RequireGrad();
  is_safe ? AutoGradUtil::MakeOutput(requires_grad, output_tensor, input_tensor)
          : AutoGradUtil::MakeOutput(requires_grad, output_tensor);

  if (requires_grad) {
    runtime::Pipeline::Get().WaitBpropStage();
    if (AutoGradUtil::NeedGrad(input_tensor)) {
      auto view_grad_node = make_func();
      UpdateNextEdges(view_grad_node, {input_tensor});
      autograd::impl::SetTensorGradMetaData(output_tensor, view_grad_node, 0);
    }
    UpdateVersion(output_tensor);
  }
}

template <typename Func>
void DoViewGrad(const TensorPtr &input_tensor, const std::vector<TensorPtr> &output_tensors, const Func &make_func,
                bool is_safe = true) {
  static const std::string kDoGradName = "DoViewGrad";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeFrontendTask,
                                     kDoGradName, false);
  const bool requires_grad = kernel::pyboost::OpRunStatus::Get().RequireGrad();
  (void)AutoGradUtil::MakeOutput(requires_grad, output_tensors, input_tensor);

  if (requires_grad) {
    runtime::Pipeline::Get().WaitBpropStage();
    if (AutoGradUtil::NeedGrad(input_tensor)) {
      auto view_grad_node = make_func();
      UpdateNextEdges(view_grad_node, {input_tensor});
      for (size_t i = 0; i < output_tensors.size(); ++i) {
        const auto &output_tensor = output_tensors[i];
        autograd::impl::SetTensorGradMetaData(output_tensor, view_grad_node, i);
      }
    }
    (void)std::for_each(output_tensors.begin(), output_tensors.end(),
                        [](const TensorPtr &tensor) { UpdateVersion(tensor); });
  }
}

class ViewBackwardNode : public BackwardNode {
 public:
  ViewBackwardNode(std::string name, std::vector<int64_t> self_shape)
      : BackwardNode(std::move(name)), self_shape_(std::move(self_shape)) {}
  ~ViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
};

class TransposeBackwardNode : public BackwardNode {
 public:
  TransposeBackwardNode(std::string name, std::vector<int64_t> perm)
      : BackwardNode(std::move(name)), perm_(std::move(perm)) {}
  ~TransposeBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> perm_;
};

class TransposeExtViewBackwardNode : public BackwardNode {
 public:
  TransposeExtViewBackwardNode(std::string name, int64_t dim0, int64_t dim1)
      : BackwardNode(std::move(name)), dim0_(dim0), dim1_(dim1) {}
  ~TransposeExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  int64_t dim0_;
  int64_t dim1_;
};

class SelectExtViewBackwardNode : public BackwardNode {
 public:
  SelectExtViewBackwardNode(std::string name, std::vector<int64_t> self_shape, int64_t dim, int64_t index)
      : BackwardNode(std::move(name)),
        self_shape_(std::move(self_shape)),
        dim_(std::move(dim)),
        index_(std::move(index)) {}
  ~SelectExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  int64_t dim_;
  int64_t index_;
};

class SliceExtViewBackwardNode : public BackwardNode {
 public:
  SliceExtViewBackwardNode(std::string name, std::vector<int64_t> self_shape, int64_t dim, int64_t start, int64_t end,
                           int64_t step)
      : BackwardNode(std::move(name)),
        self_shape_(std::move(self_shape)),
        dim_(std::move(dim)),
        start_(std::move(start)),
        end_{std::move(end)},
        step_{std::move(step)} {}
  ~SliceExtViewBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t step_;
};

class SplitTensorBackwardNode : public BackwardNode {
 public:
  SplitTensorBackwardNode(std::string name, size_t output_size, std::vector<int64_t> self_shape, TypeId self_dtype,
                          int64_t split_size, int64_t dim)
      : BackwardNode(std::move(name), output_size),
        self_shape_(std::move(self_shape)),
        self_dtype_(self_dtype),
        split_size_(split_size),
        dim_(dim) {}
  ~SplitTensorBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  TypeId self_dtype_;
  int64_t split_size_;
  int64_t dim_;
};

class SplitWithSizeBackwardNode : public BackwardNode {
 public:
  SplitWithSizeBackwardNode(std::string name, size_t output_size, std::vector<int64_t> self_shape, TypeId self_dtype,
                            std::vector<int64_t> split_size, int64_t dim)
      : BackwardNode(std::move(name), output_size),
        self_shape_(std::move(self_shape)),
        self_dtype_(self_dtype),
        split_size_(std::move(split_size)),
        dim_(dim) {}
  ~SplitWithSizeBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;

 private:
  std::vector<int64_t> self_shape_;
  TypeId self_dtype_;
  std::vector<int64_t> split_size_;
  int64_t dim_;
};
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_VIEW_GRAD_H_
