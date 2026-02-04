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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_OP_FUNCTION_CUSTOMIZE_VIEW_GRAD_IMPL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_OP_FUNCTION_CUSTOMIZE_VIEW_GRAD_IMPL_H_

#include <vector>
#include <memory>
#include "mindspore/ccsrc/pynative/backward/op_grad/view_grad.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"

namespace mindspore::pynative {
inline void DoGradReshapeImpl(const mindspore::tensor::TensorPtr &output, const mindspore::tensor::TensorPtr &input,
                              const std::vector<int64_t> &shape) {
  auto make_func = [&input]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::ViewBackwardNode>("Reshape", input->shape());
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradViewImpl(const mindspore::tensor::TensorPtr &output, const mindspore::tensor::TensorPtr &input,
                           const std::vector<int64_t> &shape) {
  auto make_func = [&input]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::ViewBackwardNode>("View", input->shape());
    return backward_node;
  };
  bool is_safe = kernel::pyboost::OpRunStatus::Get().IsSafeView();
  pynative::autograd::DoViewGrad(input, output, make_func, is_safe);
}

inline void DoGradTransposeImpl(const mindspore::tensor::TensorPtr &output, const mindspore::tensor::TensorPtr &input,
                                const std::vector<int64_t> &input_perm) {
  auto make_func = [&input_perm]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::TransposeBackwardNode>("Transpose", input_perm);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradTransposeViewImpl(const mindspore::tensor::TensorPtr &output,
                                    const mindspore::tensor::TensorPtr &input, const std::vector<int64_t> &input_perm) {
  auto make_func = [&input_perm]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::TransposeBackwardNode>("TransposeView", input_perm);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradExpandDimsImpl(const mindspore::tensor::TensorPtr &output,
                                 const mindspore::tensor::TensorPtr &input_x, const int64_t &axis) {
  auto make_func = [&input_x]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::ViewBackwardNode>("ExpandDims", input_x->shape());
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input_x, output, make_func);
}

inline void DoGradExpandDimsViewImpl(const mindspore::tensor::TensorPtr &output,
                                     const mindspore::tensor::TensorPtr &input_x, const int64_t &axis) {
  auto make_func = [&input_x]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::ViewBackwardNode>("ExpandDimsView", input_x->shape());
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input_x, output, make_func);
}

inline void DoGradSelectExtViewImpl(const mindspore::tensor::TensorPtr &output,
                                    const mindspore::tensor::TensorPtr &input, const int64_t &dim,
                                    const int64_t &index) {
  auto make_func = [&input, &dim, &index]() -> BackwardNodePtr {
    auto backward_node =
      std::make_shared<pynative::autograd::SelectExtViewBackwardNode>("SelectExtView", input->shape(), dim, index);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradSliceExtViewImpl(const mindspore::tensor::TensorPtr &output,
                                   const mindspore::tensor::TensorPtr &input, const int64_t &dim, const int64_t &start,
                                   const int64_t &end, const int64_t &step) {
  auto make_func = [&input, &dim, &start, &end, &step]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::SliceExtViewBackwardNode>("SliceExtView", input->shape(),
                                                                                        dim, start, end, step);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradTransposeExtViewImpl(const mindspore::tensor::TensorPtr &output,
                                       const mindspore::tensor::TensorPtr &input, const int64_t &dim0,
                                       const int64_t &dim1) {
  auto make_func = [&dim0, &dim1]() -> BackwardNodePtr {
    auto backward_node =
      std::make_shared<pynative::autograd::TransposeExtViewBackwardNode>("TransposeExtView", dim0, dim1);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradSplitTensorImpl(const std::vector<mindspore::tensor::TensorPtr> &output,
                                  const mindspore::tensor::TensorPtr &input, const int64_t &split_size,
                                  const int64_t &dim) {
  auto make_func = [&input, split_size, dim, output_size = output.size()]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::SplitTensorBackwardNode>(
      "SplitTensor", output_size, input->shape(), input->data_type(), split_size, dim);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradSplitWithSizeImpl(const std::vector<mindspore::tensor::TensorPtr> &output,
                                    const mindspore::tensor::TensorPtr &input, const std::vector<int64_t> &split_size,
                                    const int64_t &dim) {
  auto make_func = [&input, &split_size, dim, output_size = output.size()]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::SplitWithSizeBackwardNode>(
      "SplitWithSize", output_size, input->shape(), input->data_type(), split_size, dim);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradSqueezeImpl(const mindspore::tensor::TensorPtr &output, const mindspore::tensor::TensorPtr &input_x,
                              const std::vector<int64_t> &axis) {
  auto make_func = [&input_x]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::ViewBackwardNode>("Squeeze", input_x->shape());
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input_x, output, make_func);
}

inline void DoGradSplitTensorViewImpl(const std::vector<mindspore::tensor::TensorPtr> &output,
                                      const mindspore::tensor::TensorPtr &input, const int64_t &split_size,
                                      const int64_t &dim) {
  auto make_func = [&input, split_size, dim, output_size = output.size()]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::SplitTensorBackwardNode>(
      "SplitTensorView", output_size, input->shape(), input->data_type(), split_size, dim);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}

inline void DoGradSplitWithSizeViewImpl(const std::vector<mindspore::tensor::TensorPtr> &output,
                                        const mindspore::tensor::TensorPtr &input,
                                        const std::vector<int64_t> &split_size, const int64_t &dim) {
  auto make_func = [&input, &split_size, dim, output_size = output.size()]() -> BackwardNodePtr {
    auto backward_node = std::make_shared<pynative::autograd::SplitWithSizeBackwardNode>(
      "SplitWithSizeView", output_size, input->shape(), input->data_type(), split_size, dim);
    return backward_node;
  };
  pynative::autograd::DoViewGrad(input, output, make_func);
}
}  // namespace mindspore::pynative
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_OP_FUNCTION_CUSTOMIZE_VIEW_GRAD_IMPL_H_
