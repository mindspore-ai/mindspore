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

#include "mindspore/ccsrc/pynative/backward/op_grad/view_grad.h"
#include <memory>
#include <vector>
#include "mindspore/core/include/utils/device_manager_conf.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "view/view_strides_calculator.h"
#include "utils/convert_utils.h"

namespace mindspore::pynative::autograd {
namespace {
inline void SetDeviceTarget() {
  const auto &device_target = DeviceManagerConf::GetInstance()->device_type();
  kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, device_target));
}
}  // namespace

ValuePtrList ViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto output = kernel::pyboost::reshape(grads[0]->cast<TensorPtr>(), self_shape_);
  return {output};
}

ValuePtrList TransposeBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  int64_t ndims = static_cast<int64_t>(perm_.size());
  std::vector<int64_t> invert_perm(ndims);
  for (int64_t i = 0; i < ndims; ++i) {
    invert_perm[ops::DynamicDimWrap(perm_[i], ndims)] = i;
  }
  auto output = kernel::pyboost::transpose(grads[0]->cast<TensorPtr>(), invert_perm);
  return {output};
}

ValuePtrList TransposeExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto output = kernel::pyboost::transpose_ext_view(grads[0]->cast<TensorPtr>(), dim0_, dim1_);
  return {output};
}

ValuePtrList SelectExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto size = PackToValue(self_shape_);
  auto grad = grads.at(0)->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad);
  auto dtype = std::make_shared<Int64Imm>(grad->data_type());
  auto grad_input = kernel::pyboost::zeros(size, dtype);
  auto select_part = kernel::pyboost::select_ext_view(grad_input, dim_, index_);
  (void)kernel::pyboost::inplace_copy(select_part, grad, std::make_shared<BoolImm>(true));
  return {grad_input};
}

ValuePtrList SliceExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto size = PackToValue(self_shape_);
  auto grad = grads.at(0)->cast<TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad);
  auto dtype = std::make_shared<Int64Imm>(grad->data_type());
  auto grad_input = kernel::pyboost::zeros(size, dtype);
  auto slice_part = kernel::pyboost::slice_ext_view(grad_input, dim_, start_, end_, step_);
  (void)kernel::pyboost::inplace_copy(slice_part, grad, std::make_shared<BoolImm>(true));
  return {grad_input};
}

namespace {
inline static ValuePtrList SplitWithSizeBackward(const ValuePtrList &grads, const std::vector<int64_t> &split_sizes,
                                                 const int64_t dim, const std::vector<int64_t> &self_shape,
                                                 const TypeId self_dtype) {
  auto grad_dtype = std::make_shared<Int64Imm>(static_cast<int64_t>(self_dtype));
  const int64_t ndim = static_cast<int64_t>(self_shape.size());
  const auto real_dim = ops::DynamicDimWrap(dim, ndim);

  std::vector<ValuePtr> real_grads(grads.size());
  for (size_t i = 0; i < grads.size(); ++i) {
    const auto &grad_i = grads[i];
    if (!grad_i->isa<None>()) {
      real_grads[i] = grad_i;
    } else {
      const auto &length = split_sizes[i];
      auto grad_size = self_shape;
      grad_size[real_dim] = length;
      real_grads[i] = kernel::pyboost::zeros(PackToValue(grad_size), grad_dtype);
    }
  }

  const auto all_grads_tuple = std::make_shared<ValueTuple>(real_grads);
  const auto concat_dim = std::make_shared<Int64Imm>(real_dim);
  auto grad_input = kernel::pyboost::concat(all_grads_tuple, concat_dim);
  return {grad_input};
}
}  // namespace

ValuePtrList SplitTensorBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  const int64_t ndim = static_cast<int64_t>(self_shape_.size());
  const auto real_dim = ops::DynamicDimWrap(dim_, ndim);
  int64_t num_splits = static_cast<int64_t>(grads.size());
  std::vector<int64_t> split_sizes(num_splits, split_size_);
  split_sizes[split_sizes.size() - 1] = split_size_ - (num_splits * split_size_ - self_shape_[real_dim]);
  return SplitWithSizeBackward(grads, split_sizes, real_dim, self_shape_, self_dtype_);
}

ValuePtrList SplitWithSizeBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  return SplitWithSizeBackward(grads, split_size_, dim_, self_shape_, self_dtype_);
}
}  // namespace mindspore::pynative::autograd
