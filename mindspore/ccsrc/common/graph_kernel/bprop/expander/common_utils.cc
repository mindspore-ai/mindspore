/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/bprop/expander/common_utils.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <limits>

#include "ops/core_ops.h"

namespace mindspore::expander::bprop {
namespace {
NodePtr ReduceSumWithCast(const BpropIRBuilder *ib, const NodePtr &dx, const std::vector<int64_t> &axis) {
  auto dx_origin_dtype = ib->GetDtype(dx)->type_id();
  if (dx_origin_dtype == TypeId::kNumberTypeInt16 || dx_origin_dtype == TypeId::kNumberTypeInt32 ||
      dx_origin_dtype == TypeId::kNumberTypeInt64) {
    auto dx_fp32 = ib->Cast(dx, kFloat32);
    return ib->Emit("ReduceSum", {dx_fp32}, {{"axis", MakeValue(axis)}, {"keep_dims", MakeValue(false)}});
  }
  return ib->Emit("ReduceSum", {dx}, {{"axis", MakeValue(axis)}, {"keep_dims", MakeValue(false)}});
}

void ComputeReduceIndex(const std::vector<int64_t> &x_rev, const std::vector<int64_t> &y_rev,
                        std::vector<int64_t> *grad_x_reduce_idx, std::vector<int64_t> *grad_y_reduce_idy) {
  MS_EXCEPTION_IF_NULL(grad_x_reduce_idx);
  MS_EXCEPTION_IF_NULL(grad_y_reduce_idy);
  const size_t n = x_rev.size();
  if (y_rev.size() < n) {
    MS_LOG(EXCEPTION) << "The size of y_rev is less than the size of x_rev.";
  }
  for (size_t i = 0; i < n; ++i) {
    const int64_t x_i = x_rev[i];
    const int64_t y_i = y_rev[i];
    const int64_t reduce_idx = SizeToLong(n - 1 - i);
    if (x_i == y_i) {
      if (x_i == 1) {
        grad_x_reduce_idx->push_back(reduce_idx);
        grad_y_reduce_idy->push_back(reduce_idx);
      }
    } else if (x_i == 1) {
      grad_x_reduce_idx->push_back(reduce_idx);
    } else if (y_i == 1) {
      grad_y_reduce_idy->push_back(reduce_idx);
    } else {
      MS_LOG(EXCEPTION) << "not compatible shape input for BroadcastGradientArgs.";
    }
  }

  std::reverse(grad_x_reduce_idx->begin(), grad_x_reduce_idx->end());
  std::reverse(grad_y_reduce_idy->begin(), grad_y_reduce_idy->end());
}
}  // namespace

std::vector<std::vector<int64_t>> BroadcastGradientArgs(const std::vector<int64_t> &x_shape,
                                                        const std::vector<int64_t> &y_shape) {
  std::vector<std::vector<int64_t>> bc_axis;
  if (x_shape == y_shape) {
    (void)bc_axis.emplace_back(std::vector<int64_t>{});
    (void)bc_axis.emplace_back(std::vector<int64_t>{});
    return bc_axis;
  }
  std::vector<int64_t> reverse_x;
  std::vector<int64_t> reverse_y;

  (void)std::transform(x_shape.rbegin(), x_shape.rend(), std::back_inserter(reverse_x),
                       [](const int64_t &c) { return c; });
  (void)std::transform(y_shape.rbegin(), y_shape.rend(), std::back_inserter(reverse_y),
                       [](const int64_t &c) { return c; });

  if (reverse_x.size() > reverse_y.size()) {
    reverse_y.resize(reverse_x.size(), 1);
  } else {
    reverse_x.resize(reverse_y.size(), 1);
  }

  std::vector<int64_t> grad_x_reduce_idx;
  std::vector<int64_t> grad_y_reduce_idy;
  ComputeReduceIndex(reverse_x, reverse_y, &grad_x_reduce_idx, &grad_y_reduce_idy);

  (void)bc_axis.emplace_back(std::move(grad_x_reduce_idx));
  (void)bc_axis.emplace_back(std::move(grad_y_reduce_idy));
  return bc_axis;
}

NodePtrList BinopGradCommon(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                            const NodePtr &dy) {
  // Common grad definition for binary operations.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  auto shape_x = ib->GetShape(x);
  auto shape_y = ib->GetShape(y);
  auto reduce_dx = dx;
  auto reduce_dy = dy;

  std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(shape_x, shape_y);
  if (!bc_axis[0].empty()) {
    auto dx_shape = ib->GetShape(dx);
    if (!dx_shape.empty()) {
      reduce_dx = ReduceSumWithCast(ib, reduce_dx, bc_axis[0]);
    }
    reduce_dx = ib->Reshape(reduce_dx, shape_x);
  }

  if (!bc_axis[1].empty()) {
    auto dy_shape = ib->GetShape(dy);
    if (!dy_shape.empty()) {
      reduce_dy = ReduceSumWithCast(ib, reduce_dy, bc_axis[1]);
    }
    reduce_dy = ib->Reshape(reduce_dy, shape_y);
  }
  return {reduce_dx, reduce_dy};
}

NodePtrList BinopGradCommonWithShift(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dx,
                                     const NodePtr &dy, size_t shift) {
  // Common grad definition for binary operations with shift.
  // The function is usually used in backprop op to reduce additional dimensions
  // created by broadcasting.
  auto shape_x = ib->GetShape(x);
  auto shape_y = ib->GetShape(y);
  auto reduce_dx = dx;
  auto reduce_dy = dy;
  std::vector<int64_t> broadcast_shape_of_x;
  std::vector<int64_t> broadcast_shape_of_y;
  for (size_t i = 0; i < shape_x.size() - shift; i++) {
    broadcast_shape_of_x.push_back(shape_x[i]);
  }
  for (size_t i = 0; i < shape_y.size() - shift; i++) {
    broadcast_shape_of_y.push_back(shape_y[i]);
  }

  std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(broadcast_shape_of_x, broadcast_shape_of_y);
  if (!bc_axis[0].empty()) {
    auto dx_shape = ib->GetShape(dx);
    if (!dx_shape.empty()) {
      reduce_dx = ReduceSumWithCast(ib, reduce_dx, bc_axis[0]);
    }
    reduce_dx = ib->Reshape(reduce_dx, shape_x);
  }

  if (!bc_axis[1].empty()) {
    auto dy_shape = ib->GetShape(dy);
    if (!dy_shape.empty()) {
      reduce_dy = ReduceSumWithCast(ib, reduce_dy, bc_axis[1]);
    }
    reduce_dy = ib->Reshape(reduce_dy, shape_y);
  }
  return {reduce_dx, reduce_dy};
}

std::vector<int64_t> Range(int64_t start, int64_t stop, int64_t step) {
  auto size = (stop - start) / step;
  size = ((stop - start) % step == 0) ? size : size + 1;
  std::vector<int64_t> range(size);
  std::generate(range.begin(), range.end(), [n = start - step, step]() mutable {
    n = n + step;
    return n;
  });
  return range;
}

std::vector<int64_t> Range(int64_t stop) { return Range(0, stop); }

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  std::vector<int64_t> reverse_axis;
  for (int64_t i = 0; i < rk; ++i) {
    reverse_axis.emplace_back(i);
  }
  reverse_axis[axis] = rk - 1;
  reverse_axis[rk - 1] = axis;
  return reverse_axis;
}

NodePtr GetEps(const BpropIRBuilder *ib, const TypePtr &type) {
  switch (type->type_id()) {
    case kNumberTypeFloat16:
      return ib->Tensor(0.000977, type);
    case kNumberTypeFloat32:
      return ib->Tensor(std::numeric_limits<float>::epsilon(), type);
    case kNumberTypeFloat64:
      return ib->Tensor(std::numeric_limits<double>::epsilon(), type);
    default:
      return ib->Tensor(0, type);
  }
}
}  // namespace mindspore::expander::bprop
