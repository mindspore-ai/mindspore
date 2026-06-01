/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include "ms_extension/all.h"
#include "include/utils/pynative/dispatch/ms_dispatch_py_obj.h"

namespace custom {
ms::Tensor GenReduceSumResultTensor(const ms::Tensor &t, const std::vector<int64_t> &axis, bool keepdims,
                                    ms::TypeId type_id) {
  auto s = t.shape();
  size_t n = s.size();
  ShapeVector out_shape;
  if (!axis.empty()) {
    std::set<size_t> axis_set;
    for (auto x : axis) {
      axis_set.insert(static_cast<size_t>(x >= 0 ? x : x + static_cast<int64_t>(n)));
    }
    for (size_t i = 0; i < n; i++) {
      if (axis_set.count(i) > 0) {
        if (keepdims) {
          out_shape.push_back(1LL);
        }
      } else {
        out_shape.push_back(s[i]);
      }
    }
  } else {
    if (keepdims) {
      out_shape.resize(n, 1);
    }
  }
  return ms::Tensor(type_id, out_shape);
}

ms::Tensor npu_abs_reduce_sum(const ms::Tensor &x, std::optional<std::vector<int64_t>> axis, bool keepdims,
                              std::optional<int64_t> dtype) {
  auto type_id = dtype.has_value() ? static_cast<ms::TypeId>(dtype.value()) : x.data_type();
  std::vector<int64_t> axis_vec = axis.has_value() ? axis.value() : std::vector<int64_t>();
  auto result = GenReduceSumResultTensor(x, axis_vec, keepdims, type_id);
  auto y = ms::Tensor(mindspore::kernel::pyboost::abs(x.tensor()));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ReduceSum");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnReduceSum, y, axis_vec, keepdims, type_id, result));
  runner->Run({y}, {result});
  return result;
}

static ShapeVector InferArgMinShape(const ShapeVector &in_shape, int64_t dim, bool keep_dims) {
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  if (rank == 0) {
    return in_shape;
  }

  int64_t axis = (dim < 0) ? (dim + rank) : dim;
  if (axis < 0 || axis >= rank) {
    MS_LOG(EXCEPTION) << "Infer shape failed";
  }

  ShapeVector out_shape;
  out_shape.reserve(keep_dims ? rank : rank - 1);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  return out_shape;
}

ms::Tensor GenArgMinResultTensor(const ms::Tensor &t, int64_t dim, bool keep_dim, ms::TypeId type_id) {
  ShapeVector in_shape = t.shape();
  ShapeVector out_shape = InferArgMinShape(in_shape, dim, keep_dim);
  return ms::Tensor(type_id, out_shape);
}

ms::Tensor npu_arg_min(const ms::Tensor &x, int64_t dim, bool keep_dim) {
  auto result = GenArgMinResultTensor(x, dim, keep_dim, ms::TypeId::kNumberTypeInt64);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ReduceSum");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, dim, keep_dim, result));
  runner->Run({x}, {result});
  return result;
}

ms::Tensor npu_cast(const ms::Tensor &x, int64_t dst_type_id) {
  auto type_id = static_cast<ms::TypeId>(dst_type_id);
  auto result = ms::Tensor(type_id, x.shape());
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnCast");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnCast, x, type_id, result));
  runner->Run({x}, {result});
  return result;
}

std::tuple<ShapeVector, ShapeVector, ShapeVector> InferBatchNormShape(const ShapeVector &x_shape) {
  const int64_t rank = static_cast<int64_t>(x_shape.size());
  if (rank < 2) {
    MS_LOG(EXCEPTION) << "BatchNorm input rank must >= 2";
  }
  const int64_t channels = x_shape[1];
  ShapeVector saved_shape{channels};
  return {x_shape, saved_shape, saved_shape};
}

std::tuple<ms::Tensor, ms::Tensor, ms::Tensor> GenBatchNormResultTensors(const ms::Tensor &x, ms::TypeId dtype) {
  ShapeVector x_shape = x.shape();
  auto [y_shape, sm_shape, sv_shape] = InferBatchNormShape(x_shape);

  ms::Tensor y(dtype, y_shape);
  ms::Tensor saved_mean(ms::TypeId::kNumberTypeFloat32, sm_shape);
  ms::Tensor saved_var(ms::TypeId::kNumberTypeFloat32, sv_shape);
  return {std::move(y), std::move(saved_mean), std::move(saved_var)};
}

std::vector<ms::Tensor> npu_batch_norm(const ms::Tensor &x, const ms::Tensor &scale, const ms::Tensor &bias,
                                       const ms::Tensor &mean, const ms::Tensor &var, bool training, double momentum,
                                       double eps) {
  auto [y, save_mean, save_var] = GenBatchNormResultTensors(x, x.data_type());

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnBatchNorm");
  runner->SetLaunchFunc(
    LAUNCH_ACLNN_FUNC(aclnnBatchNorm, x, scale, bias, mean, var, training, momentum, eps, y, save_mean, save_var));
  runner->Run({x, scale, bias, mean, var}, {y, save_mean, save_var});
  return {y, save_mean, save_var};
}

std::vector<int64_t> InferAvgPool2dShape(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &kernel_size,
                                         const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                         bool ceil_mode) {
  if (in_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "AvgPool2d input rank must >= 2";
  }

  const int64_t rank = static_cast<int64_t>(in_shape.size());
  std::vector<int64_t> out_shape = in_shape;
  int64_t h = in_shape[rank - 2];
  int64_t w = in_shape[rank - 1];

  auto compute_out = [=](int64_t in, int64_t k, int64_t s, int64_t p) -> int64_t {
    int64_t tmp = in + 2 * p - (ceil_mode ? k - 1 : k);
    int64_t o = tmp / s + 1;
    if (ceil_mode && tmp % s != 0) {
      ++o;
    }
    return o > 0 ? o : 0;
  };

  out_shape[rank - 2] = compute_out(h, kernel_size[0], stride[0], padding[0]);
  out_shape[rank - 1] = compute_out(w, kernel_size[1], stride[1], padding[1]);
  return out_shape;
}

ms::Tensor GenAvgPool2dResultTensor(const ms::Tensor &x, const std::vector<int64_t> &kernel_size,
                                    const std::vector<int64_t> &stride, const std::vector<int64_t> &padding,
                                    bool ceil_mode) {
  auto out_shape = InferAvgPool2dShape(x.shape(), kernel_size, stride, padding, ceil_mode);
  return ms::Tensor(x.data_type(), out_shape);
}

ms::Tensor npu_avgpool2d(const ms::Tensor &x, const std::vector<int64_t> &kernel_size,
                         const std::vector<int64_t> &stride, const std::vector<int64_t> &padding, bool ceil_mode,
                         bool count_include_pad, int64_t divisor_override, bool cube_math_type) {
  auto y = GenAvgPool2dResultTensor(x, kernel_size, stride, padding, ceil_mode);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnAvgPool2d");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAvgPool2d, x, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                          divisor_override, cube_math_type, y));
  runner->Run({x}, {y});
  return y;
}

bool has_ms_dispatch_source(const ms::Tensor &x) {
  return x.tensor()->has_ms_dispatch() && mindspore::pynative::HasSourcePyObj(x.tensor());
}

ms::Tensor first_tensor_from_list(const std::vector<ms::Tensor> &inputs) { return inputs[0]; }
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_abs_reduce_sum", PYBOOST_CALLER(1, custom::npu_abs_reduce_sum));
  m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min));
  m.def("npu_cast", PYBOOST_CALLER(1, custom::npu_cast));
  m.def("npu_batch_norm", PYBOOST_CALLER(3, custom::npu_batch_norm));
  m.def("npu_avgpool2d", PYBOOST_CALLER(1, custom::npu_avgpool2d));
  m.def("first_tensor_from_list", PYBOOST_CALLER(1, custom::first_tensor_from_list));
  m.def("has_ms_dispatch_source", custom::has_ms_dispatch_source);
}
