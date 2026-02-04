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

#include <set>
#include <optional>
#include "ms_extension/all.h"

namespace custom {
ms::Tensor GenResultTensor(const ms::Tensor &t, const std::vector<int64_t> &axis, bool keepdims, ms::TypeId type_id) {
  auto s = t.shape();
  size_t n = s.size();
  ShapeVector out_shape;
  if (!axis.empty()) {
    std::set<size_t> axis_set;
    for (auto x : axis) {
      axis_set.insert(static_cast<size_t>(x > 0 ? x : x + static_cast<int64_t>(n)));
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
  auto result = GenResultTensor(x, axis_vec, keepdims, type_id);
  auto y = ms::Tensor(mindspore::kernel::pyboost::abs(x.tensor()));
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ReduceSum");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnReduceSum, y, axis_vec, keepdims, type_id, result));
  runner->Run({y}, {result});
  return result;
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_abs_reduce_sum", PYBOOST_CALLER(1, custom::npu_abs_reduce_sum)); }
