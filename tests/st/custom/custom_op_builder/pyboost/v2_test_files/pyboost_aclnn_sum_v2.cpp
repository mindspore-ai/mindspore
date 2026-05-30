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
#include <vector>
#include "ms_extension/all.h"

namespace custom {

static ShapeVector InferReduceShape(const ShapeVector &in_shape, const std::vector<int64_t> &axis, bool keep_dims) {
  size_t rank = in_shape.size();
  ShapeVector out_shape;
  if (!axis.empty()) {
    std::set<size_t> axis_set;
    for (auto axis_item : axis) {
      auto normalized_axis = axis_item >= 0 ? axis_item : axis_item + static_cast<int64_t>(rank);
      if (normalized_axis < 0 || normalized_axis >= static_cast<int64_t>(rank)) {
        MS_LOG(EXCEPTION) << "Reduce axis " << axis_item << " is out of range for rank " << rank;
      }
      axis_set.insert(static_cast<size_t>(normalized_axis));
    }
    for (size_t i = 0; i < rank; ++i) {
      if (axis_set.count(i) != 0) {
        if (keep_dims) {
          out_shape.push_back(1);
        }
      } else {
        out_shape.push_back(in_shape[i]);
      }
    }
  } else if (keep_dims) {
    out_shape.resize(rank, 1);
  }
  return out_shape;
}

ms::Tensor npu_abs_reduce_sum_v2(const ms::Tensor &x, const std::optional<std::vector<int64_t>> &axis, bool keep_dims,
                                 const std::optional<int64_t> &dtype) {
  auto type_id = dtype.has_value() ? static_cast<ms::TypeId>(dtype.value()) : x.data_type();
  auto axis_value = axis.has_value() ? axis.value() : std::vector<int64_t>();
  auto abs_out = ms::Tensor(x.data_type(), x.shape());
  auto result = ms::Tensor(type_id, InferReduceShape(x.shape(), axis_value, keep_dims));

  ms::TensorToDevice(x);
  ms::TensorAllocate(abs_out, result);
  MS_DISPATCH_ACLNN(aclnnAbs, x, abs_out);
  MS_DISPATCH_ACLNN(aclnnReduceSum, abs_out, axis_value, keep_dims, type_id, result);
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_abs_reduce_sum_v2", PYBOOST_CALLER(1, custom::npu_abs_reduce_sum_v2));
}
