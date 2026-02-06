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

ms::Tensor GenResultTensor(const ms::Tensor &t, int64_t dim, bool keep_dim, ms::TypeId type_id) {
  ShapeVector in_shape = t.shape();
  ShapeVector out_shape = InferArgMinShape(in_shape, dim, keep_dim);
  return ms::Tensor(type_id, out_shape);
}

ms::Tensor npu_arg_min(const ms::Tensor &x, int64_t dim, bool keep_dim) {
  auto result = GenResultTensor(x, dim, keep_dim, ms::TypeId::kNumberTypeInt64);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ReduceSum");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, dim, keep_dim, result));
  runner->Run({x}, {result});
  return result;
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min)); }
