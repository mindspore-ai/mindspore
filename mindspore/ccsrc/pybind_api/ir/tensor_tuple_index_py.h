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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pybind11/numpy.h"

#include "ir/tensor.h"
#include "pybind_api/ir/tensor_index_py.h"
#include "pybind_api/ir/tensor_tuple_index_view_info_py.h"
#include "mindspore/core/ops/array_ops.h"

namespace py = pybind11;
namespace mindspore::tensor {

class IGetItemByTupleWithView {
 public:
  virtual ~IGetItemByTupleWithView() = default;
  virtual void GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) = 0;
  static inline int64_t CheckRange(int64_t x, int64_t dim_size) {
    MS_EXCEPTION_IF_ZERO("dim_size", dim_size);
    return (dim_size + (x % dim_size)) % dim_size;
  }
};

class GetItemByIntWithView final : public IGetItemByTupleWithView {
 public:
  void GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) override;
};

class GetItemBySliceWithView final : public IGetItemByTupleWithView {
 public:
  void GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) override;
};

class GetItemByEllipsisWithView final : public IGetItemByTupleWithView {
 public:
  void GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) override;
};

class GetItemByNoneWithView final : public IGetItemByTupleWithView {
 public:
  void GetItemWithView(TensorTupleIndexInfoForView *ViewInfo, py::handle obj) override;
};
}  // namespace mindspore::tensor
