/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

using ShapeValueDType = int64_t;
using ShapeVector = std::vector<ShapeValueDType>;

namespace py = pybind11;
namespace mindspore::tensor {

class TensorTupleIndexInfoForView final {
 public:
  TensorTupleIndexInfoForView(void) = default;
  TensorTupleIndexInfoForView(const size_t &dim, const ShapeVector &data_shape, const size_t &specified_dimensions,
                              std::vector<int64_t> *data_transfer_types, std::vector<py::object> *data_transfer_args,
                              std::vector<pynative::SliceOpInfoPtr> *slice_op_infos)
      : m_dim(dim),
        m_data_shape(data_shape),
        m_specified_dimensions(specified_dimensions),
        m_new_data_shape(data_shape),
        m_data_transfer_types(data_transfer_types),
        m_data_transfer_args(data_transfer_args),
        m_slice_op_infos(slice_op_infos) {}

  inline bool CouldApplyView() { return m_could_apply_view; }
  inline void CheckEllipsisCounter() {
    if (m_ellipsis_counter > 0) {
      MS_EXCEPTION(IndexError) << "An index can only have a single ellipsis('...')";
    }
  }

  size_t m_dim{};
  ShapeVector m_data_shape{};
  size_t m_specified_dimensions{};
  ShapeVector m_new_data_shape{};
  std::vector<int64_t> *m_data_transfer_types{};
  std::vector<py::object> *m_data_transfer_args{};
  std::vector<pynative::SliceOpInfoPtr> *m_slice_op_infos{};
  bool m_could_apply_view{true};
  bool m_first_occurrence{true};
  size_t m_ellipsis_counter{0};
  bool m_empty_strided_slice_result{false};
};
}  // namespace mindspore::tensor
