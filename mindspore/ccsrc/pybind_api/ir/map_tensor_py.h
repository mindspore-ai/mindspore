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

#ifndef MINDSPORE_CCSRC_UTILS_MAP_TENSOR_PY_H_
#define MINDSPORE_CCSRC_UTILS_MAP_TENSOR_PY_H_

#include <tuple>
#include <memory>
#include "pybind11/numpy.h"
#include "ir/map_tensor.h"

namespace py = pybind11;

namespace mindspore {
using MapTensor = tensor::MapTensor;
using MapTensorPtr = std::shared_ptr<MapTensor>;
//
// MapTensor python adapter class.
//
class MapTensorPy {
 public:
  static MapTensorPtr MakeMapTensor(const TypePtr &key_dtype, const TypePtr &value_dtype,
                                    const ShapeVector &value_shape, const py::object &default_value_obj);

  static void UpdateFromNumpy(const MapTensorPtr &map_tensor,
                              const std::tuple<py::array, py::array, py::array> &numpy_data);

  static std::tuple<py::array, py::array, py::array> ExportAsNumpy(const MapTensorPtr &map_tensor, bool full = false);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_MAP_TENSOR_PY_H_
