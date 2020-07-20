/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_PYBIND_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_PYBIND_H_

#include <string>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace nlohmann {
template <>
struct adl_serializer<py::object> {
  py::object FromJson(const json &j);

  void ToJson(json *j, const py::object &obj);
};

namespace detail {
py::object FromJsonImpl(const json &j);

json ToJsonImpl(const py::handle &obj);
}  // namespace detail
}  // namespace nlohmann
#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_COMMON_SHARD_PYBIND_H_
