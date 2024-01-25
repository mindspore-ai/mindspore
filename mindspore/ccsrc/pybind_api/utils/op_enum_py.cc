/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "mindspore/core/ops/op_enum.h"
#include "mindspore/core/ops/op_def.h"
#include "mindspore/core/mindapi/base/format.h"
#include "include/common/pybind_api/api_register.h"
#include "mindapi/base/types.h"

namespace mindspore::ops {

namespace {
// The pybind11 registers the enum globally. Therefore, if different modules register the same enum type,
// conflicts occur.
// To avoid conflicts with the mindspore_lite, the FormatEnum added for use in the pybind.
enum FormatEnum : int64_t {
  DEFAULT_FORMAT = Format::DEFAULT_FORMAT,
  NCHW = Format::NCHW,
  NHWC = Format::NHWC,
  NHWC4 = Format::NHWC4,
  HWKC = Format::HWKC,
  HWCK = Format::HWCK,
  KCHW = Format::KCHW,
  CKHW = Format::CKHW,
  KHWC = Format::KHWC,
  CHWK = Format::CHWK,
  HW = Format::HW,
  HW4 = Format::HW4,
  NC = Format::NC,
  NC4 = Format::NC4,
  NC4HW4 = Format::NC4HW4,
  NCDHW = Format::NCDHW,
  NWC = Format::NWC,
  NCW = Format::NCW,
  NDHWC = Format::NDHWC,
  NC8HW8 = Format::NC8HW8
};
}  // namespace

void RegOpEnum(py::module *m) {
  auto m_sub = m->def_submodule("op_enum", "submodule for op enum");
  (void)m_sub.def("str_to_enum", &StringToEnumImpl, "string to enum value");
  (void)py::enum_<OP_DTYPE>(*m, "OpDtype", py::arithmetic())
    .value("DT_BEGIN", OP_DTYPE::DT_BEGIN)
    .value("DT_BOOL", OP_DTYPE::DT_BOOL)
    .value("DT_INT", OP_DTYPE::DT_INT)
    .value("DT_FLOAT", OP_DTYPE::DT_FLOAT)
    .value("DT_NUMBER", OP_DTYPE::DT_NUMBER)
    .value("DT_TENSOR", OP_DTYPE::DT_TENSOR)
    .value("DT_STR", OP_DTYPE::DT_STR)
    .value("DT_ANY", OP_DTYPE::DT_ANY)
    .value("DT_TUPLE_BOOL", OP_DTYPE::DT_TUPLE_BOOL)
    .value("DT_TUPLE_INT", OP_DTYPE::DT_TUPLE_INT)
    .value("DT_TUPLE_FLOAT", OP_DTYPE::DT_TUPLE_FLOAT)
    .value("DT_TUPLE_NUMBER", OP_DTYPE::DT_TUPLE_NUMBER)
    .value("DT_TUPLE_TENSOR", OP_DTYPE::DT_TUPLE_TENSOR)
    .value("DT_TUPLE_STR", OP_DTYPE::DT_TUPLE_STR)
    .value("DT_TUPLE_ANY", OP_DTYPE::DT_TUPLE_ANY)
    .value("DT_LIST_BOOL", OP_DTYPE::DT_LIST_BOOL)
    .value("DT_LIST_INT", OP_DTYPE::DT_LIST_INT)
    .value("DT_LIST_FLOAT", OP_DTYPE::DT_LIST_FLOAT)
    .value("DT_LIST_NUMBER", OP_DTYPE::DT_LIST_NUMBER)
    .value("DT_LIST_TENSOR", OP_DTYPE::DT_LIST_TENSOR)
    .value("DT_LIST_STR", OP_DTYPE::DT_LIST_STR)
    .value("DT_LIST_ANY", OP_DTYPE::DT_LIST_ANY)
    .value("DT_TYPE", OP_DTYPE::DT_TYPE)
    .value("DT_END", OP_DTYPE::DT_END);
  // There are currently some deficiencies in format, which will be filled in later.
  (void)py::enum_<FormatEnum>(*m, "FormatEnum", py::arithmetic())
    .value("DEFAULT_FORMAT", FormatEnum::DEFAULT_FORMAT)
    .value("NCHW", FormatEnum::NCHW)
    .value("NHWC", FormatEnum::NHWC)
    .value("NHWC4", FormatEnum::NHWC4)
    .value("HWKC", FormatEnum::HWKC)
    .value("HWCK", FormatEnum::HWCK)
    .value("KCHW", FormatEnum::KCHW)
    .value("CKHW", FormatEnum::CKHW)
    .value("KHWC", FormatEnum::KHWC)
    .value("CHWK", FormatEnum::CHWK)
    .value("HW", FormatEnum::HW)
    .value("HW4", FormatEnum::HW4)
    .value("NC", FormatEnum::NC)
    .value("NC4", FormatEnum::NC4)
    .value("NC4HW4", FormatEnum::NC4HW4)
    .value("NCDHW", FormatEnum::NCDHW)
    .value("NWC", FormatEnum::NWC)
    .value("NCW", FormatEnum::NCW)
    .value("NDHWC", FormatEnum::NDHWC)
    .value("NC8HW8", FormatEnum::NC8HW8);
  (void)py::enum_<Reduction>(*m, "ReductionEnum", py::arithmetic())
    .value("SUM", Reduction::REDUCTION_SUM)
    .value("MEAN", Reduction::MEAN)
    .value("NONE", Reduction::NONE);
}
}  // namespace mindspore::ops
