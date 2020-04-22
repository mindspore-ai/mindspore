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

#include <string>
#include <vector>
#include "common/utils.h"
#include "mindrecord/include/common/shard_utils.h"
#include "mindrecord/include/shard_error.h"
#include "mindrecord/include/shard_index_generator.h"
#include "mindrecord/include/shard_reader.h"
#include "mindrecord/include/shard_segment.h"
#include "mindrecord/include/shard_writer.h"
#include "nlohmann/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utils/log_adapter.h"

namespace py = pybind11;

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
void BindSchema(py::module *m) {
  (void)py::class_<Schema, std::shared_ptr<Schema>>(*m, "Schema", py::module_local())
    .def_static("build", (std::shared_ptr<Schema>(*)(std::string, py::handle)) & Schema::Build)
    .def("get_desc", &Schema::get_desc)
    .def("get_schema_content", (py::object(Schema::*)()) & Schema::GetSchemaForPython)
    .def("get_blob_fields", &Schema::get_blob_fields)
    .def("get_schema_id", &Schema::get_schema_id);
}

void BindStatistics(const py::module *m) {
  (void)py::class_<Statistics, std::shared_ptr<Statistics>>(*m, "Statistics", py::module_local())
    .def_static("build", (std::shared_ptr<Statistics>(*)(std::string, py::handle)) & Statistics::Build)
    .def("get_desc", &Statistics::get_desc)
    .def("get_statistics", (py::object(Statistics::*)()) & Statistics::GetStatisticsForPython)
    .def("get_statistics_id", &Statistics::get_statistics_id);
}

void BindShardHeader(const py::module *m) {
  (void)py::class_<ShardHeader, std::shared_ptr<ShardHeader>>(*m, "ShardHeader", py::module_local())
    .def(py::init<>())
    .def("add_schema", &ShardHeader::AddSchema)
    .def("add_statistics", &ShardHeader::AddStatistic)
    .def("add_index_fields",
         (MSRStatus(ShardHeader::*)(const std::vector<std::string> &)) & ShardHeader::AddIndexFields)
    .def("get_meta", &ShardHeader::get_schemas)
    .def("get_statistics", &ShardHeader::get_statistics)
    .def("get_fields", &ShardHeader::get_fields)
    .def("get_schema_by_id", &ShardHeader::GetSchemaByID)
    .def("get_statistic_by_id", &ShardHeader::GetStatisticByID);
}

void BindShardWriter(py::module *m) {
  (void)py::class_<ShardWriter>(*m, "ShardWriter", py::module_local())
    .def(py::init<>())
    .def("open", &ShardWriter::Open)
    .def("open_for_append", &ShardWriter::OpenForAppend)
    .def("set_header_size", &ShardWriter::set_header_size)
    .def("set_page_size", &ShardWriter::set_page_size)
    .def("set_shard_header", &ShardWriter::SetShardHeader)
    .def("write_raw_data", (MSRStatus(ShardWriter::*)(std::map<uint64_t, std::vector<py::handle>> &,
                                                      vector<vector<uint8_t>> &, bool, bool)) &
                             ShardWriter::WriteRawData)
    .def("commit", &ShardWriter::Commit);
}

void BindShardReader(const py::module *m) {
  (void)py::class_<ShardReader, std::shared_ptr<ShardReader>>(*m, "ShardReader", py::module_local())
    .def(py::init<>())
    .def("open", (MSRStatus(ShardReader::*)(const std::string &, const int &, const std::vector<std::string> &,
                                            const std::vector<std::shared_ptr<ShardOperator>> &)) &
                   ShardReader::OpenPy)
    .def("launch", &ShardReader::Launch)
    .def("get_header", &ShardReader::get_shard_header)
    .def("get_blob_fields", &ShardReader::get_blob_fields)
    .def("get_next",
         (std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>(ShardReader::*)()) & ShardReader::GetNextPy)
    .def("finish", &ShardReader::Finish)
    .def("close", &ShardReader::Close);
}

void BindShardIndexGenerator(const py::module *m) {
  (void)py::class_<ShardIndexGenerator>(*m, "ShardIndexGenerator", py::module_local())
    .def(py::init<const std::string &, bool>())
    .def("build", &ShardIndexGenerator::Build)
    .def("write_to_db", &ShardIndexGenerator::WriteToDatabase);
}

void BindShardSegment(py::module *m) {
  (void)py::class_<ShardSegment>(*m, "ShardSegment", py::module_local())
    .def(py::init<>())
    .def("open", (MSRStatus(ShardSegment::*)(const std::string &, const int &, const std::vector<std::string> &,
                                             const std::vector<std::shared_ptr<ShardOperator>> &)) &
                   ShardSegment::OpenPy)
    .def("get_category_fields",
         (std::pair<MSRStatus, vector<std::string>>(ShardSegment::*)()) & ShardSegment::GetCategoryFields)
    .def("set_category_field", (MSRStatus(ShardSegment::*)(std::string)) & ShardSegment::SetCategoryField)
    .def("read_category_info", (std::pair<MSRStatus, std::string>(ShardSegment::*)()) & ShardSegment::ReadCategoryInfo)
    .def("read_at_page_by_id", (std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>>(
                                 ShardSegment::*)(int64_t, int64_t, int64_t)) &
                                 ShardSegment::ReadAtPageByIdPy)
    .def("read_at_page_by_name", (std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>>(
                                   ShardSegment::*)(std::string, int64_t, int64_t)) &
                                   ShardSegment::ReadAtPageByNamePy)
    .def("get_header", &ShardSegment::get_shard_header)
    .def("get_blob_fields",
         (std::pair<ShardType, std::vector<std::string>>(ShardSegment::*)()) & ShardSegment::get_blob_fields);
}

void BindGlobalParams(py::module *m) {
  (*m).attr("MIN_HEADER_SIZE") = kMinHeaderSize;
  (*m).attr("MAX_HEADER_SIZE") = kMaxHeaderSize;
  (*m).attr("MIN_PAGE_SIZE") = kMinPageSize;
  (*m).attr("MAX_PAGE_SIZE") = kMaxPageSize;
  (*m).attr("MIN_SHARD_COUNT") = kMinShardCount;
  (*m).attr("MAX_SHARD_COUNT") = kMaxShardCount;
  (*m).attr("MIN_CONSUMER_COUNT") = kMinConsumerCount;
  (void)(*m).def("get_max_thread_num", &GetMaxThreadNum);
}

PYBIND11_MODULE(_c_mindrecord, m) {
  m.doc() = "pybind11 mindrecord plugin";  // optional module docstring
  (void)py::enum_<MSRStatus>(m, "MSRStatus", py::module_local())
    .value("SUCCESS", SUCCESS)
    .value("FAILED", FAILED)
    .export_values();
  (void)py::enum_<ShardType>(m, "ShardType", py::module_local()).value("NLP", kNLP).value("CV", kCV).export_values();
  BindGlobalParams(&m);
  BindSchema(&m);
  BindStatistics(&m);
  BindShardHeader(&m);
  BindShardWriter(&m);
  BindShardReader(&m);
  BindShardIndexGenerator(&m);
  BindShardSegment(&m);
}
}  // namespace mindrecord
}  // namespace mindspore

namespace nlohmann {
namespace detail {
py::object FromJsonImpl(const json &j) {
  if (j.is_null()) {
    return py::none();
  } else if (j.is_boolean()) {
    return py::bool_(j.get<bool>());
  } else if (j.is_number()) {
    double number = j.get<double>();
    if (fabs(number - std::floor(number)) < mindspore::mindrecord::kEpsilon) {
      return py::int_(j.get<int64_t>());
    } else {
      return py::float_(number);
    }
  } else if (j.is_string()) {
    return py::str(j.get<std::string>());
  } else if (j.is_array()) {
    py::list obj;
    for (const auto &el : j) {
      (void)obj.attr("append")(FromJsonImpl(el));
    }
    return std::move(obj);
  } else {
    py::dict obj;
    for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it) {
      obj[py::str(it.key())] = FromJsonImpl(it.value());
    }
    return std::move(obj);
  }
}

json ToJsonImpl(const py::handle &obj) {
  if (obj.is_none()) {
    return nullptr;
  }
  if (py::isinstance<py::bool_>(obj)) {
    return obj.cast<bool>();
  }
  if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int64_t>();
  }
  if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  }
  if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto out = json::array();
    for (const py::handle &value : obj) {
      out.push_back(ToJsonImpl(value));
    }
    return out;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto out = json::object();
    for (const py::handle &key : obj) {
      out[py::str(key).cast<std::string>()] = ToJsonImpl(obj[key]);
    }
    return out;
  }
  MS_LOG(ERROR) << "Python to json failed, obj is: " << py::cast<std::string>(obj);
  return json();
}
}  // namespace detail

py::object adl_serializer<py::object>::FromJson(const json &j) { return detail::FromJsonImpl(j); }

void adl_serializer<py::object>::ToJson(json *j, const py::object &obj) {
  *j = detail::ToJsonImpl(obj);
}  // namespace detail
}  // namespace nlohmann
