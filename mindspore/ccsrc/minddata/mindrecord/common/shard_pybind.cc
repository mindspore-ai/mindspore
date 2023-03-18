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

#include "utils/ms_utils.h"
#include "minddata/dataset/util/md_log_adapter.h"
#include "minddata/mindrecord/include/common/log_adapter.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_segment.h"
#include "minddata/mindrecord/include/shard_writer.h"
#include "nlohmann/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using mindspore::dataset::MDLogAdapter;

namespace mindspore {
namespace mindrecord {
#define THROW_IF_ERROR(s)                                                            \
  do {                                                                               \
    Status rc = std::move(s);                                                        \
    if (rc.IsError()) throw std::runtime_error(MDLogAdapter::Apply(&rc).ToString()); \
  } while (false)

void BindSchema(py::module *m) {
  (void)py::class_<Schema, std::shared_ptr<Schema>>(*m, "Schema", py::module_local())
    .def_static("build",
                [](std::string desc, const pybind11::handle &schema) {
                  json schema_json = nlohmann::detail::ToJsonImpl(schema);
                  return Schema::Build(std::move(desc), schema_json);
                })
    .def("get_desc", &Schema::GetDesc)
    .def("get_schema_content",
         [](Schema &s) {
           json schema_json = s.GetSchema();
           return nlohmann::detail::FromJsonImpl(schema_json);
         })
    .def("get_blob_fields", &Schema::GetBlobFields)
    .def("get_schema_id", &Schema::GetSchemaID);
}

void BindStatistics(const py::module *m) {
  (void)py::class_<Statistics, std::shared_ptr<Statistics>>(*m, "Statistics", py::module_local())
    .def_static("build",
                [](std::string desc, const pybind11::handle &statistics) {
                  json statistics_json = nlohmann::detail::ToJsonImpl(statistics);
                  return Statistics::Build(std::move(desc), statistics_json);
                })
    .def("get_desc", &Statistics::GetDesc)
    .def("get_statistics",
         [](Statistics &s) {
           json statistics_json = s.GetStatistics();
           return nlohmann::detail::FromJsonImpl(statistics_json);
         })
    .def("get_statistics_id", &Statistics::GetStatisticsID);
}

void BindShardHeader(const py::module *m) {
  (void)py::class_<ShardHeader, std::shared_ptr<ShardHeader>>(*m, "ShardHeader", py::module_local())
    .def(py::init<>())
    .def("add_schema", &ShardHeader::AddSchema)
    .def("add_statistics", &ShardHeader::AddStatistic)
    .def("add_index_fields",
         [](ShardHeader &s, const std::vector<std::string> &fields) {
           THROW_IF_ERROR(s.AddIndexFields(fields));
           return SUCCESS;
         })
    .def("get_meta", &ShardHeader::GetSchemas)
    .def("get_statistics", &ShardHeader::GetStatistics)
    .def("get_fields", &ShardHeader::GetFields)
    .def("get_schema_by_id",
         [](ShardHeader &s, int64_t schema_id) {
           std::shared_ptr<Schema> schema_ptr;
           THROW_IF_ERROR(s.GetSchemaByID(schema_id, &schema_ptr));
           return schema_ptr;
         })
    .def("get_statistic_by_id", [](ShardHeader &s, int64_t statistic_id) {
      std::shared_ptr<Statistics> statistics_ptr;
      THROW_IF_ERROR(s.GetStatisticByID(statistic_id, &statistics_ptr));
      return statistics_ptr;
    });
}

void BindShardWriter(py::module *m) {
  (void)py::class_<ShardWriter>(*m, "ShardWriter", py::module_local())
    .def(py::init<>())
    .def("open",
         [](ShardWriter &s, const std::vector<std::string> &paths, bool append, bool overwrite) {
           THROW_IF_ERROR(s.Open(paths, append, overwrite));
           return SUCCESS;
         })
    .def("open_for_append",
         [](ShardWriter &s, const std::string &path) {
           THROW_IF_ERROR(s.OpenForAppend(path));
           return SUCCESS;
         })
    .def("set_header_size",
         [](ShardWriter &s, const uint64_t &header_size) {
           THROW_IF_ERROR(s.SetHeaderSize(header_size));
           return SUCCESS;
         })
    .def("set_page_size",
         [](ShardWriter &s, const uint64_t &page_size) {
           THROW_IF_ERROR(s.SetPageSize(page_size));
           return SUCCESS;
         })
    .def("set_shard_header",
         [](ShardWriter &s, std::shared_ptr<ShardHeader> header_data) {
           THROW_IF_ERROR(s.SetShardHeader(header_data));
           return SUCCESS;
         })
    .def("write_raw_data",
         [](ShardWriter &s, std::map<uint64_t, std::vector<py::handle>> &raw_data, vector<py::bytes> &blob_data,
            bool sign, bool parallel_writer) {
           // convert the raw_data from dict to json
           std::map<uint64_t, std::vector<json>> raw_data_json;
           (void)std::transform(raw_data.begin(), raw_data.end(), std::inserter(raw_data_json, raw_data_json.end()),
                                [](const std::pair<uint64_t, std::vector<py::handle>> &p) {
                                  auto &py_raw_data = p.second;
                                  std::vector<json> json_raw_data;
                                  (void)std::transform(
                                    py_raw_data.begin(), py_raw_data.end(), std::back_inserter(json_raw_data),
                                    [](const py::handle &obj) { return nlohmann::detail::ToJsonImpl(obj); });
                                  return std::make_pair(p.first, std::move(json_raw_data));
                                });

           // parallel convert blob_data from vector<py::bytes> to vector<vector<uint8_t>>
           int32_t parallel_convert = kParallelConvert;
           if (parallel_convert > blob_data.size()) {
             parallel_convert = blob_data.size();
           }
           parallel_convert = parallel_convert != 0 ? parallel_convert : 1;
           std::vector<std::thread> thread_set(parallel_convert);
           vector<vector<uint8_t>> vector_blob_data(blob_data.size());
           uint32_t step = uint32_t(blob_data.size() / parallel_convert);
           if (blob_data.size() % parallel_convert != 0) {
             step = step + 1;
           }
           for (int x = 0; x < parallel_convert; ++x) {
             uint32_t start = x * step;
             uint32_t end = ((x + 1) * step) < blob_data.size() ? ((x + 1) * step) : blob_data.size();
             thread_set[x] = std::thread([&vector_blob_data, &blob_data, start, end]() {
               for (auto i = start; i < end; i++) {
                 char *buffer = nullptr;
                 ssize_t length = 0;
                 if (PYBIND11_BYTES_AS_STRING_AND_SIZE(blob_data[i].ptr(), &buffer, &length) != 0) {
                   MS_LOG(ERROR) << "Unable to extract bytes contents!";
                   return FAILED;
                 }
                 vector<uint8_t> blob_data_item(length);
                 if (length < SECUREC_MEM_MAX_LEN) {
                   int ret_code = memcpy_s(&blob_data_item[0], length, buffer, length);
                   if (ret_code != EOK) {
                     MS_LOG(ERROR) << "memcpy_s failed for py::bytes to vector<uint8_t>.";
                     return FAILED;
                   }
                 } else {
                   auto ret_code = std::memcpy(&blob_data_item[0], buffer, length);
                   if (ret_code != &blob_data_item[0]) {
                     MS_LOG(ERROR) << "memcpy failed for py::bytes to vector<uint8_t>.";
                     return FAILED;
                   }
                 }
                 vector_blob_data[i] = blob_data_item;
               }
             });
           }

           // wait for the threads join
           for (int x = 0; x < parallel_convert; ++x) {
             thread_set[x].join();
           }
           THROW_IF_ERROR(s.WriteRawData(raw_data_json, vector_blob_data, sign, parallel_writer));
           return SUCCESS;
         })
    .def("commit", [](ShardWriter &s) {
      THROW_IF_ERROR(s.Commit());
      return SUCCESS;
    });
}

void BindShardReader(const py::module *m) {
  (void)py::class_<ShardReader, std::shared_ptr<ShardReader>>(*m, "ShardReader", py::module_local())
    .def(py::init<>())
    .def("open",
         [](ShardReader &s, const std::vector<std::string> &file_paths, bool load_dataset, const int &n_consumer,
            const std::vector<std::string> &selected_columns,
            const std::vector<std::shared_ptr<ShardOperator>> &operators) {
           THROW_IF_ERROR(s.Open(file_paths, load_dataset, n_consumer, selected_columns, operators));
           return SUCCESS;
         })
    .def("launch",
         [](ShardReader &s) {
           THROW_IF_ERROR(s.Launch(false));
           return SUCCESS;
         })
    .def("get_header", &ShardReader::GetShardHeader)
    .def("get_blob_fields", &ShardReader::GetBlobFields)
    .def("get_next",
         [](ShardReader &s) {
           auto data = s.GetNext();
           vector<std::tuple<std::vector<std::vector<uint8_t>>, pybind11::object>> res;
           std::transform(data.begin(), data.end(), std::back_inserter(res),
                          [&s](const std::tuple<std::vector<uint8_t>, json> &item) {
                            auto &j = std::get<1>(item);
                            pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                            auto blob_data_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
                            (void)s.UnCompressBlob(std::get<0>(item), &blob_data_ptr);
                            return std::make_tuple(*blob_data_ptr, std::move(obj));
                          });
           return res;
         })
    .def("close", &ShardReader::Close)
    .def("len", &ShardReader::GetNumRows);
}

void BindShardIndexGenerator(const py::module *m) {
  (void)py::class_<ShardIndexGenerator>(*m, "ShardIndexGenerator", py::module_local())
    .def(py::init<const std::string &, bool>())
    .def("build",
         [](ShardIndexGenerator &s) {
           THROW_IF_ERROR(s.Build());
           return SUCCESS;
         })
    .def("write_to_db", [](ShardIndexGenerator &s) {
      THROW_IF_ERROR(s.WriteToDatabase());
      return SUCCESS;
    });
}

void BindShardSegment(py::module *m) {
  (void)py::class_<ShardSegment>(*m, "ShardSegment", py::module_local())
    .def(py::init<>())
    .def("open",
         [](ShardSegment &s, const std::vector<std::string> &file_paths, bool load_dataset, const int &n_consumer,
            const std::vector<std::string> &selected_columns,
            const std::vector<std::shared_ptr<ShardOperator>> &operators) {
           THROW_IF_ERROR(s.Open(file_paths, load_dataset, n_consumer, selected_columns, operators));
           return SUCCESS;
         })
    .def("get_category_fields",
         [](ShardSegment &s) {
           auto fields_ptr = std::make_shared<vector<std::string>>();
           THROW_IF_ERROR(s.GetCategoryFields(&fields_ptr));
           return *fields_ptr;
         })
    .def("set_category_field",
         [](ShardSegment &s, const std::string &category_field) {
           THROW_IF_ERROR(s.SetCategoryField(category_field));
           return SUCCESS;
         })
    .def("read_category_info",
         [](ShardSegment &s) {
           std::shared_ptr<std::string> category_ptr;
           THROW_IF_ERROR(s.ReadCategoryInfo(&category_ptr));
           return *category_ptr;
         })
    .def("read_at_page_by_id",
         [](ShardSegment &s, int64_t category_id, int64_t page_no, int64_t n_rows_of_page) {
           auto pages_load_ptr = std::make_shared<PAGES_LOAD>();
           auto pages_ptr = std::make_shared<PAGES>();
           THROW_IF_ERROR(s.ReadAllAtPageById(category_id, page_no, n_rows_of_page, &pages_ptr));
           (void)std::transform(pages_ptr->begin(), pages_ptr->end(), std::back_inserter(*pages_load_ptr),
                                [](const std::tuple<std::vector<uint8_t>, json> &item) {
                                  auto &j = std::get<1>(item);
                                  pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                                  return std::make_tuple(std::get<0>(item), std::move(obj));
                                });
           return *pages_load_ptr;
         })
    .def("read_at_page_by_name",
         [](ShardSegment &s, std::string category_name, int64_t page_no, int64_t n_rows_of_page) {
           auto pages_load_ptr = std::make_shared<PAGES_LOAD>();
           auto pages_ptr = std::make_shared<PAGES>();
           THROW_IF_ERROR(s.ReadAllAtPageByName(category_name, page_no, n_rows_of_page, &pages_ptr));
           (void)std::transform(pages_ptr->begin(), pages_ptr->end(), std::back_inserter(*pages_load_ptr),
                                [](const std::tuple<std::vector<uint8_t>, json> &item) {
                                  auto &j = std::get<1>(item);
                                  pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                                  return std::make_tuple(std::get<0>(item), std::move(obj));
                                });
           return *pages_load_ptr;
         })
    .def("get_header", &ShardSegment::GetShardHeader)
    .def("get_blob_fields", [](ShardSegment &s) { return s.GetBlobFields(); });
}

void BindGlobalParams(py::module *m) {
  (*m).attr("MIN_HEADER_SIZE") = kMinHeaderSize;
  (*m).attr("MAX_HEADER_SIZE") = kMaxHeaderSize;
  (*m).attr("MIN_PAGE_SIZE") = kMinPageSize;
  (*m).attr("MAX_PAGE_SIZE") = kMaxPageSize;
  (*m).attr("MIN_SHARD_COUNT") = kMinShardCount;
  (*m).attr("MAX_SHARD_COUNT") = kMaxShardCount;
  (*m).attr("MAX_FILE_COUNT") = kMaxFileCount;
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
  MS_LOG(ERROR) << "[Internal ERROR] Failed to convert Python object: " << py::cast<std::string>(obj)
                << " to type json.";
  return json();
}
}  // namespace detail

py::object adl_serializer<py::object>::FromJson(const json &j) { return detail::FromJsonImpl(j); }

void adl_serializer<py::object>::ToJson(json *j, const py::object &obj) {
  *j = detail::ToJsonImpl(obj);
}  // namespace detail
}  // namespace nlohmann
