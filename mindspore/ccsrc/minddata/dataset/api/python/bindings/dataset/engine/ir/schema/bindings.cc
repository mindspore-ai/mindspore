/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  SchemaObj, 0, ([](const py::module *m) {
    (void)py::class_<SchemaObj, std::shared_ptr<SchemaObj>>(*m, "SchemaObj", "to create a SchemaObj")
      .def(py::init([](std::string schema_file) {
        auto schema = std::make_shared<SchemaObj>(schema_file);
        THROW_IF_ERROR(schema->Init());
        return schema;
      }))
      .def("add_column",
           [](SchemaObj &self, std::string name, TypeId de_type, std::vector<int32_t> shape) {
             THROW_IF_ERROR(self.add_column(name, static_cast<mindspore::DataType>(de_type), shape));
           })
      .def("add_column", [](SchemaObj &self, std::string name, std::string de_type,
                            std::vector<int32_t> shape) { THROW_IF_ERROR(self.add_column(name, de_type, shape)); })
      .def("add_column",
           [](SchemaObj &self, std::string name, TypeId de_type) {
             THROW_IF_ERROR(self.add_column(name, static_cast<mindspore::DataType>(de_type)));
           })
      .def("add_column", [](SchemaObj &self, std::string name,
                            std::string de_type) { THROW_IF_ERROR(self.add_column(name, de_type)); })
      .def("parse_columns",
           [](SchemaObj &self, std::string json_string) { THROW_IF_ERROR(self.ParseColumnString(json_string)); })
      .def("to_json", &SchemaObj::to_json)
      .def("to_string", &SchemaObj::to_string)
      .def("from_string",
           [](SchemaObj &self, std::string json_string) { THROW_IF_ERROR(self.FromJSONString(json_string)); })
      .def("set_dataset_type", [](SchemaObj &self, std::string dataset_type) { self.set_dataset_type(dataset_type); })
      .def("set_num_rows", [](SchemaObj &self, int32_t num_rows) { self.set_num_rows(num_rows); })
      .def("get_num_rows", &SchemaObj::get_num_rows)
      .def("__deepcopy__", [](py::object &schema, py::dict memo) { return schema; });
  }));

}  // namespace dataset
}  // namespace mindspore
