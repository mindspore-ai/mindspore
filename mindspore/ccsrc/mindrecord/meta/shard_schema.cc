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

#include "mindrecord/include/shard_schema.h"
#include "common/utils.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
std::shared_ptr<Schema> Schema::Build(std::string desc, const json &schema) {
  // validate check
  if (!Validate(schema)) {
    return nullptr;
  }

  std::vector<std::string> blob_fields = PopulateBlobFields(schema);
  Schema object_schema;
  object_schema.desc_ = std::move(desc);
  object_schema.blob_fields_ = std::move(blob_fields);
  object_schema.schema_ = schema;
  object_schema.schema_id_ = -1;
  return std::make_shared<Schema>(object_schema);
}

std::shared_ptr<Schema> Schema::Build(std::string desc, pybind11::handle schema) {
  // validate check
  json schema_json = nlohmann::detail::ToJsonImpl(schema);
  return Build(std::move(desc), schema_json);
}

std::string Schema::get_desc() const { return desc_; }

json Schema::GetSchema() const {
  json str_schema;
  str_schema["desc"] = desc_;
  str_schema["schema"] = schema_;
  str_schema["blob_fields"] = blob_fields_;
  return str_schema;
}

pybind11::object Schema::GetSchemaForPython() const {
  json schema_json = GetSchema();
  pybind11::object schema_py = nlohmann::detail::FromJsonImpl(schema_json);
  return schema_py;
}

void Schema::set_schema_id(int64_t id) { schema_id_ = id; }

int64_t Schema::get_schema_id() const { return schema_id_; }

std::vector<std::string> Schema::get_blob_fields() const { return blob_fields_; }

std::vector<std::string> Schema::PopulateBlobFields(json schema) {
  std::vector<std::string> blob_fields;
  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    json it_value = it.value();
    if ((it_value.size() == kInt2 && it_value.find("shape") != it_value.end()) || it_value["type"] == "bytes") {
      blob_fields.emplace_back(it.key());
    }
  }
  return blob_fields;
}

bool Schema::ValidateNumberShape(const json &it_value) {
  if (it_value.find("shape") == it_value.end()) {
    MS_LOG(ERROR) << "%s supports shape only." << it_value["type"].dump();
    return false;
  }

  auto shape = it_value["shape"];
  if (!shape.is_array()) {
    MS_LOG(ERROR) << "%s shape format is wrong." << it_value["type"].dump();
    return false;
  }

  int num_negtive_one = 0;
  for (const auto &i : shape) {
    if (i == 0 || i < -1) {
      MS_LOG(ERROR) << "Shape %s, number is wrong." << it_value["shape"].dump();
      return false;
    }
    if (i == -1) {
      num_negtive_one++;
    }
  }

  if (num_negtive_one > 1) {
    MS_LOG(ERROR) << "Shape %s, have at most 1 variable-length dimension." << it_value["shape"].dump();
    return false;
  }

  return true;
}

bool Schema::Validate(json schema) {
  if (schema.size() == kInt0) {
    MS_LOG(ERROR) << "Schema is null";
    return false;
  }

  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    // make sure schema key name must be composed of '0-9' or 'a-z' or 'A-Z' or '_'
    if (!ValidateFieldName(it.key())) {
      MS_LOG(ERROR) << "Field name must be composed of '0-9' or 'a-z' or 'A-Z' or '_', fieldName: " << it.key();
      return false;
    }

    json it_value = it.value();
    if (it_value.find("type") == it_value.end()) {
      MS_LOG(ERROR) << "No 'type' field exist: " << it_value.dump();
      return false;
    }

    if (kFieldTypeSet.find(it_value["type"]) == kFieldTypeSet.end()) {
      MS_LOG(ERROR) << "Wrong type: " << it_value["type"].dump();
      return false;
    }

    if (it_value.size() == kInt1) {
      continue;
    }

    if (it_value["type"] == "bytes" || it_value["type"] == "string") {
      MS_LOG(ERROR) << it_value["type"].dump() << " can not 1 field only.";
      return false;
    }

    if (it_value.size() != kInt2) {
      MS_LOG(ERROR) << it_value["type"].dump() << " can have at most 2 fields.";
      return false;
    }

    if (!ValidateNumberShape(it_value)) {
      return false;
    }
  }

  return true;
}

bool Schema::operator==(const mindrecord::Schema &b) const {
  if (this->get_desc() != b.get_desc() || this->GetSchema() != b.GetSchema()) {
    return false;
  }
  return true;
}
}  // namespace mindrecord
}  // namespace mindspore
