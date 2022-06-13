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

#include "minddata/mindrecord/include/shard_schema.h"
#include "utils/ms_utils.h"

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

std::string Schema::GetDesc() const { return desc_; }

json Schema::GetSchema() const {
  json str_schema;
  str_schema["desc"] = desc_;
  str_schema["schema"] = schema_;
  str_schema["blob_fields"] = blob_fields_;
  return str_schema;
}

void Schema::SetSchemaID(int64_t id) { schema_id_ = id; }

int64_t Schema::GetSchemaID() const { return schema_id_; }

std::vector<std::string> Schema::GetBlobFields() const { return blob_fields_; }

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
    MS_LOG(ERROR) << "Invalid schema, 'shape' object can not found in " << it_value.dump()
                  << ". Please check the input schema.";
    return false;
  }

  auto shape = it_value["shape"];
  if (!shape.is_array()) {
    MS_LOG(ERROR) << "Invalid schema, the value of 'shape' should be list format but got: " << it_value["shape"]
                  << ". Please check the input schema.";
    return false;
  }

  int num_negtive_one = 0;
  for (const auto &i : shape) {
    if (i == 0 || i < -1) {
      MS_LOG(ERROR) << "Invalid schema, the element of 'shape' value should be -1 or greater than 0 but got: " << i
                    << ". Please check the input schema.";
      return false;
    }
    if (i == -1) {
      num_negtive_one++;
    }
  }

  if (num_negtive_one > 1) {
    MS_LOG(ERROR) << "Invalid schema, only 1 variable dimension(-1) allowed in 'shape' value but got: "
                  << it_value["shape"] << ". Please check the input schema.";
    return false;
  }

  return true;
}

bool Schema::Validate(json schema) {
  if (schema.empty()) {
    MS_LOG(ERROR) << "Invalid schema, schema is empty. Please check the input schema.";
    return false;
  }

  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    // make sure schema key name must be composed of '0-9' or 'a-z' or 'A-Z' or '_'
    if (!ValidateFieldName(it.key())) {
      MS_LOG(ERROR) << "Invalid schema, field name: " << it.key()
                    << "is not composed of '0-9' or 'a-z' or 'A-Z' or '_'. Please rename the field name in schema.";
      return false;
    }

    json it_value = it.value();
    if (it_value.find("type") == it_value.end()) {
      MS_LOG(ERROR) << "Invalid schema, 'type' object can not found in field " << it_value.dump()
                    << ". Please add the 'type' object for field in schema.";
      return false;
    }

    if (kFieldTypeSet.find(it_value["type"]) == kFieldTypeSet.end()) {
      MS_LOG(ERROR) << "Invalid schema, the value of 'type': " << it_value["type"]
                    << " is not supported.\nPlease modify the value of 'type' to 'int32', 'int64', 'float32', "
                       "'float64', 'string', 'bytes' in schema.";
      return false;
    }

    if (it_value.size() == kInt1) {
      continue;
    }

    if (it_value["type"] == "bytes" || it_value["type"] == "string") {
      MS_LOG(ERROR)
        << "Invalid schema, no other field can be added when the value of 'type' is 'string' or 'types' but got: "
        << it_value.dump() << ". Please remove other fields in schema.";
      return false;
    }

    if (it_value.size() != kInt2) {
      MS_LOG(ERROR) << "Invalid schema, the fields should be 'type' or 'type' and 'shape' but got: " << it_value.dump()
                    << ". Please check the schema.";
      return false;
    }

    if (!ValidateNumberShape(it_value)) {
      return false;
    }
  }

  return true;
}

bool Schema::operator==(const mindrecord::Schema &b) const {
  if (this->GetDesc() != b.GetDesc() || this->GetSchema() != b.GetSchema()) {
    return false;
  }
  return true;
}
}  // namespace mindrecord
}  // namespace mindspore
