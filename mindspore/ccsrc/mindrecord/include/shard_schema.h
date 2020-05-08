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

#ifndef MINDRECORD_INCLUDE_SHARD_SCHEMA_H_
#define MINDRECORD_INCLUDE_SHARD_SCHEMA_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "mindrecord/include/common/shard_pybind.h"
#include "mindrecord/include/common/shard_utils.h"
#include "mindrecord/include/shard_error.h"
#include "pybind11/pybind11.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
class Schema {
 public:
  ~Schema() = default;

  /// \brief obtain the json schema ,its description, its block fields
  /// \param[in] desc the description of the schema
  /// \param[in] schema the schema's json
  static std::shared_ptr<Schema> Build(std::string desc, const json &schema);

  /// \brief obtain the json schema and its description for python
  /// \param[in] desc the description of the schema
  /// \param[in] schema the schema's json
  static std::shared_ptr<Schema> Build(std::string desc, pybind11::handle schema);

  /// \brief compare two schema to judge if they are equal
  /// \param b another schema to be judged
  /// \return true if they are equal,false if not
  bool operator==(const Schema &b) const;

  /// \brief get the schema and its description
  /// \return the json format of the schema and its description
  std::string GetDesc() const;

  /// \brief get the schema and its description
  /// \return the json format of the schema and its description
  json GetSchema() const;

  /// \brief get the schema and its description for python method
  /// \return the python object of the schema and its description
  pybind11::object GetSchemaForPython() const;

  /// set the schema id
  /// \param[in] id the id need to be set
  void SetSchemaID(int64_t id);

  /// get the schema id
  /// \return the int64 schema id
  int64_t GetSchemaID() const;

  /// get the blob fields
  /// \return the vector<string> blob fields
  std::vector<std::string> GetBlobFields() const;

 private:
  Schema() = default;
  static bool ValidateNumberShape(const json &it_value);
  static bool Validate(json schema);
  static std::vector<std::string> PopulateBlobFields(json schema);

  std::string desc_;
  json schema_;
  std::vector<std::string> blob_fields_;
  int64_t schema_id_ = -1;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_SCHEMA_H_
