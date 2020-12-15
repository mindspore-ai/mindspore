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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
#pragma once

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_schema.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
using std::cin;
using std::endl;
using std::pair;
using std::string;
using std::vector;

class __attribute__((visibility("default"))) Index {
 public:
  Index();

  ~Index() {}

  /// \brief Add field which from schema according to schemaId
  /// \param[in] schemaId the id of schema to be added
  /// \param[in] field the field need to be added
  ///
  /// add the field to the fields_ vector
  void AddIndexField(const int64_t &schemaId, const std::string &field);

  /// \brief get stored fields
  /// \return fields stored
  std::vector<std::pair<uint64_t, std::string> > GetFields();

 private:
  std::vector<std::pair<uint64_t, std::string> > fields_;
  string database_name_;
  string table_name_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INDEX_H
