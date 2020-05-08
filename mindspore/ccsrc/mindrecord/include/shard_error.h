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

#ifndef MINDRECORD_INCLUDE_SHARD_ERROR_H_
#define MINDRECORD_INCLUDE_SHARD_ERROR_H_

#include <map>
#include <string>

namespace mindspore {
namespace mindrecord {
enum MSRStatus {
  SUCCESS = 0,
  FAILED = 1,
  OPEN_FILE_FAILED,
  CLOSE_FILE_FAILED,
  WRITE_METADATA_FAILED,
  WRITE_RAWDATA_FAILED,
  GET_SCHEMA_FAILED,
  ILLEGAL_RAWDATA,
  PYTHON_TO_JSON_FAILED,
  DIR_CREATE_FAILED,
  OPEN_DIR_FAILED,
  INVALID_STATISTICS,
  OPEN_DATABASE_FAILED,
  CLOSE_DATABASE_FAILED,
  DATABASE_OPERATE_FAILED,
  BUILD_SCHEMA_FAILED,
  DIVISOR_IS_ILLEGAL,
  INVALID_FILE_PATH,
  SECURE_FUNC_FAILED,
  ALLOCATE_MEM_FAILED,
  ILLEGAL_FIELD_NAME,
  ILLEGAL_FIELD_TYPE,
  SET_METADATA_FAILED,
  ILLEGAL_SCHEMA_DEFINITION,
  ILLEGAL_COLUMN_LIST,
  SQL_ERROR,
  ILLEGAL_SHARD_COUNT,
  ILLEGAL_SCHEMA_COUNT,
  VERSION_ERROR,
  ADD_SCHEMA_FAILED,
  ILLEGAL_Header_SIZE,
  ILLEGAL_Page_SIZE,
  ILLEGAL_SIZE_VALUE,
  INDEX_FIELD_ERROR,
  GET_CANDIDATE_CATEGORYFIELDS_FAILED,
  GET_CATEGORY_INFO_FAILED,
  ILLEGAL_CATEGORY_ID,
  ILLEGAL_ROWNUMBER_OF_PAGE,
  ILLEGAL_SCHEMA_ID,
  DESERIALIZE_SCHEMA_FAILED,
  DESERIALIZE_STATISTICS_FAILED,
  ILLEGAL_DB_FILE,
  OVERWRITE_DB_FILE,
  OVERWRITE_MINDRECORD_FILE,
  ILLEGAL_MINDRECORD_FILE,
  PARSE_JSON_FAILED,
  ILLEGAL_PARAMETERS,
  GET_PAGE_BY_GROUP_ID_FAILED,
  GET_SYSTEM_STATE_FAILED,
  IO_FAILED,
  MATCH_HEADER_FAILED
};

// convert error no to string message
std::string ErrnoToMessage(MSRStatus status);
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_ERROR_H_
