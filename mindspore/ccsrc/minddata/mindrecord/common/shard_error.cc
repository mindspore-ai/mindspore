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

#include "minddata/mindrecord/include/shard_error.h"

namespace mindspore {
namespace mindrecord {
static const std::map<MSRStatus, std::string> kErrnoToMessage = {
  {FAILED, "operator failed"},
  {SUCCESS, "operator success"},
  {OPEN_FILE_FAILED, "open file failed"},
  {CLOSE_FILE_FAILED, "close file failed"},
  {WRITE_METADATA_FAILED, "write metadata failed"},
  {WRITE_RAWDATA_FAILED, "write rawdata failed"},
  {GET_SCHEMA_FAILED, "get schema failed"},
  {ILLEGAL_RAWDATA, "illegal raw data"},
  {PYTHON_TO_JSON_FAILED, "pybind: python object to json failed"},
  {DIR_CREATE_FAILED, "directory create failed"},
  {OPEN_DIR_FAILED, "open directory failed"},
  {INVALID_STATISTICS, "invalid statistics object"},
  {OPEN_DATABASE_FAILED, "open database failed"},
  {CLOSE_DATABASE_FAILED, "close database failed"},
  {DATABASE_OPERATE_FAILED, "database operate failed"},
  {BUILD_SCHEMA_FAILED, "build schema failed"},
  {DIVISOR_IS_ILLEGAL, "divisor is illegal"},
  {INVALID_FILE_PATH, "file path is invalid"},
  {SECURE_FUNC_FAILED, "secure function failed"},
  {ALLOCATE_MEM_FAILED, "allocate memory failed"},
  {ILLEGAL_FIELD_NAME, "illegal field name"},
  {ILLEGAL_FIELD_TYPE, "illegal field type"},
  {SET_METADATA_FAILED, "set metadata failed"},
  {ILLEGAL_SCHEMA_DEFINITION, "illegal schema definition"},
  {ILLEGAL_COLUMN_LIST, "illegal column list"},
  {SQL_ERROR, "sql error"},
  {ILLEGAL_SHARD_COUNT, "illegal shard count"},
  {ILLEGAL_SCHEMA_COUNT, "illegal schema count"},
  {VERSION_ERROR, "data version is not matched"},
  {ADD_SCHEMA_FAILED, "add schema failed"},
  {ILLEGAL_Header_SIZE, "illegal header size"},
  {ILLEGAL_Page_SIZE, "illegal page size"},
  {ILLEGAL_SIZE_VALUE, "illegal size value"},
  {INDEX_FIELD_ERROR, "add index fields failed"},
  {GET_CANDIDATE_CATEGORYFIELDS_FAILED, "get candidate category fields failed"},
  {GET_CATEGORY_INFO_FAILED, "get category information failed"},
  {ILLEGAL_CATEGORY_ID, "illegal category id"},
  {ILLEGAL_ROWNUMBER_OF_PAGE, "illegal row number of page"},
  {ILLEGAL_SCHEMA_ID, "illegal schema id"},
  {DESERIALIZE_SCHEMA_FAILED, "deserialize schema failed"},
  {DESERIALIZE_STATISTICS_FAILED, "deserialize statistics failed"},
  {ILLEGAL_DB_FILE, "illegal db file"},
  {OVERWRITE_DB_FILE, "overwrite db file"},
  {OVERWRITE_MINDRECORD_FILE, "overwrite mindrecord file"},
  {ILLEGAL_MINDRECORD_FILE, "illegal mindrecord file"},
  {PARSE_JSON_FAILED, "parse json failed"},
  {ILLEGAL_PARAMETERS, "illegal parameters"},
  {GET_PAGE_BY_GROUP_ID_FAILED, "get page by group id failed"},
  {GET_SYSTEM_STATE_FAILED, "get system state failed"},
  {IO_FAILED, "io operate failed"},
  {MATCH_HEADER_FAILED, "match header failed"}};

std::string ErrnoToMessage(MSRStatus status) {
  auto iter = kErrnoToMessage.find(status);
  if (iter != kErrnoToMessage.end()) {
    return kErrnoToMessage.at(status);
  } else {
    return "invalid error no";
  }
}
}  // namespace mindrecord
}  // namespace mindspore
