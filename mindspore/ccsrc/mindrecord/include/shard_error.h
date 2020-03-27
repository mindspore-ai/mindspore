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
#include "utils/error_code.h"

namespace mindspore {
namespace mindrecord {
DE_ERRORNO_MINDRECORD(OPEN_FILE_FAILED, 0, "open file failed");
DE_ERRORNO_MINDRECORD(CLOSE_FILE_FAILED, 1, "close file failed");
DE_ERRORNO_MINDRECORD(WRITE_METADATA_FAILED, 2, "write metadata failed");
DE_ERRORNO_MINDRECORD(WRITE_RAWDATA_FAILED, 3, "write rawdata failed");
DE_ERRORNO_MINDRECORD(GET_SCHEMA_FAILED, 4, "get schema failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_RAWDATA, 5, "illegal raw data");
DE_ERRORNO_MINDRECORD(PYTHON_TO_JSON_FAILED, 6, "pybind: python object to json failed");
DE_ERRORNO_MINDRECORD(DIR_CREATE_FAILED, 7, "directory create failed");
DE_ERRORNO_MINDRECORD(OPEN_DIR_FAILED, 8, "open directory failed");
DE_ERRORNO_MINDRECORD(INVALID_STATISTICS, 9, "invalid statistics object");
DE_ERRORNO_MINDRECORD(OPEN_DATABASE_FAILED, 10, "open database failed");
DE_ERRORNO_MINDRECORD(CLOSE_DATABASE_FAILED, 11, "close database failed");
DE_ERRORNO_MINDRECORD(DATABASE_OPERATE_FAILED, 12, "database operate failed");
DE_ERRORNO_MINDRECORD(BUILD_SCHEMA_FAILED, 13, "build schema failed");
DE_ERRORNO_MINDRECORD(DIVISOR_IS_ILLEGAL, 14, "divisor is illegal");
DE_ERRORNO_MINDRECORD(INVALID_FILE_PATH, 15, "file path is invalid");
DE_ERRORNO_MINDRECORD(SECURE_FUNC_FAILED, 16, "secure function failed");
DE_ERRORNO_MINDRECORD(ALLOCATE_MEM_FAILED, 17, "allocate memory failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_FIELD_NAME, 18, "illegal field name");
DE_ERRORNO_MINDRECORD(ILLEGAL_FIELD_TYPE, 19, "illegal field type");
DE_ERRORNO_MINDRECORD(SET_METADATA_FAILED, 20, "set metadata failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_SCHEMA_DEFINITION, 21, "illegal schema definition");
DE_ERRORNO_MINDRECORD(ILLEGAL_COLUMN_LIST, 22, "illegal column list");
DE_ERRORNO_MINDRECORD(SQL_ERROR, 23, "sql error");
DE_ERRORNO_MINDRECORD(ILLEGAL_SHARD_COUNT, 24, "illegal shard count");
DE_ERRORNO_MINDRECORD(ILLEGAL_SCHEMA_COUNT, 25, "illegal schema count");
DE_ERRORNO_MINDRECORD(VERSION_ERROR, 26, "data version is not matched");
DE_ERRORNO_MINDRECORD(ADD_SCHEMA_FAILED, 27, "add schema failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_Header_SIZE, 28, "illegal header size");
DE_ERRORNO_MINDRECORD(ILLEGAL_Page_SIZE, 29, "illegal page size");
DE_ERRORNO_MINDRECORD(ILLEGAL_SIZE_VALUE, 30, "illegal size value");
DE_ERRORNO_MINDRECORD(INDEX_FIELD_FAILED, 31, "add index fields failed");
DE_ERRORNO_MINDRECORD(GET_CANDIDATE_CATEGORYFIELDS_FAILED, 32, "get candidate categoryFields failed");
DE_ERRORNO_MINDRECORD(GET_CATEGORY_INFO, 33, "get category information failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_CATEGORY_ID, 34, "illegal category id");
DE_ERRORNO_MINDRECORD(ILLEGAL_ROWNUMBER_OF_PAGE, 35, "illegal row number of page");
DE_ERRORNO_MINDRECORD(ILLEGAL_SCHEMA_ID, 36, "illegal schema id");
DE_ERRORNO_MINDRECORD(DESERIALIZE_SCHEMA_FAILED, 37, "deserialize schema failed");
DE_ERRORNO_MINDRECORD(DESERIALIZE_STATISTICS_FAILED, 38, "deserialize statistics failed");
DE_ERRORNO_MINDRECORD(ILLEGAL_DB_FILE, 39, "illegal db file.");
DE_ERRORNO_MINDRECORD(OVERWRITE_DB_FILE, 40, "overwrite db file.");
DE_ERRORNO_MINDRECORD(OVERWRITE_MINDRECORD_FILE, 41, "overwrite mindrecord file.");
DE_ERRORNO_MINDRECORD(ILLEGAL_MINDRECORD_FILE, 42, "illegal mindrecord file.");
DE_ERRORNO_MINDRECORD(PARSE_JSON_FAILED, 43, "parse json failed.");
DE_ERRORNO_MINDRECORD(ILLEGAL_PARAMETERS, 44, "illegal parameters.");
DE_ERRORNO_MINDRECORD(GET_PAGE_BY_GROUP_ID_FAILED, 46, "get page by group id failed.");
DE_ERRORNO_MINDRECORD(GET_SYSTEM_STATE_FAILED, 47, "get system state failed.");
DE_ERRORNO_MINDRECORD(IO_FAILED, 48, "io operate failed.");

enum MSRStatus {
  SUCCESS = 0,
  FAILED = 1,
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_ERROR_H_
