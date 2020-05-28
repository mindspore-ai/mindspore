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

#include "mindrecord/include/shard_error.h"

namespace mindspore {
namespace mindrecord {
std::string ErrnoToMessage(MSRStatus status) {
  switch (status) {
    case FAILED:
      return "operator failed";
      break;
    case SUCCESS:
      return "operator success";
      break;
    case OPEN_FILE_FAILED:
      return "open file failed";
      break;
    case CLOSE_FILE_FAILED:
      return "close file failed";
      break;
    case WRITE_METADATA_FAILED:
      return "write metadata failed";
      break;
    case WRITE_RAWDATA_FAILED:
      return "write rawdata failed";
      break;
    case GET_SCHEMA_FAILED:
      return "get schema failed";
      break;
    case ILLEGAL_RAWDATA:
      return "illegal raw data";
      break;
    case PYTHON_TO_JSON_FAILED:
      return "pybind: python object to json failed";
      break;
    case DIR_CREATE_FAILED:
      return "directory create failed";
      break;
    case OPEN_DIR_FAILED:
      return "open directory failed";
      break;
    case INVALID_STATISTICS:
      return "invalid statistics object";
      break;
    case OPEN_DATABASE_FAILED:
      return "open database failed";
      break;
    case CLOSE_DATABASE_FAILED:
      return "close database failed";
      break;
    case DATABASE_OPERATE_FAILED:
      return "database operate failed";
      break;
    case BUILD_SCHEMA_FAILED:
      return "build schema failed";
      break;
    case DIVISOR_IS_ILLEGAL:
      return "divisor is illegal";
      break;
    case INVALID_FILE_PATH:
      return "file path is invalid";
      break;
    case SECURE_FUNC_FAILED:
      return "secure function failed";
      break;
    case ALLOCATE_MEM_FAILED:
      return "allocate memory failed";
      break;
    case ILLEGAL_FIELD_NAME:
      return "illegal field name";
      break;
    case ILLEGAL_FIELD_TYPE:
      return "illegal field type";
      break;
    case SET_METADATA_FAILED:
      return "set metadata failed";
      break;
    case ILLEGAL_SCHEMA_DEFINITION:
      return "illegal schema definition";
      break;
    case ILLEGAL_COLUMN_LIST:
      return "illegal column list";
      break;
    case SQL_ERROR:
      return "sql error";
      break;
    case ILLEGAL_SHARD_COUNT:
      return "illegal shard count";
      break;
    case ILLEGAL_SCHEMA_COUNT:
      return "illegal schema count";
      break;
    case VERSION_ERROR:
      return "data version is not matched";
      break;
    case ADD_SCHEMA_FAILED:
      return "add schema failed";
      break;
    case ILLEGAL_Header_SIZE:
      return "illegal header size";
      break;
    case ILLEGAL_Page_SIZE:
      return "illegal page size";
      break;
    case ILLEGAL_SIZE_VALUE:
      return "illegal size value";
      break;
    case INDEX_FIELD_ERROR:
      return "add index fields failed";
      break;
    case GET_CANDIDATE_CATEGORYFIELDS_FAILED:
      return "get candidate category fields failed";
      break;
    case GET_CATEGORY_INFO_FAILED:
      return "get category information failed";
      break;
    case ILLEGAL_CATEGORY_ID:
      return "illegal category id";
      break;
    case ILLEGAL_ROWNUMBER_OF_PAGE:
      return "illegal row number of page";
      break;
    case ILLEGAL_SCHEMA_ID:
      return "illegal schema id";
      break;
    case DESERIALIZE_SCHEMA_FAILED:
      return "deserialize schema failed";
      break;
    case DESERIALIZE_STATISTICS_FAILED:
      return "deserialize statistics failed";
      break;
    case ILLEGAL_DB_FILE:
      return "illegal db file";
      break;
    case OVERWRITE_DB_FILE:
      return "overwrite db file";
      break;
    case OVERWRITE_MINDRECORD_FILE:
      return "overwrite mindrecord file";
      break;
    case ILLEGAL_MINDRECORD_FILE:
      return "illegal mindrecord file";
      break;
    case PARSE_JSON_FAILED:
      return "parse json failed";
      break;
    case ILLEGAL_PARAMETERS:
      return "illegal parameters";
      break;
    case GET_PAGE_BY_GROUP_ID_FAILED:
      return "get page by group id failed";
      break;
    case GET_SYSTEM_STATE_FAILED:
      return "get system state failed";
      break;
    case IO_FAILED:
      return "io operate failed";
      break;
    case MATCH_HEADER_FAILED:
      return "match header failed";
      break;
    default:
      return "invalid error no";
  }
}
}  // namespace mindrecord
}  // namespace mindspore
