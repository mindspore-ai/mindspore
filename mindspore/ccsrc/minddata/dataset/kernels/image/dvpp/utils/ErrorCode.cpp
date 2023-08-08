/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/dvpp/utils/ErrorCode.h"

#include "minddata/dataset/util/log_adapter.h"

std::string GetAppErrCodeInfo(const APP_ERROR err) {
  if ((err < APP_ERR_ACL_END) && (err >= APP_ERR_ACL_FAILURE)) {
    return APP_ERR_ACL_LOG_STRING[((err < 0) ? (err + APP_ERR_ACL_END + 1) : err)];
  } else if ((err < APP_ERR_COMM_END) && (err > APP_ERR_COMM_BASE)) {
    return (err - APP_ERR_COMM_BASE) < static_cast<int>(sizeof(APP_ERR_COMMON_LOG_STRING)) /
                                         static_cast<int>(sizeof(APP_ERR_COMMON_LOG_STRING[0]))
             ? APP_ERR_COMMON_LOG_STRING[err - APP_ERR_COMM_BASE]
             : "Undefine the error code information";
  } else if ((err < APP_ERR_DVPP_END) && (err > APP_ERR_DVPP_BASE)) {
    return (err - APP_ERR_DVPP_BASE) <
               static_cast<int>(sizeof(APP_ERR_DVPP_LOG_STRING)) / static_cast<int>(sizeof(APP_ERR_DVPP_LOG_STRING[0]))
             ? APP_ERR_DVPP_LOG_STRING[err - APP_ERR_DVPP_BASE]
             : "Undefine the error code information";
  } else if ((err < APP_ERR_QUEUE_END) && (err > APP_ERR_QUEUE_BASE)) {
    return (err - APP_ERR_QUEUE_BASE) < static_cast<int>(sizeof(APP_ERR_QUEUE_LOG_STRING)) /
                                          static_cast<int>(sizeof(APP_ERR_QUEUE_LOG_STRING[0]))
             ? APP_ERR_QUEUE_LOG_STRING[err - APP_ERR_QUEUE_BASE]
             : "Undefine the error code information";
  } else {
    return "Error code unknown";
  }
}

void AssertErrorCode(int code, const std::string &file, const std::string &function, int line) {
  if (code != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed at " << file << "->" << function << "->" << line << ": error code=" << code;
  }
}

void CheckErrorCode(int code, const std::string &file, const std::string &function, int line) {
  if (code != APP_ERR_OK) {
    MS_LOG(ERROR) << "Failed at " << file << "->" << function << "->" << line << ": error code=" << code;
  }
}
