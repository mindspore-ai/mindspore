/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "include/errorcode.h"
#include <map>
#include <string>

namespace mindspore {
namespace lite {
std::string GetErrorInfo(STATUS status) {
  std::map<int, std::string> info_map = {{RET_OK, "No error occurs."},
                                         {RET_ERROR, "Common error code."},
                                         {RET_NULL_PTR, "NULL pointer returned."},
                                         {RET_PARAM_INVALID, "Invalid parameter."},
                                         {RET_NO_CHANGE, "No change."},
                                         {RET_SUCCESS_EXIT, "No error but exit."},
                                         {RET_MEMORY_FAILED, "Fail to create memory."},
                                         {RET_NOT_SUPPORT, "Fail to support."},
                                         {RET_THREAD_POOL_ERROR, "Thread pool error."},
                                         {RET_OUT_OF_TENSOR_RANGE, "Failed to check range."},
                                         {RET_INPUT_TENSOR_ERROR, "Failed to check input tensor."},
                                         {RET_REENTRANT_ERROR, "Exist executor running."},
                                         {RET_GRAPH_FILE_ERR, "Failed to verify graph file."},
                                         {RET_NOT_FIND_OP, "Failed to find operator."},
                                         {RET_INVALID_OP_NAME, "Invalid operator name."},
                                         {RET_INVALID_OP_ATTR, "Invalid operator attr."},
                                         {RET_OP_EXECUTE_FAILURE, "Failed to execution operator."},
                                         {RET_FORMAT_ERR, "Failed to checking tensor format."},
                                         {RET_INFER_ERR, "Failed to infer shape."},
                                         {RET_INFER_INVALID, "Invalid infer shape before runtime."},
                                         {RET_INPUT_PARAM_INVALID, "Invalid input param by user."}};
  return info_map.find(status) == info_map.end() ? "Unknown error" : info_map[status];
}
}  // namespace lite
}  // namespace mindspore
