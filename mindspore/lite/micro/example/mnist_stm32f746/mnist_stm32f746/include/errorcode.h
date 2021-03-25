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

#ifndef MINDSPORE_LITE_INCLUDE_ERRORCODE_H_
#define MINDSPORE_LITE_INCLUDE_ERRORCODE_H_

#include "include/lite_utils.h"

namespace mindspore {
namespace lite {
/// \brief STATUS defined for holding error code in MindSpore Lite.
using STATUS = int;

/* Success */
constexpr int RET_OK = 0; /**< No error occurs. */

/* Common error code, range: [-1, -100）*/
constexpr int RET_ERROR = -1;             /**< Common error code. */
constexpr int RET_NULL_PTR = -2;          /**< NULL pointer returned.*/
constexpr int RET_PARAM_INVALID = -3;     /**< Invalid parameter.*/
constexpr int RET_NO_CHANGE = -4;         /**< No change. */
constexpr int RET_SUCCESS_EXIT = -5;      /**< No error but exit. */
constexpr int RET_MEMORY_FAILED = -6;     /**< Fail to create memory. */
constexpr int RET_NOT_SUPPORT = -7;       /**< Fail to support. */
constexpr int RET_THREAD_POOL_ERROR = -8; /**< Error occur in thread pool. */

/* Executor error code, range: [-100,-200) */
constexpr int RET_OUT_OF_TENSOR_RANGE = -100; /**< Failed to check range. */
constexpr int RET_INPUT_TENSOR_ERROR = -101;  /**< Failed to check input tensor. */
constexpr int RET_REENTRANT_ERROR = -102;     /**< Exist executor running. */

/* Graph error code, range: [-200,-300) */
constexpr int RET_GRAPH_FILE_ERR = -200; /**< Failed to verify graph file. */

/* Node error code, range: [-300,-400) */
constexpr int RET_NOT_FIND_OP = -300;        /**< Failed to find operator. */
constexpr int RET_INVALID_OP_NAME = -301;    /**< Invalid operator name. */
constexpr int RET_INVALID_OP_ATTR = -302;    /**< Invalid operator attr. */
constexpr int RET_OP_EXECUTE_FAILURE = -303; /**< Failed to execution operator. */

/* Tensor error code, range: [-400,-500) */
constexpr int RET_FORMAT_ERR = -400; /**< Failed to checking tensor format. */

/* InferShape error code, range: [-500,-600) */
constexpr int RET_INFER_ERR = -500;     /**< Failed to infer shape. */
constexpr int RET_INFER_INVALID = -501; /**< Invalid infer shape before runtime. */

/* User input param error code, range: [-600, 700)*/
constexpr int RET_INPUT_PARAM_INVALID = -600; /**< Invalid input param by user. */

/// \brief Print description of errorcode.
///
/// \param[in] error_code define return status of procedure.
///
/// \return String of errorcode info.
String GetErrorInfo(STATUS error_code);

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_ERRORCODE_H_
