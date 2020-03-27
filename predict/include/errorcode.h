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

#ifndef PREDICT_INCLUDE_ERRORCODE_H_
#define PREDICT_INCLUDE_ERRORCODE_H_

namespace mindspore {
namespace predict {
using STATUS = int;

/* Success */
constexpr int RET_OK = 0; /**< No error occurs. */

/* Common error code, range: [-1, -100]*/
constexpr int RET_ERROR = -1;         /**< Common error code. */
constexpr int RET_NULL_PTR = -2;      /**< NULL pointer returned.*/
constexpr int RET_PARAM_INVALID = -3; /**< Invalid parameter.*/
constexpr int RET_NO_CHANGE = -4;     /**< No change. */

/* Executor error code, range: [-101,-200] */
constexpr int RET_OUT_OF_TENSOR_RANGE = -101; /**< Failed to checking range. */
constexpr int RET_INPUT_TENSOR_ERROR = -102;  /**< Failed to checking input tensor. */
constexpr int RET_REENTRANT_ERROR = -103;     /**< Exist executor running. */

/* Graph error code, range: [-201,-300] */
constexpr int RET_GRAPH_FILE_ERR = -201; /**< Failed to verify graph file. */

/* Node error code, range: [-301,-400] */
constexpr int RET_NOT_FIND_OP = -301;        /**< Failed to find OP. */
constexpr int RET_INVALID_OP_NAME = -302;    /**< Invalid OP name. */
constexpr int RET_INVALID_OP_ATTR = -303;    /**< Invalid OP attr. */
constexpr int RET_OP_EXECUTE_FAILURE = -304; /**< Failed to execution OP. */

/* Tensor error code, range: [-401,-500] */
constexpr int RET_FORMAT_ERR = -401; /**< Failed to checking tensor format. */
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_INCLUDE_ERRORCODE_H_
