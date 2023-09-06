/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file error_code.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTILS_ERROR_CODE_H_
#define CUSTOMIZE_OP_PROTO_UTILS_ERROR_CODE_H_

namespace ge {
// error code for report purpose.
// 30000~34999 for aicpu engine error
// and 35000~39999 for infershape error of aicpu op
enum ViewErrorCode {
  INVALID_INFER_SHAPE = 14001,
  INVALID_INPUT_SHAPE = 35000,
  INVALID_ATTR_VALUE = 35001,
  INVALID_ATTR_SIZE = 35002,
  OTHER_ERROR = 35003,
  AICPU_INFER_SHAPE_ERROR = 39999,
  INVALID_CONV_ATTR_VALUE = 50029,
  INVALID_CONV_SET_ATTR = 50057,
  INVALID_CONV_SHAPE = 50058,
  INVALID_MISS_INPUT = 70001,
  INVALID_INPUT_FORMAT = 70002,
  INVALID_INPUT_DTYPE = 70003,
  INVALID_INPUT_TYPE = 70004,
  INVALID_GET_ATTR = 70005,
  INVALID_SET_ATTR = 70006,
  INVALID_OPS_ATTR_VALUE = 70007,
  FAILED_UPDATE_OP = 70008,
  INVALID_SHAPE = 70009,
  INVALID_SHAPE_SIZE = 70010,
  INVALID_SHAPE_DIM = 70011,
  INVALID_BROADCAST_SHAPE = 70012,
  INVALID_TWO_INPUT_DTYPE = 70013,
  INVALID_AIPP_ERROR = 70014,
  INVALID_ONE_INPUT_SHAPE = 70015,
  INVALID_TWO_INPUT_SHAPE = 70016,
  INVALID_ONE_OUTPUT_SHAPE = 70017,
  FAILED_GET_COMPILIE_PARAMS = 70018,
  VECTOR_INNER_ERROR = 89999,
  CUBE_INNER_ERROR = 69999,
  CUBE_INNER_ERROR_PLUGIN = 59999
};
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_UTILS_ERROR_CODE_H_
