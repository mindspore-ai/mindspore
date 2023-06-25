/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_C_API_STATUS_C_H
#define MINDSPORE_INCLUDE_C_API_STATUS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum MSCompCode {
  kMSCompCodeCore = 0x00000000u,
  kMSCompCodeMD = 0x10000000u,
  kMSCompCodeME = 0x20000000u,
  kMSCompCodeMC = 0x30000000u,
  kMSCompCodeLite = 0xF0000000u,
};

typedef enum MSStatus {
  kMSStatusSuccess = 0,
  // Core
  kMSStatusCoreFailed = kMSCompCodeCore | 0x1,

  // Lite  // Common error code, range: [-1, -100)
  kMSStatusLiteError = kMSCompCodeLite | (0x0FFFFFFF & -1),            /**< Common error code. */
  kMSStatusLiteNullptr = kMSCompCodeLite | (0x0FFFFFFF & -2),          /**< NULL pointer returned.*/
  kMSStatusLiteParamInvalid = kMSCompCodeLite | (0x0FFFFFFF & -3),     /**< Invalid parameter.*/
  kMSStatusLiteNoChange = kMSCompCodeLite | (0x0FFFFFFF & -4),         /**< No change. */
  kMSStatusLiteSuccessExit = kMSCompCodeLite | (0x0FFFFFFF & -5),      /**< No error but exit. */
  kMSStatusLiteMemoryFailed = kMSCompCodeLite | (0x0FFFFFFF & -6),     /**< Fail to create memory. */
  kMSStatusLiteNotSupport = kMSCompCodeLite | (0x0FFFFFFF & -7),       /**< Fail to support. */
  kMSStatusLiteThreadPoolError = kMSCompCodeLite | (0x0FFFFFFF & -8),  /**< Error occur in thread pool. */
  kMSStatusLiteUninitializedObj = kMSCompCodeLite | (0x0FFFFFFF & -9), /**< Object is not initialized. */
  kMSStatusLiteFileError = kMSCompCodeLite | (0x0FFFFFFF & -10),       /**< Invalid file. */
  kMSStatusLiteServiceDeny = kMSCompCodeLite | (0x0FFFFFFF & -11),     /**< Denial of service. */
  kMSStatusLiteModelRebuild = kMSCompCodeLite | (0x0FFFFFFF & -12),    /**< Model has been built. */

  // Executor error code, range: [-100,-200)
  kMSStatusLiteOutOfTensorRange = kMSCompCodeLite | (0x0FFFFFFF & -100), /**< Failed to check range. */
  kMSStatusLiteInputTensorError = kMSCompCodeLite | (0x0FFFFFFF & -101), /**< Failed to check input tensor. */
  kMSStatusLiteReentrantError = kMSCompCodeLite | (0x0FFFFFFF & -102),   /**< Exist executor running. */

  // Graph error code, range: [-200,-300)
  kMSStatusLiteGraphFileError = kMSCompCodeLite | (0x0FFFFFFF & -200), /**< Failed to verify graph file. */

  // Node error code, range: [-300,-400)
  kMSStatusLiteNotFindOp = kMSCompCodeLite | (0x0FFFFFFF & -300),        /**< Failed to find operator. */
  kMSStatusLiteInvalidOpName = kMSCompCodeLite | (0x0FFFFFFF & -301),    /**< Invalid operator name. */
  kMSStatusLiteInvalidOpAttr = kMSCompCodeLite | (0x0FFFFFFF & -302),    /**< Invalid operator attr. */
  kMSStatusLiteOpExecuteFailure = kMSCompCodeLite | (0x0FFFFFFF & -303), /**< Failed to execution operator. */

  // Tensor error code, range: [-400,-500)
  kMSStatusLiteFormatError = kMSCompCodeLite | (0x0FFFFFFF & -400), /**< Failed to checking tensor format. */

  // InferShape error code, range: [-500,-600)
  kMSStatusLiteInferError = kMSCompCodeLite | (0x0FFFFFFF & -500),   /**< Failed to infer shape. */
  kMSStatusLiteInferInvalid = kMSCompCodeLite | (0x0FFFFFFF & -501), /**< Invalid infer shape before runtime. */

  // User input param error code, range: [-600, 700)
  kMSStatusLiteInputParamInvalid = kMSCompCodeLite | (0x0FFFFFFF & -600), /**< Invalid input param by user. */
} MSStatus;
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_STATUS_C_H
