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

#ifndef MINDSPORE_LITE_INCLUDE_THREAD_POOL_CONFIG_H_
#define MINDSPORE_LITE_INCLUDE_THREAD_POOL_CONFIG_H_

/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum Mode {
  MID_CPU = -1,   /**< bind middle cpu first */
  HIGHER_CPU = 1, /**< bind higher cpu first */
  NO_BIND = 0     /**< no bind */
} CpuBindMode;

/// \brief ThreadPoolId defined for specifying which thread pool to use.
typedef enum Id {
  THREAD_POOL_DEFAULT = 0, /**< default thread pool id */
  THREAD_POOL_SECOND = 1,  /**< the second thread pool id */
  THREAD_POOL_THIRD = 2,   /**< the third thread pool id */
  THREAD_POOL_FOURTH = 3   /**< the fourth thread pool id */
} ThreadPoolId;

#endif  // LITE_MINDSPORE_LITE_INCLUDE_THREAD_POOL_CONFIG_H_
