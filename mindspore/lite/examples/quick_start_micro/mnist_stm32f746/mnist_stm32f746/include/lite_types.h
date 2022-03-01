/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_
#define MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_

namespace mindspore::lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND,    /**< no bind */
  HIGHER_CPU, /**< bind higher cpu first */
  MID_CPU     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type */
} DeviceType;

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_
