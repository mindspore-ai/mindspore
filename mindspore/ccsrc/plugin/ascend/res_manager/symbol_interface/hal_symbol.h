/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_HAL_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_HAL_SYMBOL_H_
#include <string>
#include <cstdint>

#if __has_include("driver/ascend_hal_error.h")
#include "driver/ascend_hal_error.h"
#else
#include "experiment/ascend_hal/driver/ascend_hal_error.h"
#endif

#if __has_include("driver/ascend_hal_define.h")
#include "driver/ascend_hal_define.h"
#else
#include "experiment/ascend_hal/driver/ascend_hal_define.h"
#endif
#include "utils/dlopen_macro.h"
#include "runtime/base_type.h"

namespace mindspore {
namespace device {
namespace ascend {
using UINT64 = uint64_t;

ORIGIN_METHOD(halHostRegister, drvError_t, void *, UINT64, UINT32, UINT32, void **);
ORIGIN_METHOD(halHostUnregister, drvError_t, void *, UINT32);

extern halHostRegisterFunObj halHostRegister_;
extern halHostUnregisterFunObj halHostUnregister_;

void LoadHalApiSymbol(const std::string &driver_path);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_HAL_SYMBOL_H_
