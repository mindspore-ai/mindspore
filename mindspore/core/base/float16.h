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
#ifndef MINDSPORE_CORE_BASE_FLOAT16_H_
#define MINDSPORE_CORE_BASE_FLOAT16_H_

#if defined(ENABLE_ARM32) || defined(ENABLE_ARM64)
// Built for lite and ARM
#include <arm_neon.h>

using float16 = float16_t;
inline float half_to_float(float16 h) { return static_cast<float>(h); }
#else
#ifndef ENABLE_MD_LITE_X86_64
#include <functional>
#include "Eigen/Core"

using float16 = Eigen::half;
using HalfToFloat = std::function<float(float16)>;
const inline HalfToFloat half_to_float = Eigen::half_impl::half_to_float;
#endif
#endif
#endif  // MINDSPORE_CORE_BASE_FLOAT16_H_
