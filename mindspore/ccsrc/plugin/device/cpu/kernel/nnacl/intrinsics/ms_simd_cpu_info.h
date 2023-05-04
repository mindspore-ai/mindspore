/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef NNACL_MS_SIMD_CPU_INFO_H_
#define NNACL_MS_SIMD_CPU_INFO_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ENABLE_AVX512
#define AVX512_HARDWARE_SELF_AWARENESS
#endif

#if defined(AVX512_HARDWARE_SELF_AWARENESS)
#define AVX512_HARDWARE_SELF_AWARENESS_BEGIN if (X86_Avx512_Support()) {
#define AVX512_HARDWARE_SELF_AWARENESS_END }
#else
#define AVX512_HARDWARE_SELF_AWARENESS_BEGIN
#define AVX512_HARDWARE_SELF_AWARENESS_END
#endif

typedef enum X86CpuInfoErrorCodeEnum {
  X86CPUINFO_OK = 0,
  X86CPUINFO_PLATFORM_ERR = 1,
  X86CPUINFO_AVX512_ERR,
  X86CPUINFO_AVX_ERR,
  X86CPUINFO_SSE_ERR,
  X86CPUINFO_END = 9999
} X86CpuInfoErrorCodeEnum;

const bool X86_Fma_Support(void);
const bool X86_Sse_Support(void);
const bool X86_Avx_Support(void);
const bool X86_Avx512_Support(void);

bool IsIntelX86Platform(void);
X86CpuInfoErrorCodeEnum IntelX86InstructionSetSupportCheck(void);

int IntelX86CpuInfoInit(void);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_AVX_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
