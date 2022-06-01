

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

#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nnacl/errorcode.h"

typedef unsigned int DWORD;
struct X86CpuInfoContext {
  bool fma_flag_;
  bool sse4_1_flag_;
  bool avx2_flag_;
  bool avx512_flag_;
};

static struct X86CpuInfoContext g_x86_cpu_info_context_;

inline const bool X86_Fma_Support(void) { return g_x86_cpu_info_context_.fma_flag_; }

inline const bool X86_Sse_Support(void) {
#ifdef ENABLE_SSE
  return g_x86_cpu_info_context_.sse4_1_flag_;
#else
  return false;
#endif
}

inline const bool X86_Avx_Support(void) {
#ifdef ENABLE_AVX
  return g_x86_cpu_info_context_.avx2_flag_;
#else
  return false;
#endif
}

inline const bool X86_Avx512_Support(void) {
#ifdef ENABLE_AVX512
  return g_x86_cpu_info_context_.avx512_flag_;
#else
  return false;
#endif
}

void ExecuteCpuIdCmd(DWORD cmd_code, DWORD *eax_data, DWORD *ebx_data, DWORD *ecx_data, DWORD *edx_data) {
  DWORD deax, debx, decx, dedx;
  asm volatile(
    "movl %4, %%eax;\n"
    "movl $0, %%ecx;\n"
    "cpuid;\n"
    "movl %%eax, %0;\n"
    "movl %%ebx, %1;\n"
    "movl %%ecx, %2;\n"
    "movl %%edx, %3;\n"
    : "=r"(deax), "=r"(debx), "=r"(decx), "=r"(dedx)
    : "r"(cmd_code)
    : "%eax", "%ebx", "%ecx", "%edx");

  *eax_data = deax;
  *ebx_data = debx;
  *ecx_data = decx;
  *edx_data = dedx;
}

bool IsIntelX86Platform(void) {
  DWORD eax_data, ebx_data, ecx_data, edx_data;

  const int vid_info_size = 13;
  char *vid_info = malloc(sizeof(char) * vid_info_size);
  memset(vid_info, 0, vid_info_size);

  ExecuteCpuIdCmd(0, &eax_data, &ebx_data, &ecx_data, &edx_data);  // eax = 0, execute cpuid to get vid info

  memcpy(vid_info, &ebx_data, 4);      // Copy the first 4 characters to the array[0:3]
  memcpy(vid_info + 4, &edx_data, 4);  // Copy the middle 4 characters to the array[4:8]
  memcpy(vid_info + 8, &ecx_data, 4);  // Copy the last 4 characters to the array[8:12]

  int x86_intel_flag = (strcmp(vid_info, "GenuineIntel") == 0 || strcmp(vid_info, "AuthenticAMD") == 0) ? 1 : 0;

  free(vid_info);
  return x86_intel_flag;
}

int IntelX86CpuInfoInit(void) {
  if (!IsIntelX86Platform()) {
    return NNACL_ERR;
  }
  DWORD eax_data, ebx_data, ecx_data, edx_data;
  ExecuteCpuIdCmd(1, &eax_data, &ebx_data, &ecx_data, &edx_data);  // eax = 1, execute cpuid to get sse/fma flag
  g_x86_cpu_info_context_.sse4_1_flag_ = (ecx_data & (1 << 19)) == 0 ? false : true;  // sse flag is ecx 19 bit
  g_x86_cpu_info_context_.fma_flag_ = (ecx_data & (1 << 12)) == 0 ? false : true;     // fma flag is ecx 12 bit

  ExecuteCpuIdCmd(7, &eax_data, &ebx_data, &ecx_data, &edx_data);  // eax = 7, execute cpuid to get avx2/avx512 flag
  g_x86_cpu_info_context_.avx2_flag_ = (ebx_data & (1 << 5)) == 0 ? false : true;     // avx2 flag is ecx 5 bit
  g_x86_cpu_info_context_.avx512_flag_ = (ebx_data & (1 << 16)) == 0 ? false : true;  // avx512 flag is ecx 16 bit

  return NNACL_OK;
}

X86CpuInfoErrorCodeEnum IntelX86InstructionSetSupportCheck(void) {
  if (IntelX86CpuInfoInit() != NNACL_OK) {
    return X86CPUINFO_PLATFORM_ERR;
  }
#if defined(ENABLE_AVX512) && !defined(AVX512_HARDWARE_SELF_AWARENESS)
  if (!X86_Avx512_Support()) {
    return X86CPUINFO_AVX512_ERR;
  }
#endif

#ifdef ENABLE_AVX
  if (!X86_Avx_Support()) {
    return X86CPUINFO_AVX_ERR;
  }
#endif

#ifdef ENABLE_SSE
  if (!X86_Sse_Support()) {
    return X86CPUINFO_SSE_ERR;
  }
#endif
  return X86CPUINFO_OK;
}
