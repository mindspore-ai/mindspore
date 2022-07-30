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
#ifdef ENABLE_ARM
#include "src/litert/cpu_info.h"
#include <set>
#include <fstream>
#include "src/common/log_adapter.h"
#include "nnacl/nnacl_utils.h"
#if defined(ENABLE_ARM64) && !defined(SUPPORT_NNIE) && !defined(MS_COMPILE_IOS)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#ifdef MS_COMPILE_IOS
#include <mach/mach.h>
#include <mach/machine.h>
#include <mach/thread_act.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include "TargetConditionals.h"
// define missing cpu model for old sdk
#ifndef CPUFAMILY_ARM_HURRICANE
#define CPUFAMILY_ARM_HURRICANE 0x67ceee93
#endif
// A11
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
// A12
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST 0x07d34b9f
#endif
// A13
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif
// A14
#ifndef CPUFAMILY_ARM_FIRESTORM_ICESTORM
#define CPUFAMILY_ARM_FIRESTORM_ICESTORM 0x1b588bb3
#endif
// A15
#ifndef CPUFAMILY_ARM_AVALANCHE_BLIZZARD
#define CPUFAMILY_ARM_AVALANCHE_BLIZZARD 0xda33d83d
#endif
#endif
namespace mindspore::lite {
#ifdef __ANDROID__
uint32_t CpuInfo::MidrSetPart(uint32_t part) {
  return ((part << ARM_CPU_PART_OFFSET) & ARM_CPU_PART_MASK) | (midr_ & ~ARM_CPU_PART_MASK);
}

uint32_t CpuInfo::MidrSetImplementer(uint32_t implementer) {
  return ((implementer << ARM_CPU_IMPLEMENTER_OFFSET) & ARM_CPU_IMPLEMENTER_MASK) | (midr_ & ~ARM_CPU_IMPLEMENTER_MASK);
}

uint32_t CpuInfo::StringToDigit(const std::string &str) {
  // hex string to digit
  constexpr size_t base_16 = 16;
  constexpr size_t base_10 = 10;
  // verify hex prefix '0' and 'x'
  if (str[0] != '0' || str[1] != 'x') {
    return 0;
  }
  auto str_length = str.length();
  uint32_t str_digit = 0;
  for (unsigned int i = 2; i < str_length; ++i) {
    auto tmp_char = str[i];
    uint32_t digit;
    if (tmp_char >= '0' && tmp_char <= '9') {
      digit = tmp_char - '0';
    } else if ((uint32_t)(tmp_char - 'A') < 6) {
      digit = base_10 + (tmp_char - 'A');
    } else if ((uint32_t)(tmp_char - 'a') < 6) {
      digit = base_10 + (tmp_char - 'a');
    } else {
      return 0;
    }
    str_digit = str_digit * base_16 + digit;
  }
  return str_digit;
}

uint32_t CpuInfo::ParseArmCpuPart(const std::string &cpu_part) {
  // cpu_part string length is in [3, 5]
  constexpr size_t cpu_part_min_len = 3;
  constexpr size_t cpu_part_max_len = 5;
  auto cpu_part_length = cpu_part.length();
  if (cpu_part_length < cpu_part_min_len || cpu_part_length > cpu_part_max_len) {
    return 0;
  }
  return StringToDigit(cpu_part);
}

uint32_t CpuInfo::ParseArmCpuImplementer(const std::string &str) {
  auto str_length = str.length();
  switch (str_length) {
    case 3:
    case 4:
      break;
    default:
      return 0;
  }
  return StringToDigit(str);
}

void CpuInfo::GetArmProcCpuInfo(AndroidCpuInfo *android_cpu_info) {
  // only get cpu part, implementer and hardware
  std::ifstream infile("/proc/cpuinfo", std::ios::in);
  std::string line;
  while (getline(infile, line)) {
    for (unsigned int i = 0; i < line.length(); ++i) {
      if (line[i] == ':') {
        std::string prefix = line.substr(0, i);
        prefix.erase(0, prefix.find_first_not_of(' '));
        prefix.erase(prefix.find_last_not_of('\t') + 1);
        std::string suffix = line.substr(i + 2);
        if (prefix == "CPU implementer" && android_cpu_info->cpu_implementer == 0) {
          android_cpu_info->cpu_implementer = ParseArmCpuImplementer(suffix);
        } else if (prefix == "CPU part" && android_cpu_info->cpu_part == 0) {
          android_cpu_info->cpu_part = ParseArmCpuPart(suffix);
        } else if (prefix == "Hardware" && android_cpu_info->hardware.empty()) {
          android_cpu_info->hardware = suffix;
        }
      }
    }
  }
  infile.close();
}
#endif

bool CpuInfo::ArmIsSupportFp16() {
#ifdef MS_COMPILE_IOS
  unsigned int value = 0;
  size_t len = sizeof(value);
  sysctlbyname("hw.cpufamily", &value, &len, NULL, 0);
  if (value == CPUFAMILY_ARM_MONSOON_MISTRAL || value == CPUFAMILY_ARM_VORTEX_TEMPEST ||
      value == CPUFAMILY_ARM_LIGHTNING_THUNDER || value == CPUFAMILY_ARM_FIRESTORM_ICESTORM ||
      CPUFAMILY_ARM_AVALANCHE_BLIZZARD) {
    return true;
  }
  return false;
#else
#ifdef __ANDROID__
#ifdef ENABLE_ARM32
  GetArmProcCpuInfo(&android_cpu_info_);
  midr_ = MidrSetPart(android_cpu_info_.cpu_part);
  midr_ = MidrSetImplementer(android_cpu_info_.cpu_implementer);
  midr_ = (ARM_CPU_IMPLEMENTER_MASK | ARM_CPU_PART_MASK) & midr_;
  std::set<uint32_t> cpu_list_support_fp16 = {
    UINT32_C(0x4800D400),  // Cortex-A76 in HiSilicon Cpu
    UINT32_C(0x4100D050),  // Arm Cortex-A55
    UINT32_C(0x4100D060),  // Arm Cortex-A65
    UINT32_C(0x4100D0B0),  // Arm Cortex-A76
    UINT32_C(0x4100D0E0),  // Arm Cortex-A76-AE
    UINT32_C(0x4100D0D0),  // Arm Cortex-A77
    UINT32_C(0x4100D0C0),  // Neoverse-N1 Cpu
    UINT32_C(0x53000030),  // Exynos-M4 Cpu
    UINT32_C(0x53000040),  // Exynos-M5 Cpu
    UINT32_C(0x51008050),  // Cortex-A55 in Kryo-485-Silver
    UINT32_C(0x51008040),  // Cortex-A76 in Kryo-485-Gold
    UINT32_C(0x51008030),  // Cortex-A55 in Kryo-385-Silver
    UINT32_C(0x51008020)   // Cortex-A75 in Kryo-385-Gold
  };
  if (cpu_list_support_fp16.find(midr_) != cpu_list_support_fp16.end()) {
    fp16_flag_ = true;
  }
#ifdef Debug
  if (!fp16_flag_) {
    MS_LOG(DEBUG) << "cpu midr_:" << midr_ << "is not support fp16!";
  }
#endif
#elif defined(ENABLE_ARM64)
  int hwcap_type = 16;
  uint32_t hwcap = getHwCap(hwcap_type);
  if (hwcap & HWCAP_FPHP) {
    MS_LOG(DEBUG) << "Hw cap support FP16, hwcap: 0x" << hwcap;
    fp16_flag_ = true;
  } else {
    MS_LOG(DEBUG) << "Hw cap NOT support FP16, hwcap: 0x" << hwcap;
  }
#endif
#elif defined(ENABLE_ARM64) && defined(MACHINE_LINUX_ARM64)
  const uint32_t hwcap = getauxval(AT_HWCAP);
  if (hwcap & HWCAP_FPHP) {
    MS_LOG(DEBUG) << "Hw cap support FP16, hwcap: 0x" << hwcap;
    fp16_flag_ = true;
  } else {
    MS_LOG(DEBUG) << "Hw cap NOT support FP16, hwcap: 0x" << hwcap;
  }
#endif
  return fp16_flag_;
#endif
}
}  // namespace mindspore::lite
#endif
