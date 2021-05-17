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
#include <string>
#ifndef MINDSPORE_LITE_SRC_CPU_INFO_H
#define MINDSPORE_LITE_SRC_CPU_INFO_H
namespace mindspore::lite {
#define CPUINFO_HARDWARE_VALUE_MAX 64
/* As per include/sys/system_properties.h in Android NDK */
#define CPUINFO_ARM_MIDR_IMPLEMENTER_MASK UINT32_C(0xFF000000)
#define CPUINFO_ARM_MIDR_PART_MASK UINT32_C(0x0000FFF0)
#define CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET 24
#define CPUINFO_ARM_MIDR_PART_OFFSET 4
typedef struct AndroidCpuInfo {
  uint32_t cpu_implementer = 0;
  uint32_t cpu_part = 0;
  std::string hardware = "";
} AndroidCpuInfo;

class CpuInfo {
 public:
  CpuInfo() = default;
  virtual ~CpuInfo() = default;
  void GetArmProcCpuInfo(AndroidCpuInfo *android_cpu_info);
  uint32_t ParseArmCpuImplementer(const std::string &suffix);
  uint32_t ParseArmCpuPart(const std::string &suffix);
  uint32_t MidrSetPart(uint32_t part);
  uint32_t MidrSetImplementer(uint32_t implementer);
  bool ArmIsSupportFp16();
  uint32_t StringToDigit(const std::string &str);

 private:
  bool fp16_flag_ = false;
  uint32_t midr_ = 0;
  AndroidCpuInfo android_cpu_info_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CPU_INFO_H
#endif
