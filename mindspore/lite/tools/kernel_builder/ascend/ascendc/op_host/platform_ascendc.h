/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef PLATFORM_ASCENDC_H
#define PLATFORM_ASCENDC_H

#include <cstdint>

namespace fe {
class PlatFormInfos;
}

namespace platform_ascendc {
enum class CoreMemType { L0_A = 0, L0_B = 1, L0_C = 2, L1 = 3, L2 = 4, UB = 5, HBM = 6, RESERVED };

enum class SocVersion { ASCEND910 = 0, ASCEND910B, ASCEND310P, RESERVED_VERSION = 99999 };

class PlatformAscendC {
 public:
  PlatformAscendC() = delete;
  ~PlatformAscendC() {}
  explicit PlatformAscendC(fe::PlatFormInfos *platformInfo) : platformInfo_(platformInfo) {}
  /**
   * Get Core Number
   * On Ascend910B MIX model, return AICore number
   * @return core number by core type
   */
  uint32_t GetCoreNum(void) const;
  /**
   * Get Core Number AiCore
   * @return ai_core_num
   */
  uint32_t GetCoreNumAic(void) const;
  /**
   * Get Core Number VectorCore
   * @return vector_core_num
   */
  uint32_t GetCoreNumAiv(void) const;
  /**
   * Calc task schedule block dim
   * @sliceNum number slice of data division
   * @aicCoreNum value of GetCoreNumAic() if used cube API, otherwise 0
   * @aivCoreNum value of GetCoreNumAiv() if used vector API, otherwise 0
   * @return task schedule block dim
   */
  uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const;
  /**
   * Get Work Space Size
   * @return work sapce size by chip type
   */
  uint32_t GetLibApiWorkSpaceSize(void) const;
  void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const;
  void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const;
  /**
   * Get Soc Version Enum
   * @return Enum SocVersion
   */
  SocVersion GetSocVersion(void) const;

 private:
  fe::PlatFormInfos *platformInfo_;
};
}  // namespace platform_ascendc
#endif
