/*
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

#include "wrapper/base/common_wrapper.h"
#ifdef __ANDROID__
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

bool GetSupportOptFlag() {
  bool status = false;
#ifdef ENABLE_ARM64
  int hwcap_type = 16;
  // getHwCap
  const uint32_t hwcap = getauxval(hwcap_type);
  if (hwcap & HWCAP_ASIMDDP) {
    status = true;
  } else {
    status = false;
  }
#endif
  return status;
}
