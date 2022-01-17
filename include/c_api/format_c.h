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
#ifndef MINDSPORE_INCLUDE_C_API_FORMAT_C_H
#define MINDSPORE_INCLUDE_C_API_FORMAT_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MSFormat {
  kMSFormatNCHW = 0,
  kMSFormatNHWC = 1,
  kMSFormatNHWC4 = 2,
  kMSFormatHWKC = 3,
  kMSFormatHWCK = 4,
  kMSFormatKCHW = 5,
  kMSFormatCKHW = 6,
  kMSFormatKHWC = 7,
  kMSFormatCHWK = 8,
  kMSFormatHW = 9,
  kMSFormatHW4 = 10,
  kMSFormatNC = 11,
  kMSFormatNC4 = 12,
  kMSFormatNC4HW4 = 13,
  kMSFormatNCDHW = 15,
  kMSFormatNWC = 16,
  kMSFormatNCW = 17
} MSFormat;

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_FORMAT_C_H
