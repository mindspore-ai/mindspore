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
#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_PRINT_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_PRINT_H_
#include "include/mpi_nnie.h"
#include "include/hi_type.h"
#include "src/nnie_common.h"
#include "src/nnie_memory.h"

#define LOG_TAG1 "NNIE"
#define LOGE(format, ...)                                                                       \
  do {                                                                                          \
    if (1) {                                                                                    \
      fprintf(stderr, "\n[ERROR] " LOG_TAG1 " [" __FILE__ ":%d] %s] ", __LINE__, __FUNCTION__); \
      fprintf(stderr, format, ##__VA_ARGS__);                                                   \
    }                                                                                           \
  } while (0)

#define LOGW(format, ...)                                                                         \
  do {                                                                                            \
    if (1) {                                                                                      \
      fprintf(stderr, "\n[Warning] " LOG_TAG1 " [" __FILE__ ":%d] %s] ", __LINE__, __FUNCTION__); \
      fprintf(stderr, format, ##__VA_ARGS__);                                                     \
    }                                                                                             \
  } while (0)

#define LOGI(format, ...)                                                                         \
  do {                                                                                            \
    if (0) {                                                                                      \
      fprintf(stderr, "\n[Warning] " LOG_TAG1 " [" __FILE__ ":%d] %s] ", __LINE__, __FUNCTION__); \
      fprintf(stderr, format, ##__VA_ARGS__);                                                     \
    }                                                                                             \
  } while (0)

constexpr int kMaxSize = 1024;
constexpr int kDecimal = 10;

namespace mindspore {
namespace nnie {
HI_S32 NniePrintReportResult(NnieParam *pst_nnie_param);

HI_S32 NniePrintReportResultInputSeg(NnieParam *pst_nnie_param, int segnum);
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_PRINT_H_
