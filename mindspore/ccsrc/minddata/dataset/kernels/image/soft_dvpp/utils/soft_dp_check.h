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

#ifndef SOFT_DP_CHECK_H
#define SOFT_DP_CHECK_H

#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_log.h"

#define CHECK_COND_FAIL_RETURN(model, cond, ...)                \
  do {                                                          \
    if (!(cond)) {                                              \
      DP_LOG(model, DP_ERR, "check condition: %s fail", #cond); \
      return __VA_ARGS__;                                       \
    }                                                           \
  } while (0)

#define VPC_CHECK_COND_FAIL_RETURN(cond, ret) CHECK_COND_FAIL_RETURN("VPC", cond, ret)

#define CHECK_COND_FAIL_PRINT_RETURN(module, cond, ret, format, argv...) \
  do {                                                                   \
    if (!(cond)) {                                                       \
      DP_LOG(module, DP_ERR, format, ##argv);                            \
      return ret;                                                        \
    }                                                                    \
  } while (0)

#define VPC_CHECK_COND_FAIL_PRINT_RETURN(cond, ret, format, argv...) \
  CHECK_COND_FAIL_PRINT_RETURN("VPC", cond, ret, format, ##argv)

#define JPEGD_CHECK_COND_FAIL_PRINT_RETURN(cond, ret, format, argv...) \
  CHECK_COND_FAIL_PRINT_RETURN("JPEGD", cond, ret, format, ##argv)

#endif  // SOFT_DP_CHECK_H
