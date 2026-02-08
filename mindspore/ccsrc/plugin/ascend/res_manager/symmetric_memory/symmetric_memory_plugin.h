/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_PLUGIN_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_PLUGIN_H_

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include "utils/dlopen_macro.h"
#ifdef ENABLE_SYMMETRIC_MEMORY_KERNELS
#include "shmem/shmem/include/shmem_api.h"
#else
// dummy struct for compiling without shmem
typedef struct {
  int dummy;
} shmem_init_attr_t;
#endif

// FunObj define (func_name, ret_type, arg_types...)
ORIGIN_METHOD(shmem_init_status, int);
ORIGIN_METHOD(shmem_set_attr, int, int, int, uint64_t, const char *, shmem_init_attr_t **);
ORIGIN_METHOD(shmem_init_attr, int, shmem_init_attr_t *);
ORIGIN_METHOD(shmem_finalize, int);
ORIGIN_METHOD(shmemx_get_ffts_config, uint64_t);
ORIGIN_METHOD(shmem_malloc, void *, size_t);
ORIGIN_METHOD(shmem_free, void, void *);
ORIGIN_METHOD(shmem_set_conf_store_tls, int32_t, bool, const char *, const uint32_t);

#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_SYMMETRIC_MEMORY_PLUGIN_H_
