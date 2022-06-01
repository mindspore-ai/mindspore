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
#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MEMORY_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MEMORY_H_
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "include/hi_common.h"
#include "include/hi_debug.h"
#include "include/hi_comm_svp.h"
#include "include/hi_nnie.h"
#include "include/mpi_nnie.h"
#include "include/mpi_sys.h"

namespace mindspore {
namespace nnie {
#define NNIE_MEM_FREE(phy, vir)                                                     \
  do {                                                                              \
    if ((0 != (phy)) && (0 != (vir))) {                                             \
      HI_MPI_SYS_MmzFree((phy), reinterpret_cast<void *>(static_cast<HI_UL>(vir))); \
      (phy) = 0;                                                                    \
      (vir) = 0;                                                                    \
    }                                                                               \
  } while (0)

HI_S32 NnieMemMalloc(const HI_CHAR *mmb, HI_CHAR *zone, HI_U64 *pu_phy_addr, HI_VOID **ppv_vir_addr, HI_U32 size);

HI_S32 NnieMemMallocCached(const HI_CHAR *mmb, HI_CHAR *zone, HI_U64 *pu_phy_addr, HI_VOID **ppv_vir_addr, HI_U32 size);

HI_S32 NnieMemFlushCache(HI_U64 phy_addr, HI_VOID *pv_vir_addr, HI_U32 size);

HI_S32 NnieGetVirMemInfo(HI_U64 pv_vir_addr, HI_U64 *phy_addr);
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_MEMORY_H_
