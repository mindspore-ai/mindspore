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
#include "src/nnie_memory.h"
#include "include/hi_common.h"
#include "include/mpi_sys.h"
#include "src/nnie_common.h"

namespace mindspore {
namespace nnie {
HI_S32 NnieMemMalloc(const HI_CHAR *mmb, HI_CHAR *zone, HI_U64 *pu_phy_addr, HI_VOID **ppv_vir_addr, HI_U32 size) {
  return HI_MPI_SYS_MmzAlloc(pu_phy_addr, ppv_vir_addr, mmb, zone, size);
}

HI_S32 NnieMemMallocCached(const HI_CHAR *mmb, HI_CHAR *zone, HI_U64 *pu_phy_addr, HI_VOID **ppv_vir_addr,
                           HI_U32 size) {
  return HI_MPI_SYS_MmzAlloc_Cached(pu_phy_addr, ppv_vir_addr, mmb, zone, size);
}

HI_S32 NnieMemFlushCache(HI_U64 phy_addr, HI_VOID *pv_vir_addr, HI_U32 size) {
  return HI_MPI_SYS_MmzFlushCache(phy_addr, pv_vir_addr, size);
}

HI_S32 NnieGetVirMemInfo(HI_U64 pv_vir_addr, HI_U64 *phy_addr) {
  SYS_VIRMEM_INFO_S mem_info;
  HI_S32 ret = HI_MPI_SYS_GetVirMemInfo(NNIE_CONVERT_64BIT_ADDR(HI_VOID, pv_vir_addr), &mem_info);
  if (ret == HI_SUCCESS) {
    *phy_addr = mem_info.u64PhyAddr;
  }
  return ret;
}
}  // namespace nnie
}  // namespace mindspore
