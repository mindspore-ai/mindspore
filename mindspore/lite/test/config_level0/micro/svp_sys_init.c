/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/hi_comm_vb.h"
#include "include/mpi_sys.h"
#include "include/mpi_vb.h"
#include "include/hi_common.h"

int SvpSysInit(void) {
  HI_S32 ret = HI_SUCCESS;
  VB_CONFIG_S struVbConf;

  HI_MPI_SYS_Exit();
  HI_MPI_VB_Exit();

  const int kMaxPoolCnt = 2;
  const int kBlkSize = 768 * 576 * 2;
  memset(&struVbConf, 0, sizeof(VB_CONFIG_S));
  struVbConf.u32MaxPoolCnt = kMaxPoolCnt;
  struVbConf.astCommPool[1].u64BlkSize = kBlkSize;
  struVbConf.astCommPool[1].u32BlkCnt = 1;

  ret = HI_MPI_VB_SetConfig((const VB_CONFIG_S *)&struVbConf);
  if (HI_SUCCESS != ret) {
    printf("HI_MPI_VB_SetConf failed!");
    return HI_FAILURE;
  }

  ret = HI_MPI_VB_Init();
  if (HI_SUCCESS != ret) {
    printf("HI_MPI_VB_Init failed!");
    return HI_FAILURE;
  }

  ret = HI_MPI_SYS_Init();
  if (HI_SUCCESS != ret) {
    printf("HI_MPI_SYS_Init failed!");
    return HI_FAILURE;
  }

  return HI_SUCCESS;
}

int SvpSysExit(void) {
  HI_S32 ret = HI_SUCCESS;

  ret = HI_MPI_SYS_Exit();
  if (HI_SUCCESS != ret) {
    printf("HI_MPI_SYS_Exit failed!");
    return HI_FAILURE;
  }

  ret = HI_MPI_VB_Exit();
  if (HI_SUCCESS != ret) {
    printf("HI_MPI_VB_Exit failed!");
    return HI_FAILURE;
  }

  return HI_SUCCESS;
}
