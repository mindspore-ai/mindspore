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
#include "src/nnie_common.h"
#include "include/mpi_nnie.h"
#include "include/hi_type.h"
#include "include/errorcode.h"
#include "src/nnie_print.h"
#include "src/nnie_memory.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace nnie {
constexpr int kSleepUs = 100;
constexpr int kCompressionWidth = 2;
static void NnieParamRelease(NnieParam *nnie_param) {
  if (nnie_param == nullptr) {
    return;
  }

  if (nnie_param->task_buf_.u64PhyAddr != 0 && nnie_param->task_buf_.u64VirAddr != 0) {
    NNIE_MEM_FREE(nnie_param->task_buf_.u64PhyAddr, nnie_param->task_buf_.u64VirAddr);
    nnie_param->task_buf_.u64PhyAddr = 0;
    nnie_param->task_buf_.u64VirAddr = 0;
  }

  if (nnie_param->step_buf_.u64PhyAddr != 0 && nnie_param->step_buf_.u64VirAddr != 0) {
    NNIE_MEM_FREE(nnie_param->step_buf_.u64PhyAddr, nnie_param->step_buf_.u64VirAddr);
    nnie_param->step_buf_.u64PhyAddr = 0;
    nnie_param->step_buf_.u64VirAddr = 0;
  }
}

bool CheckNnieInnerNode(const HI_CHAR *name, NnieParam *nnie_param) {
  for (HI_U32 i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    for (HI_U32 j = 0; j < nnie_param->model_->astSeg[i].u16DstNum; j++)
      if (strncmp(name, nnie_param->model_->astSeg[i].astDstNode[j].szName, SVP_NNIE_NODE_NAME_LEN) == 0) {
        nnie_param->mem_cfg_.seg_[i].dst_node_[j] = true;
        return true;
      }
  }
  return false;
}

bool ConnectNnieInnerNode(const HI_CHAR *name, NnieParam *nnie_param, SVP_SRC_BLOB_S *blob) {
  for (HI_U32 i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    for (HI_U32 j = 0; j < nnie_param->model_->astSeg[i].u16DstNum; j++)
      if (strncmp(name, nnie_param->model_->astSeg[i].astDstNode[j].szName, SVP_NNIE_NODE_NAME_LEN) == 0) {
        blob->u64PhyAddr = nnie_param->seg_data_[i].dst_[j].u64PhyAddr;
        blob->u64VirAddr = nnie_param->seg_data_[i].dst_[j].u64VirAddr;
        return true;
      }
  }
  return false;
}

static void FillForwardInfo(NnieCfg *nnie_cfg, NnieParam *nnie_param) {
  HI_U32 i, j;
  HI_U32 num;
  memset(&nnie_param->mem_cfg_, false, sizeof(NNIEMemCfg));
  for (i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    if (SVP_NNIE_NET_TYPE_ROI == nnie_param->model_->astSeg[i].enNetType) {
      nnie_param->forward_with_bbox_ctrl_[i].enNnieId = nnie_cfg->nnie_core_id_[i];
      nnie_param->forward_with_bbox_ctrl_[i].u32SrcNum = nnie_param->model_->astSeg[i].u16SrcNum;
      nnie_param->forward_with_bbox_ctrl_[i].u32DstNum = nnie_param->model_->astSeg[i].u16DstNum;
      nnie_param->forward_with_bbox_ctrl_[i].u32ProposalNum = 1;
      nnie_param->forward_with_bbox_ctrl_[i].u32NetSegId = i;
    } else if (SVP_NNIE_NET_TYPE_CNN == nnie_param->model_->astSeg[i].enNetType ||
               SVP_NNIE_NET_TYPE_RECURRENT == nnie_param->model_->astSeg[i].enNetType) {
      nnie_param->forward_ctrl_[i].enNnieId = nnie_cfg->nnie_core_id_[i];
      nnie_param->forward_ctrl_[i].u32SrcNum = nnie_param->model_->astSeg[i].u16SrcNum;
      nnie_param->forward_ctrl_[i].u32DstNum = nnie_param->model_->astSeg[i].u16DstNum;
      nnie_param->forward_ctrl_[i].u32NetSegId = i;
    }

    for (j = 0; j < nnie_param->model_->astSeg[i].u16SrcNum; j++) {
      if (i > 0) {
        if (CheckNnieInnerNode(nnie_param->model_->astSeg[i].astSrcNode[j].szName, nnie_param)) {
          nnie_param->mem_cfg_.seg_[i].src_node_[j] = true;
        }
      }

      if (SVP_BLOB_TYPE_SEQ_S32 == nnie_param->model_->astSeg[i].astSrcNode[j].enType) {
        nnie_param->seg_data_[i].src_[j].enType = nnie_param->model_->astSeg[i].astSrcNode[j].enType;
        nnie_param->seg_data_[i].src_[j].unShape.stSeq.u32Dim =
          nnie_param->model_->astSeg[i].astSrcNode[j].unShape.u32Dim;
        nnie_param->seg_data_[i].src_[j].u32Num = nnie_cfg->max_input_num_;
        nnie_param->seg_data_[i].src_[j].unShape.stSeq.u64VirAddrStep =
          nnie_cfg->step_vir_addr_[i * NNIE_EACH_SEG_STEP_ADDR_NUM];
      } else {
        nnie_param->seg_data_[i].src_[j].enType = nnie_param->model_->astSeg[i].astSrcNode[j].enType;
        nnie_param->seg_data_[i].src_[j].unShape.stWhc.u32Chn =
          nnie_param->model_->astSeg[i].astSrcNode[j].unShape.stWhc.u32Chn;
        nnie_param->seg_data_[i].src_[j].unShape.stWhc.u32Height =
          nnie_param->model_->astSeg[i].astSrcNode[j].unShape.stWhc.u32Height;
        nnie_param->seg_data_[i].src_[j].unShape.stWhc.u32Width =
          nnie_param->model_->astSeg[i].astSrcNode[j].unShape.stWhc.u32Width;
        nnie_param->seg_data_[i].src_[j].u32Num = nnie_cfg->max_input_num_;
      }
    }

    if (SVP_NNIE_NET_TYPE_ROI == nnie_param->model_->astSeg[i].enNetType) {
      num = nnie_cfg->max_roi_num_ * nnie_cfg->max_input_num_;
    } else {
      num = nnie_cfg->max_input_num_;
    }

    for (j = 0; j < nnie_param->model_->astSeg[i].u16DstNum; j++) {
      if (SVP_BLOB_TYPE_SEQ_S32 == nnie_param->model_->astSeg[i].astDstNode[j].enType) {
        nnie_param->seg_data_[i].dst_[j].enType = nnie_param->model_->astSeg[i].astDstNode[j].enType;
        nnie_param->seg_data_[i].dst_[j].unShape.stSeq.u32Dim =
          nnie_param->model_->astSeg[i].astDstNode[j].unShape.u32Dim;
        nnie_param->seg_data_[i].dst_[j].u32Num = num;
        nnie_param->seg_data_[i].dst_[j].unShape.stSeq.u64VirAddrStep =
          nnie_cfg->step_vir_addr_[i * NNIE_EACH_SEG_STEP_ADDR_NUM + 1];
      } else {
        nnie_param->seg_data_[i].dst_[j].enType = nnie_param->model_->astSeg[i].astDstNode[j].enType;
        nnie_param->seg_data_[i].dst_[j].unShape.stWhc.u32Chn =
          nnie_param->model_->astSeg[i].astDstNode[j].unShape.stWhc.u32Chn;
        nnie_param->seg_data_[i].dst_[j].unShape.stWhc.u32Height =
          nnie_param->model_->astSeg[i].astDstNode[j].unShape.stWhc.u32Height;
        nnie_param->seg_data_[i].dst_[j].unShape.stWhc.u32Width =
          nnie_param->model_->astSeg[i].astDstNode[j].unShape.stWhc.u32Width;
        nnie_param->seg_data_[i].dst_[j].u32Num = num;
      }
    }
  }
}

static void GetBlobMemSize(SVP_NNIE_NODE_S nnie_node[], HI_U32 node_num, HI_U32 total_step, SVP_BLOB_S blob[],
                           HI_U32 align32, HI_U32 *total_size, HI_U32 blob_size[], bool malloc_allow,
                           bool *mem_alloc = nullptr) {
  HI_U32 i = 0;
  HI_U32 size;
  HI_U32 stride;

  for (i = 0; i < node_num; i++) {
    if (SVP_BLOB_TYPE_S32 == nnie_node[i].enType || SVP_BLOB_TYPE_VEC_S32 == nnie_node[i].enType ||
        SVP_BLOB_TYPE_SEQ_S32 == nnie_node[i].enType) {
      size = sizeof(HI_U32);
    } else {
      size = sizeof(HI_U8);
    }
    if (SVP_BLOB_TYPE_SEQ_S32 == nnie_node[i].enType) {
      if (NNIE_ALIGN_16 == align32) {
        stride = NNIE_ALIGN16(nnie_node[i].unShape.u32Dim * size);
      } else {
        stride = NNIE_ALIGN32(nnie_node[i].unShape.u32Dim * size);
      }
      blob_size[i] = total_step * stride;
    } else {
      if (NNIE_ALIGN_16 == align32) {
        stride = NNIE_ALIGN16(nnie_node[i].unShape.stWhc.u32Width * size);
      } else {
        stride = NNIE_ALIGN32(nnie_node[i].unShape.stWhc.u32Width * size);
      }
      blob_size[i] = blob[i].u32Num * stride * nnie_node[i].unShape.stWhc.u32Height * nnie_node[i].unShape.stWhc.u32Chn;
    }
    if (mem_alloc != nullptr) {
      if (mem_alloc[i]) {
        blob_size[i] = 0;
      }
    }
    if (malloc_allow) {
      *total_size += blob_size[i];
    }
    blob[i].u32Stride = stride;
  }
}

static int GetTaskAndBlobBufSize(NnieCfg *nnie_cfg, NnieParam *nnie_param, HI_U32 *total_task_buf_size,
                                 HI_U32 *tmp_buf_size, NnieBlobSize blob_size[], HI_U32 blob_size_len,
                                 HI_U32 *total_size) {
  HI_S32 ret = HI_SUCCESS;
  HI_U32 i, j;
  HI_U32 total_step = 0;

  ret = HI_MPI_SVP_NNIE_GetTskBufSize(nnie_cfg->max_input_num_, nnie_cfg->max_roi_num_, nnie_param->model_,
                                      nnie_param->task_buf_size_, nnie_param->model_->u32NetSegNum);
  if (HI_SUCCESS != ret) {
    LOGE("HI_MPI_SVP_NNIE_GetTskBufSize");
    return RET_ERROR;
  }

  *total_task_buf_size = 0;
  for (i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    *total_task_buf_size += nnie_param->task_buf_size_[i];
  }

  *tmp_buf_size = nnie_param->model_->u32TmpBufSize;
  *total_size += *total_task_buf_size + *tmp_buf_size;

  for (i = 0; i < nnie_param->model_->u32NetSegNum && i < blob_size_len; i++) {
    if (SVP_NNIE_NET_TYPE_RECURRENT == nnie_param->model_->astSeg[i].enNetType) {
      for (j = 0; j < nnie_param->seg_data_[i].src_[0].u32Num; j++) {
        total_step += *(reinterpret_cast<HI_S32 *>(
                          static_cast<HI_UL>(nnie_param->seg_data_[i].src_[0].unShape.stSeq.u64VirAddrStep)) +
                        j);
      }
    }
    bool malloc_allow = (!nnie_cfg->pass_align16_io_) || i != 0;
    GetBlobMemSize(&(nnie_param->model_->astSeg[i].astSrcNode[0]), nnie_param->model_->astSeg[i].u16SrcNum, total_step,
                   &(nnie_param->seg_data_[i].src_[0]), NNIE_ALIGN_16, total_size, &(blob_size[i].src_size_[0]),
                   malloc_allow, &(nnie_param->mem_cfg_.seg_[i].src_node_[0]));

    malloc_allow = (!nnie_cfg->pass_align16_io_) || (i + 1) != nnie_param->model_->u32NetSegNum;
    GetBlobMemSize(&(nnie_param->model_->astSeg[i].astDstNode[0]), nnie_param->model_->astSeg[i].u16DstNum, total_step,
                   &(nnie_param->seg_data_[i].dst_[0]), NNIE_ALIGN_16, total_size, &(blob_size[i].dst_size_[0]),
                   malloc_allow);
  }
  return RET_OK;
}

static int NnieSetBlobAddr(HI_U64 *phy_addr, HI_U8 **vir_addr, NnieParam *nnie_param, NnieBlobSize *blob_size,
                           bool pass_align16_io) {
  HI_U32 i, j;
  for (i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    if ((!pass_align16_io) || i != 0) {
      for (j = 0; j < nnie_param->model_->astSeg[i].u16SrcNum; j++) {
        if (j != 0) {
          *phy_addr += blob_size[i].src_size_[j - 1];
          *vir_addr += blob_size[i].src_size_[j - 1];
        }
        if (nnie_param->mem_cfg_.seg_[i].src_node_[j]) {
          if (!ConnectNnieInnerNode(nnie_param->model_->astSeg[i].astSrcNode[j].szName, nnie_param,
                                    &(nnie_param->seg_data_[i].src_[j]))) {
            LOGE("ConnectNnieInnerNode failed! ");
            return RET_ERROR;
          }
        } else {
          nnie_param->seg_data_[i].src_[j].u64PhyAddr = *phy_addr;
          nnie_param->seg_data_[i].src_[j].u64VirAddr = (HI_U64)(HI_UL)*vir_addr;
        }
      }
      *phy_addr += blob_size[i].src_size_[j - 1];
      *vir_addr += blob_size[i].src_size_[j - 1];
    } else {
      for (j = 0; j < nnie_param->model_->astSeg[i].u16SrcNum; j++) {
        nnie_param->seg_data_[i].src_[j].u64PhyAddr = 0;
        nnie_param->seg_data_[i].src_[j].u64VirAddr = 0;
      }
    }
    if ((!pass_align16_io) || (i + 1) != nnie_param->model_->u32NetSegNum) {
      for (j = 0; j < nnie_param->model_->astSeg[i].u16DstNum; j++) {
        if (j != 0) {
          *phy_addr += blob_size[i].dst_size_[j - 1];
          *vir_addr += blob_size[i].dst_size_[j - 1];
        }
        nnie_param->seg_data_[i].dst_[j].u64PhyAddr = *phy_addr;
        nnie_param->seg_data_[i].dst_[j].u64VirAddr = (HI_U64)(HI_UL)*vir_addr;
      }
      *phy_addr += blob_size[i].dst_size_[j - 1];
      *vir_addr += blob_size[i].dst_size_[j - 1];
    } else {
      for (j = 0; j < nnie_param->model_->astSeg[i].u16SrcNum; j++) {
        nnie_param->seg_data_[i].dst_[j].u64PhyAddr = 0;
        nnie_param->seg_data_[i].dst_[j].u64VirAddr = 0;
      }
    }
  }
  return RET_OK;
}

static int NnieParamInit(NnieCfg *nnie_cfg, NnieParam *nnie_param) {
  HI_U32 i;
  HI_U32 total_size = 0, total_task_buf_size = 0, tmp_buf_size_ = 0;
  HI_S32 ret = HI_SUCCESS;
  HI_U32 off_set = 0;
  HI_U64 phy_addr = 0;
  HI_U8 *vir_addr = nullptr;
  NnieBlobSize blob_size[SVP_NNIE_MAX_NET_SEG_NUM] = {0};

  FillForwardInfo(nnie_cfg, nnie_param);

  HI_U32 blob_size_len = sizeof(blob_size) / sizeof(blob_size[0]);
  ret = GetTaskAndBlobBufSize(nnie_cfg, nnie_param, &total_task_buf_size, &tmp_buf_size_, blob_size, blob_size_len,
                              &total_size);
  if (HI_SUCCESS != ret) {
    LOGE("Error,Malloc memory failed! ");
    return RET_ERROR;
  }
  bool has_roi = false;
  for (i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    if (SVP_NNIE_NET_TYPE_ROI == nnie_param->model_->astSeg[i].enNetType) {
      has_roi = true;
    }
  }
  if (has_roi) {
    nnie_param->rpn_bbox_.enType = SVP_BLOB_TYPE_S32;
    nnie_param->rpn_bbox_.unShape.stWhc.u32Chn = 1;
    nnie_param->rpn_bbox_.unShape.stWhc.u32Height = nnie_cfg->max_roi_num_;
    nnie_param->rpn_bbox_.unShape.stWhc.u32Width = NNIE_COORDI_NUM;
    nnie_param->rpn_bbox_.u32Stride = NNIE_ALIGN16(NNIE_COORDI_NUM * sizeof(HI_U32));
    nnie_param->rpn_bbox_.u32Num = nnie_cfg->max_input_num_;
    total_size +=
      nnie_param->rpn_bbox_.u32Num * nnie_param->rpn_bbox_.unShape.stWhc.u32Height * nnie_param->rpn_bbox_.u32Stride;
  }

  ret = NnieMemMallocCached(std::string("NNIE_NNIE_TASK").data(), nullptr, reinterpret_cast<HI_U64 *>(&phy_addr),
                            reinterpret_cast<void **>(&vir_addr), total_size);
  if (HI_SUCCESS != ret) {
    LOGE("Error,Malloc memory failed! ");
    return RET_ERROR;
  }
  memset(vir_addr, 0, total_size);
  NnieMemFlushCache(phy_addr, reinterpret_cast<void *>(vir_addr), total_size);

  nnie_param->task_buf_.u32Size = total_task_buf_size;
  nnie_param->task_buf_.u64PhyAddr = phy_addr;
  nnie_param->task_buf_.u64VirAddr = (HI_U64)(HI_UL)vir_addr;

  nnie_param->tmp_buf_.u32Size = tmp_buf_size_;
  nnie_param->tmp_buf_.u64PhyAddr = phy_addr + total_task_buf_size;
  nnie_param->tmp_buf_.u64VirAddr = (HI_U64)(HI_UL)vir_addr + total_task_buf_size;

  for (i = 0; i < nnie_param->model_->u32NetSegNum; i++) {
    if (SVP_NNIE_NET_TYPE_ROI == nnie_param->model_->astSeg[i].enNetType) {
      nnie_param->forward_with_bbox_ctrl_[i].stTmpBuf = nnie_param->tmp_buf_;
      nnie_param->forward_with_bbox_ctrl_[i].stTskBuf.u64PhyAddr = nnie_param->task_buf_.u64PhyAddr + off_set;
      nnie_param->forward_with_bbox_ctrl_[i].stTskBuf.u64VirAddr = nnie_param->task_buf_.u64VirAddr + off_set;
      nnie_param->forward_with_bbox_ctrl_[i].stTskBuf.u32Size = nnie_param->task_buf_size_[i];
    } else if (SVP_NNIE_NET_TYPE_CNN == nnie_param->model_->astSeg[i].enNetType ||
               SVP_NNIE_NET_TYPE_RECURRENT == nnie_param->model_->astSeg[i].enNetType) {
      nnie_param->forward_ctrl_[i].stTmpBuf = nnie_param->tmp_buf_;
      nnie_param->forward_ctrl_[i].stTskBuf.u64PhyAddr = nnie_param->task_buf_.u64PhyAddr + off_set;
      nnie_param->forward_ctrl_[i].stTskBuf.u64VirAddr = nnie_param->task_buf_.u64VirAddr + off_set;
      nnie_param->forward_ctrl_[i].stTskBuf.u32Size = nnie_param->task_buf_size_[i];
    }
    off_set += nnie_param->task_buf_size_[i];
  }

  phy_addr = phy_addr + total_task_buf_size + tmp_buf_size_;
  vir_addr = vir_addr + total_task_buf_size + tmp_buf_size_;
  if (NnieSetBlobAddr(&phy_addr, &vir_addr, nnie_param, blob_size, nnie_cfg->pass_align16_io_) != RET_OK) {
    LOGE("SetBlobAddr failed!");
    return RET_ERROR;
  }
  if (has_roi) {
    nnie_param->rpn_bbox_.u64PhyAddr = phy_addr;
    nnie_param->rpn_bbox_.u64VirAddr = (HI_U64)((HI_UL)vir_addr);
  }
  return RET_OK;
}

int NnieLoadModel(char *model_buf, int size, NnieModel *nnie_model) {
  HI_S32 ret = HI_INVALID_VALUE;
  HI_U64 phy_addr = 0;
  HI_U8 *vir_addr = nullptr;
  ret = NnieMemMalloc(std::string("NNIE_NNIE_MODEL").data(), nullptr, reinterpret_cast<HI_U64 *>(&phy_addr),
                      reinterpret_cast<void **>(&vir_addr), size);
  if (HI_SUCCESS != ret) {
    LOGE("Error,Malloc memory failed! ");
    return RET_ERROR;
  }
  nnie_model->model_buf_.u32Size = (HI_U32)size;
  nnie_model->model_buf_.u64PhyAddr = phy_addr;
  nnie_model->model_buf_.u64VirAddr = (HI_U64)(HI_UL)vir_addr;
  memcpy(vir_addr, model_buf, size);
  ret = HI_MPI_SVP_NNIE_LoadModel(&nnie_model->model_buf_, &nnie_model->model_);
  if (HI_SUCCESS != ret) {
    NNIE_MEM_FREE(nnie_model->model_buf_.u64PhyAddr, nnie_model->model_buf_.u64VirAddr);
    nnie_model->model_buf_.u32Size = 0;
    LOGE("HI_MPI_SVP_NNIE_LoadModel failed!");
    return RET_ERROR;
  }
  return RET_OK;
}

static void NnieUnloadModel(NnieModel *nnie_model) {
  if (nnie_model == nullptr) {
    return;
  }

  if (nnie_model->model_buf_.u64PhyAddr != 0 && nnie_model->model_buf_.u64VirAddr != 0) {
    NNIE_MEM_FREE(nnie_model->model_buf_.u64PhyAddr, nnie_model->model_buf_.u64VirAddr);
    nnie_model->model_buf_.u64PhyAddr = 0;
    nnie_model->model_buf_.u64VirAddr = 0;
  }
}

static int NnieForward(NnieParam *nnie_param, NnieDataIndex *input_data_idx, HI_BOOL instant) {
  HI_S32 ret = HI_SUCCESS;
  HI_U32 i, j;
  HI_BOOL finish = HI_FALSE;
  SVP_NNIE_HANDLE svp_nnie_handle = 0;
  HI_U32 total_step_num = 0;
  SVP_NNIE_FORWARD_CTRL_S *forward_handle = &nnie_param->forward_ctrl_[input_data_idx->seg_idx_];
  NnieSegData *seg_data = &nnie_param->seg_data_[input_data_idx->seg_idx_];

  NnieMemFlushCache(forward_handle->stTskBuf.u64PhyAddr,
                    NNIE_CONVERT_64BIT_ADDR(HI_VOID, forward_handle->stTskBuf.u64VirAddr),
                    forward_handle->stTskBuf.u32Size);

  for (i = 0; i < forward_handle->u32DstNum; i++) {
    if (SVP_BLOB_TYPE_SEQ_S32 == seg_data->dst_[i].enType) {
      for (j = 0; j < seg_data->dst_[i].u32Num; j++) {
        total_step_num += *(NNIE_CONVERT_64BIT_ADDR(HI_U32, seg_data->dst_[i].unShape.stSeq.u64VirAddrStep) + j);
      }
      NnieMemFlushCache(seg_data->dst_[i].u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, seg_data->dst_[i].u64VirAddr),
                        total_step_num * seg_data->dst_[i].u32Stride);
    } else {
      NnieMemFlushCache(seg_data->dst_[i].u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, seg_data->dst_[i].u64VirAddr),
                        seg_data->dst_[i].u32Num * seg_data->dst_[i].unShape.stWhc.u32Chn *
                          seg_data->dst_[i].unShape.stWhc.u32Height * seg_data->dst_[i].u32Stride);
    }
  }

  ret = HI_MPI_SVP_NNIE_Forward(&svp_nnie_handle, seg_data->src_, nnie_param->model_, seg_data->dst_, forward_handle,
                                instant);
  if (HI_SUCCESS != ret) {
    LOGE("HI_MPI_SVP_NNIE_Forward failed!");
    return RET_ERROR;
  }

  if (instant) {
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT ==
           (ret = HI_MPI_SVP_NNIE_Query(forward_handle->enNnieId, svp_nnie_handle, &finish, HI_TRUE))) {
      usleep(kSleepUs);
    }
  }

  total_step_num = 0;
  for (i = 0; i < forward_handle->u32DstNum; i++) {
    if (SVP_BLOB_TYPE_SEQ_S32 == seg_data->dst_[i].enType) {
      for (j = 0; j < seg_data->dst_[i].u32Num; j++) {
        total_step_num += *(NNIE_CONVERT_64BIT_ADDR(HI_U32, seg_data->dst_[i].unShape.stSeq.u64VirAddrStep) + j);
      }
      NnieMemFlushCache(seg_data->dst_[i].u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, seg_data->dst_[i].u64VirAddr),
                        total_step_num * seg_data->dst_[i].u32Stride);
    } else {
      NnieMemFlushCache(seg_data->dst_[i].u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, seg_data->dst_[i].u64VirAddr),
                        seg_data->dst_[i].u32Num * seg_data->dst_[i].unShape.stWhc.u32Chn *
                          seg_data->dst_[i].unShape.stWhc.u32Height * seg_data->dst_[i].u32Stride);
    }
  }

  return RET_OK;
}

static HI_S32 NNIE_ForwardWithBbox(NnieParam *pstNnieParam, NnieDataIndex *pstInputDataIdx, SVP_SRC_BLOB_S astBbox[],
                                   HI_BOOL bInstant) {
  HI_S32 ret = HI_SUCCESS;
  HI_BOOL finish = HI_FALSE;
  SVP_NNIE_HANDLE svp_nnie_handle = 0;
  HI_U32 total_step_num = 0;
  HI_U32 i, j;

  NnieMemFlushCache(pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].stTskBuf.u64PhyAddr,
                    NNIE_CONVERT_64BIT_ADDR(
                      HI_VOID, pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].stTskBuf.u64VirAddr),
                    pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].stTskBuf.u32Size);

  for (i = 0; i < pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].u32DstNum; i++) {
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].enType) {
      for (j = 0; j < pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Num; j++) {
        total_step_num +=
          *(NNIE_CONVERT_64BIT_ADDR(
              HI_U32, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stSeq.u64VirAddrStep) +
            j);
      }
      NnieMemFlushCache(
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64PhyAddr,
        NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64VirAddr),
        total_step_num * pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Stride);
    } else {
      NnieMemFlushCache(
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64PhyAddr,
        NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64VirAddr),
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Num *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stWhc.u32Chn *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stWhc.u32Height *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Stride);
    }
  }

  ret =
    HI_MPI_SVP_NNIE_ForwardWithBbox(&svp_nnie_handle, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].src_, astBbox,
                                    pstNnieParam->model_, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_,
                                    &pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_], bInstant);
  if (HI_SUCCESS != ret) {
    LOGE("HI_MPI_SVP_NNIE_ForwardWithBbox failed!");
    return RET_ERROR;
  }

  if (bInstant) {
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT ==
           (ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].enNnieId,
                                        svp_nnie_handle, &finish, HI_TRUE))) {
      usleep(kSleepUs);
      LOGE("HI_MPI_SVP_NNIE_Query Query timeout!");
    }
  }

  total_step_num = 0;

  for (i = 0; i < pstNnieParam->forward_with_bbox_ctrl_[pstInputDataIdx->seg_idx_].u32DstNum; i++) {
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].enType) {
      for (j = 0; j < pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Num; j++) {
        total_step_num +=
          *(NNIE_CONVERT_64BIT_ADDR(
              HI_U32, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stSeq.u64VirAddrStep) +
            j);
      }
      NnieMemFlushCache(
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64PhyAddr,
        NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64VirAddr),
        total_step_num * pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Stride);
    } else {
      NnieMemFlushCache(
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64PhyAddr,
        NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u64VirAddr),
        pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Num *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stWhc.u32Chn *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].unShape.stWhc.u32Height *
          pstNnieParam->seg_data_[pstInputDataIdx->seg_idx_].dst_[i].u32Stride);
    }
  }

  return ret;
}

int FillByUnsignedChar(HI_U32 input_size, HI_U32 num, HI_U32 width, HI_U32 stride, HI_U8 *src, HI_U8 *dst) {
  HI_U32 i, j;
  if (input_size != num * width) {
    LOGE("input size error:%d <-> %d.", input_size, num * width);
    return RET_ERROR;
  }
  for (i = 0; i < num; i++) {
    for (j = 0; j < width; j++) {
      dst[j] = src[j];
    }
    dst += stride;
    src += width;
  }
  return RET_OK;
}

int FillByFloat(HI_U32 input_size, HI_U32 num, HI_U32 width, HI_U32 stride, HI_FLOAT *src, HI_S32 *dst, HI_U8 *dst_u8) {
  HI_U32 i, j;
  if (input_size != num * width) {
    LOGE("input size error:%d <-> %d.", input_size, num * width);
    return RET_ERROR;
  }
  for (i = 0; i < num; i++) {
    for (j = 0; j < width; j++) {
      dst[j] = (src[j] * NNIE_QUANT_BASE);
    }
    dst_u8 += stride;
    dst = reinterpret_cast<HI_S32 *>(dst_u8);
    src += width;
  }
  return RET_OK;
}

static int NnieFillSrcDataSeq(NnieCfg *nnie_cfg, SVP_SRC_BLOB_S *blob, HI_U32 input_size) {
  HI_U32 *step_addr_u32 = NNIE_CONVERT_64BIT_ADDR(HI_U32, blob->unShape.stSeq.u64VirAddrStep);
  HI_U32 dim = blob->unShape.stSeq.u32Dim;
  HI_U32 stride = blob->u32Stride;
  HI_U32 i;
  HI_U32 j;
  HI_U32 n;
  HI_U32 total_step_num = 0;
  HI_U8 *input_addr_u8 = NNIE_CONVERT_64BIT_ADDR(HI_U8, blob->u64VirAddr);
  HI_S32 *input_addr_s32 = NNIE_CONVERT_64BIT_ADDR(HI_S32, blob->u64VirAddr);
  HI_FLOAT *float_src_data = reinterpret_cast<float *>(nnie_cfg->data_ptr_);

  for (n = 0; n < blob->u32Num; n++) {
    total_step_num += *(step_addr_u32 + n);
  }

  if (input_size != total_step_num * dim) {
    LOGE("input size error:%d <-> %d.", input_size, total_step_num * dim);
    return RET_ERROR;
  }
  for (n = 0; n < blob->u32Num; n++) {
    for (i = 0; i < *(step_addr_u32 + n); i++) {
      for (j = 0; j < dim; j++) {
        input_addr_s32[j] = (float_src_data[j] * NNIE_QUANT_BASE);
      }
      input_addr_u8 += stride;
      input_addr_s32 = reinterpret_cast<HI_S32 *>(input_addr_u8);
      float_src_data += dim;
    }
  }
  NnieMemFlushCache(blob->u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, blob->u64VirAddr), total_step_num * stride);
  return RET_OK;
}

HI_U32 GetBlobSize(const SVP_SRC_BLOB_S &blob) {
  if (SVP_BLOB_TYPE_SEQ_S32 == blob.enType) {
    HI_U32 stride = blob.u32Stride;
    HI_U32 total_step_num = 0;
    HI_U32 *step_addr_u32 = NNIE_CONVERT_64BIT_ADDR(HI_U32, blob.unShape.stSeq.u64VirAddrStep);
    size_t n;
    for (n = 0; n < blob.u32Num; n++) {
      total_step_num += *(step_addr_u32 + n);
    }
    return total_step_num * stride;
  }

  HI_U32 stride = blob.u32Stride;
  HI_U32 height = blob.unShape.stWhc.u32Height;
  HI_U32 channel = blob.unShape.stWhc.u32Chn;
  if (SVP_BLOB_TYPE_YVU420SP == blob.enType) {
    return blob.u32Num * static_cast<HI_U32>(channel * height / kCompressionWidth) * stride;
  } else if (SVP_BLOB_TYPE_YVU422SP == blob.enType) {
    return blob.u32Num * height * kCompressionWidth * stride;
  } else {
    return blob.u32Num * channel * height * stride;
  }
}

static int NnieFillSrcData(NnieCfg *nnie_cfg, NnieParam *nnie_param, NnieDataIndex *input_data_idx, HI_U32 input_size) {
  HI_U32 i;
  HI_U32 ret;
  SVP_SRC_BLOB_S *blob = &nnie_param->seg_data_[input_data_idx->seg_idx_].src_[input_data_idx->node_idx_];

  if (SVP_BLOB_TYPE_SEQ_S32 == blob->enType) {
    return NnieFillSrcDataSeq(nnie_cfg, blob, input_size);
  } else {
    HI_U8 *input_addr_u8 = NNIE_CONVERT_64BIT_ADDR(HI_U8, blob->u64VirAddr);
    HI_S32 *input_addr_s32 = NNIE_CONVERT_64BIT_ADDR(HI_S32, blob->u64VirAddr);
    HI_FLOAT *float_src_data = reinterpret_cast<float *>(nnie_cfg->data_ptr_);
    HI_U8 *u8_src_data = reinterpret_cast<unsigned char *>(nnie_cfg->data_ptr_);
    HI_U32 height = blob->unShape.stWhc.u32Height;
    HI_U32 width = blob->unShape.stWhc.u32Width;
    HI_U32 channel = blob->unShape.stWhc.u32Chn;
    HI_U32 stride = blob->u32Stride;
    if (input_addr_u8 == u8_src_data) {
      if (blob->enType == SVP_BLOB_TYPE_S32) {
        for (i = 0; i < input_size; i++) {
          input_addr_s32[i] = float_src_data[i] * NNIE_QUANT_BASE;
        }
      } else {
        LOGI("\ninput no memcpy");
      }
    } else {
      if (SVP_BLOB_TYPE_YVU420SP == blob->enType) {
        ret = FillByUnsignedChar(input_size, blob->u32Num * static_cast<HI_U32>(channel * height / 2), width, stride,
                                 u8_src_data, input_addr_u8);
      } else if (SVP_BLOB_TYPE_YVU422SP == blob->enType) {
        ret = FillByUnsignedChar(input_size, blob->u32Num * height * 2, width, stride, u8_src_data, input_addr_u8);
      } else {
        if (SVP_BLOB_TYPE_U8 == blob->enType) {
          ret =
            FillByUnsignedChar(input_size, blob->u32Num * channel * height, width, stride, u8_src_data, input_addr_u8);
        } else {
          ret = FillByFloat(input_size, blob->u32Num * channel * height, width, stride, float_src_data, input_addr_s32,
                            input_addr_u8);
        }
      }
      if (ret != RET_OK) {
        return ret;
      }
    }
    NnieMemFlushCache(blob->u64PhyAddr, NNIE_CONVERT_64BIT_ADDR(HI_VOID, blob->u64VirAddr),
                      blob->u32Num * channel * height * stride);
  }

  return RET_OK;
}

static int NnieGetDstDataSEQ(SVP_SRC_BLOB_S *blob, HI_U32 input_num, NnieDataIndex *input_data_idx,
                             HI_FLOAT *float_dst_data) {
  HI_U32 i, j, n;
  HI_U32 dim = blob->unShape.stSeq.u32Dim;
  HI_U32 stride = blob->u32Stride;
  HI_U32 *step_addr_u32 = NNIE_CONVERT_64BIT_ADDR(HI_U32, blob->unShape.stSeq.u64VirAddrStep);
  HI_U32 total_step_num = 0;
  HI_U8 *output_addr_u8 = NNIE_CONVERT_64BIT_ADDR(HI_U8, blob->u64VirAddr);
  HI_S32 *output_addr_s32 = NNIE_CONVERT_64BIT_ADDR(HI_S32, blob->u64VirAddr);

  for (n = 0; n < blob->u32Num; n++) {
    total_step_num += *(step_addr_u32 + n);
  }
  if (input_num != total_step_num * dim) {
    LOGE("input shape");
    return RET_ERROR;
  }
  if (input_data_idx->seg_idx_ == input_data_idx->max_seg_id_) {
    for (n = 0; n < blob->u32Num; n++) {
      for (i = 0; i < *(step_addr_u32 + n); i++) {
        memcpy(float_dst_data, output_addr_u8, dim * sizeof(float));
        float_dst_data += dim;
        output_addr_u8 += stride;
      }
    }
  } else {
    for (n = 0; n < blob->u32Num; n++) {
      for (i = 0; i < *(step_addr_u32 + n); i++) {
        for (j = 0; j < dim; j++) {
          float_dst_data[j] = (HI_FLOAT)output_addr_s32[j] / NNIE_QUANT_BASE;
        }
        output_addr_u8 += stride;
        output_addr_s32 = reinterpret_cast<HI_S32 *>(output_addr_u8);
        float_dst_data += dim;
      }
    }
  }
  return RET_OK;
}
static int NnieGetDstData(NnieCfg *nnie_cfg, NnieParam *nnie_param, NnieDataIndex *input_data_idx, HI_U32 input_num) {
  SVP_SRC_BLOB_S *blob = &nnie_param->seg_data_[input_data_idx->seg_idx_ - 1].dst_[input_data_idx->node_idx_];
  if (SVP_BLOB_TYPE_U8 <= blob->enType && SVP_BLOB_TYPE_YVU422SP >= blob->enType) {
    LOGE("Nnie output type error");
    return RET_ERROR;
  }
  HI_FLOAT *float_dst_data = reinterpret_cast<float *>(nnie_cfg->data_ptr_);
  if (SVP_BLOB_TYPE_SEQ_S32 == blob->enType) {
    if (NnieGetDstDataSEQ(blob, input_num, input_data_idx, float_dst_data) != RET_OK) {
      LOGE("NnieGetDstDataSEQ error.");
      return RET_ERROR;
    }
  } else {
    HI_U8 *output_addr_u8 = NNIE_CONVERT_64BIT_ADDR(HI_U8, blob->u64VirAddr);
    HI_S32 *output_addr_s32 = NNIE_CONVERT_64BIT_ADDR(HI_S32, blob->u64VirAddr);
    if (float_dst_data == reinterpret_cast<float *>(output_addr_s32)) {
      if (input_data_idx->seg_idx_ != input_data_idx->max_seg_id_) {
        for (HI_U32 i = 0; i < input_num; i++) {
          float_dst_data[i] = (HI_FLOAT)output_addr_s32[i] / NNIE_QUANT_BASE;
        }
      } else {
        LOGI("\noutput no memcpy");
      }
    } else {
      HI_U32 height = blob->unShape.stWhc.u32Height;
      HI_U32 width = blob->unShape.stWhc.u32Width;
      HI_U32 channel = blob->unShape.stWhc.u32Chn;
      HI_U32 stride = blob->u32Stride;
      if (input_num != height * channel * width * blob->u32Num) {
        LOGE("output shape diff:%d<->%d.", input_num, height * channel * width * blob->u32Num);
        return RET_ERROR;
      }
      if (input_data_idx->seg_idx_ == input_data_idx->max_seg_id_) {
        if (nnie_cfg->pass_align16_io_) {
          memcpy(float_dst_data, output_addr_u8, blob->u32Num * channel * height * stride);
        } else {
          for (HI_U32 i = 0; i < (blob->u32Num * channel * height); i++) {
            memcpy(float_dst_data, output_addr_u8, width * sizeof(float));
            float_dst_data += width;
            output_addr_u8 += stride;
          }
        }
      } else {
        for (HI_U32 n = 0; n < blob->u32Num; n++) {
          for (HI_U32 i = 0; i < channel * height; i++) {
            for (HI_U32 j = 0; j < width; j++) {
              float_dst_data[j] = (HI_FLOAT)output_addr_s32[j] / NNIE_QUANT_BASE;
            }
            output_addr_u8 += stride;
            output_addr_s32 = reinterpret_cast<HI_S32 *>(output_addr_u8);
            float_dst_data += width;
          }
        }
      }
    }
  }
  return RET_OK;
}

int CheckMsShapeN(NnieRunCfg *nnie_run_cfg, const std::vector<int64_t> &input_shape, const SVP_NNIE_NODE_S &nnie_node) {
  size_t ms_input_size = 1, i;
  for (i = 1; i < input_shape.size(); i++) {
    ms_input_size *= input_shape[i];
  }

  size_t nnie_input_size;
  if (SVP_BLOB_TYPE_SEQ_S32 == nnie_node.enType) {
    if (nnie_run_cfg->cfg_.step_ == 0) {
      LOGE("request time_step set! Please export NNIE_RUNTIME_CONFIG_PATH");
      return RET_ERROR;
    }
    if (ms_input_size != nnie_node.unShape.u32Dim) {
      LOGE("The input data does not meet the required size %d <-> %d.", static_cast<int>(ms_input_size),
           nnie_node.unShape.u32Dim);
      return RET_ERROR;
    }
    if ((input_shape[0] < static_cast<int>(nnie_run_cfg->cfg_.step_)) ||
        (input_shape[0] % nnie_run_cfg->cfg_.step_ != 0)) {
      LOGW("The num value(%d) of input must be an integer multiple of time_step(%d)", static_cast<int>(input_shape[0]),
           nnie_run_cfg->cfg_.step_);
      return RET_ERROR;
    }
    nnie_input_size = nnie_node.unShape.u32Dim * nnie_run_cfg->cfg_.step_;
  } else {
    auto height = nnie_node.unShape.stWhc.u32Height;
    auto width = nnie_node.unShape.stWhc.u32Width;
    auto channel = nnie_node.unShape.stWhc.u32Chn;
    if (SVP_BLOB_TYPE_YVU420SP == nnie_node.enType) {
      nnie_input_size = static_cast<HI_U32>(channel * height / 2) * width;
    } else if (SVP_BLOB_TYPE_YVU422SP == nnie_node.enType) {
      nnie_input_size = height * 2 * width;
    } else {
      nnie_input_size = channel * height * width;
    }
    if (ms_input_size != nnie_input_size) {
      LOGE("The input data does not meet the required size %d <-> %d.", static_cast<int>(ms_input_size),
           static_cast<int>(nnie_input_size));
      return RET_ERROR;
    }
  }
  nnie_run_cfg->cfg_.max_input_num_ = (ms_input_size * input_shape[0]) / nnie_input_size;
  fprintf(stdout, "The input num is %d.", nnie_run_cfg->cfg_.max_input_num_);
  return RET_OK;
}

int NnieCommCreate(NnieRunCfg *nnie_run_cfg, const std::vector<int64_t> &input_shape) {
  HI_U8 *vir_addr = nullptr;
  HI_U32 seg_num;
  HI_U32 off_set;
  HI_U32 total_size;
  HI_U32 i, j;
  HI_S32 ret = HI_SUCCESS;
  NnieModel *model = &nnie_run_cfg->model_;
  NnieParam *param = &nnie_run_cfg->param_;
  NnieCfg *cfg = &nnie_run_cfg->cfg_;
  HI_U32 step = cfg->step_;  // time step

  if (input_shape.size() <= 1) {
    LOGE("input shape size need greater than 1!");
    return RET_ERROR;
  }

  if (CheckMsShapeN(nnie_run_cfg, input_shape, model->model_.astSeg[0].astSrcNode[0]) != RET_OK) {
    return RET_ERROR;
  }

  bool has_roi = false;
  for (i = 0; i < model->model_.u32NetSegNum; i++) {
    if (SVP_NNIE_NET_TYPE_ROI == model->model_.astSeg[i].enNetType) {
      has_roi = true;
    }
  }
  if (has_roi) {
    if (cfg->max_roi_num_ == 0) {
      LOGE("NNIE_RUNTIME_CONFIG_PATH: max_roi_num(0) should greater than 0!");
      return RET_ERROR;
    }
  } else {
    if (cfg->max_roi_num_ != 0) {
      LOGW("NNIE_RUNTIME_CONFIG_PATH: max_roi_num should euqal to 0!");
      cfg->max_roi_num_ = 0;
    }
  }

  if (model->model_.astSeg[0].enNetType == SVP_NNIE_NET_TYPE_RECURRENT) {
    if (step == 0) {
      LOGE("request time_step set! No NNIE_RUNTIME_CONFIG_PATH, please export NNIE_RUNTIME_CONFIG_PATH");
      return RET_ERROR;
    }
    seg_num = model->model_.u32NetSegNum;
    total_size = cfg->max_input_num_ * sizeof(HI_S32) * seg_num * 2;
    ret = NnieMemMalloc(std::string("SVP_NNIE_STEP").data(), nullptr,
                        reinterpret_cast<HI_U64 *>(&param->step_buf_.u64PhyAddr), reinterpret_cast<void **>(&vir_addr),
                        total_size);
    if (HI_SUCCESS != ret) {
      LOGE("Malloc memory failed:");
      return RET_ERROR;
    }

    param->step_buf_.u64VirAddr = (HI_U64)((HI_UL)vir_addr);
    for (i = 0; i < seg_num * NNIE_EACH_SEG_STEP_ADDR_NUM; i++) {
      cfg->step_vir_addr_[i] = param->step_buf_.u64VirAddr + i * cfg->max_input_num_ * sizeof(HI_S32);
    }

    for (i = 0; i < seg_num; i++) {
      off_set = i * NNIE_EACH_SEG_STEP_ADDR_NUM;
      for (j = 0; j < cfg->max_input_num_; j++) {
        *(reinterpret_cast<HI_U32 *>(static_cast<HI_UL>(cfg->step_vir_addr_[off_set])) + j) =
          step;  // step of input x_t
        *(reinterpret_cast<HI_U32 *>(static_cast<HI_UL>(cfg->step_vir_addr_[off_set + 1])) + j) =
          step;  // step of output h_t
      }
    }
  }
  param->model_ = &(model->model_);
  ret = NnieParamInit(cfg, param);
  if (ret != RET_OK) {
    LOGE("NnieParamInit failed!");
    return RET_ERROR;
  }
  nnie_run_cfg->run_idx_.seg_idx_ = 0;
  return RET_OK;
}

void NnieCommDelete(NnieParam *pstNnieParamm, NnieModel *nnie_model) {
  NnieParamRelease(pstNnieParamm);
  NnieUnloadModel(nnie_model);
}

int NnieCommGetOutputData(NnieRunCfg *nnie_run_cfg, float *data, HI_U32 output_size, int tensor_index) {
  if (nnie_run_cfg->run_idx_.seg_idx_ <= 0) {
    LOGE("output seg index error.");
    return RET_ERROR;
  }
  HI_U32 ret = 0;
  int id = tensor_index;

  nnie_run_cfg->run_idx_.node_idx_ = id;
  nnie_run_cfg->cfg_.data_ptr_ = data;
  ret = NnieGetDstData(&nnie_run_cfg->cfg_, &nnie_run_cfg->param_, &nnie_run_cfg->run_idx_, output_size);
  if (ret != RET_OK) {
    LOGE("NnieGetDstData failed!");
    return RET_ERROR;
  }
  return RET_OK;
}

int NnieCommFillData(NnieRunCfg *nnie_run_cfg, void *data, HI_U32 input_size, int tensor_index) {
  HI_U32 ret = 0;
  int id = tensor_index;
  HI_U32 seg_idx = nnie_run_cfg->run_idx_.seg_idx_;

  if (id >= nnie_run_cfg->param_.model_->astSeg[seg_idx].u16SrcNum) {
    LOGE("Nnie input node index error!");
    return RET_ERROR;
  }
  nnie_run_cfg->run_idx_.node_idx_ = id;
  nnie_run_cfg->cfg_.data_ptr_ = data;
  ret = NnieFillSrcData(&nnie_run_cfg->cfg_, &nnie_run_cfg->param_, &nnie_run_cfg->run_idx_, input_size);
  if (ret != RET_OK) {
    LOGE("NnieFillSrcData failed!");
    return RET_ERROR;
  }
  return RET_OK;
}

int NnieCommRun(NnieRunCfg *nnie_run_cfg, bool run_box) {
  HI_U32 segidx = nnie_run_cfg->run_idx_.seg_idx_;
  HI_U32 ret = 0;

  if (segidx >= nnie_run_cfg->param_.model_->u32NetSegNum) {
    LOGE("seg num err!\n");
    return RET_ERROR;
  }
  nnie_run_cfg->run_idx_.node_idx_ = 0;
  if (run_box) {
    ret =
      NNIE_ForwardWithBbox(&nnie_run_cfg->param_, &nnie_run_cfg->run_idx_, &nnie_run_cfg->param_.rpn_bbox_, HI_TRUE);
    if (HI_SUCCESS != ret) {
      LOGE("NnieForward failed!");
      return RET_ERROR;
    }
  } else {
    ret = NnieForward(&nnie_run_cfg->param_, &nnie_run_cfg->run_idx_, HI_TRUE);
    if (HI_SUCCESS != ret) {
      LOGE("NnieForward failed!");
      return RET_ERROR;
    }
  }

  nnie_run_cfg->run_idx_.seg_idx_ = ++segidx;
  return RET_OK;
}
}  // namespace nnie
}  // namespace mindspore
