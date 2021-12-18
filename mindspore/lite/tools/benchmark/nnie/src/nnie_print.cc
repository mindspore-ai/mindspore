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

#include "src/nnie_print.h"

namespace mindspore {
namespace nnie {
HI_S32 NniePrintReportResult(NnieParam *pst_nnie_param) {
  HI_U32 u32seg_num = pst_nnie_param->model_->u32NetSegNum;
  HI_U32 i, j, k, n;
  HI_U32 seg_idx_, node_idx_;
  HI_S32 ret;
  HI_CHAR acReportFileName[NNIE_REPORT_NAME_LENGTH] = {'\0'};
  FILE *fp = nullptr;
  HI_U32 *pu32StepAddr = nullptr;
  HI_S32 *ps32ResultAddr = nullptr;
  HI_U32 u32Height, u32Width, u32Chn, u32Stride, u32Dim;

  for (seg_idx_ = 0; seg_idx_ < u32seg_num; seg_idx_++) {
    for (node_idx_ = 0; node_idx_ < pst_nnie_param->model_->astSeg[seg_idx_].u16DstNum; node_idx_++) {
      ret = snprintf(acReportFileName, NNIE_REPORT_NAME_LENGTH, "./ms/fseg%d(%d,%d)_%s.txt", seg_idx_, node_idx_,
                     pst_nnie_param->model_->astSeg[seg_idx_].astDstNode[node_idx_].u32NodeId,
                     pst_nnie_param->model_->astSeg[seg_idx_].astDstNode[node_idx_].szName);
      if (ret < 0) {
        LOGE("Error,create file name failed!");
        return HI_FAILURE;
      }

      fp = fopen(acReportFileName, "w");
      if (fp == nullptr) {
        LOGE("Error,open file failed!");
        return HI_FAILURE;
      }

      if (SVP_BLOB_TYPE_SEQ_S32 == pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].enType) {
        u32Dim = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].unShape.stSeq.u32Dim;
        u32Stride = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u32Stride;
        pu32StepAddr = NNIE_CONVERT_64BIT_ADDR(
          HI_U32, pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].unShape.stSeq.u64VirAddrStep);
        ps32ResultAddr =
          NNIE_CONVERT_64BIT_ADDR(HI_S32, pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u64VirAddr);

        for (n = 0; n < pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u32Num; n++) {
          for (i = 0; i < *(pu32StepAddr + n); i++) {
            for (j = 0; j < u32Dim; j++) {
              fprintf(fp, "%f ", static_cast<float>(*(ps32ResultAddr + j)) / NNIE_QUANT_BASE);
            }
            ps32ResultAddr += u32Stride / sizeof(HI_U32);
          }
        }
      } else {
        u32Height = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].unShape.stWhc.u32Height;
        u32Width = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].unShape.stWhc.u32Width;
        u32Chn = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].unShape.stWhc.u32Chn;
        u32Stride = pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u32Stride;
        ps32ResultAddr =
          NNIE_CONVERT_64BIT_ADDR(HI_S32, pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u64VirAddr);
        fprintf(fp, "%s 4 1 %d %d %d\n", pst_nnie_param->model_->astSeg[seg_idx_].astDstNode[node_idx_].szName,
                u32Height, u32Width, u32Chn);
        for (n = 0; n < pst_nnie_param->seg_data_[seg_idx_].dst_[node_idx_].u32Num; n++) {
          for (i = 0; i < u32Chn; i++) {
            for (j = 0; j < u32Height; j++) {
              for (k = 0; k < u32Width; k++) {
                ret = fprintf(fp, "%f ", static_cast<float>(*(ps32ResultAddr + k)) / NNIE_QUANT_BASE);
                if (ret < 0) {
                  fclose(fp);
                  return HI_FAILURE;
                }
              }
              ps32ResultAddr += u32Stride / sizeof(HI_U32);
            }
          }
        }
      }
      fclose(fp);
    }
  }
  return HI_SUCCESS;
}

HI_S32 NniePrintReportResultInputSeg(NnieParam *pst_nnie_param, int segnum) {
  HI_U32 i, j, k, n;
  HI_U32 seg_idx_ = segnum, node_idx_;
  HI_S32 ret;
  HI_CHAR acReportFileName[NNIE_REPORT_NAME_LENGTH] = {'\0'};
  FILE *fp = nullptr;
  HI_U32 *pu32StepAddr = nullptr;
  HI_S32 *ps32ResultAddr = nullptr;
  HI_U8 *pu8ResultAddr = nullptr;
  HI_U32 u32Height, u32Width, u32Chn, u32Stride, u32Dim;

  for (node_idx_ = 0; node_idx_ < pst_nnie_param->model_->astSeg[seg_idx_].u16SrcNum; node_idx_++) {
    ret = snprintf(acReportFileName, NNIE_REPORT_NAME_LENGTH, "seg%d_layer%d_input(%s)_inst.linear.hex", seg_idx_,
                   pst_nnie_param->model_->astSeg[seg_idx_].astSrcNode[node_idx_].u32NodeId,
                   pst_nnie_param->model_->astSeg[seg_idx_].astSrcNode[node_idx_].szName);
    if (ret < 0) {
      LOGE("Error,create file name failed!\n");
      return HI_FAILURE;
    }

    fp = fopen(acReportFileName, "w");
    if (fp == nullptr) {
      LOGE("Error,open file failed!");
      return HI_FAILURE;
    }

    if (SVP_BLOB_TYPE_SEQ_S32 == pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].enType) {
      u32Dim = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stSeq.u32Dim;
      u32Stride = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Stride;
      pu32StepAddr = NNIE_CONVERT_64BIT_ADDR(
        HI_U32, pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stSeq.u64VirAddrStep);
      ps32ResultAddr = NNIE_CONVERT_64BIT_ADDR(HI_S32, pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u64VirAddr);

      for (n = 0; n < pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Num; n++) {
        for (i = 0; i < *(pu32StepAddr + n); i++) {
          for (j = 0; j < u32Dim; j++) {
            fprintf(fp, "%d ", *(ps32ResultAddr + j));
          }
          ps32ResultAddr += u32Stride / sizeof(HI_U32);
        }
      }
    } else if (pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].enType == SVP_BLOB_TYPE_U8) {
      u32Height = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Height;
      u32Width = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Width;
      u32Chn = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Chn;
      u32Stride = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Stride;
      pu8ResultAddr = NNIE_CONVERT_64BIT_ADDR(HI_U8, pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u64VirAddr);
      for (n = 0; n < pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Num; n++) {
        for (i = 0; i < u32Chn; i++) {
          for (j = 0; j < u32Height; j++) {
            for (k = 0; k < u32Width; k++) {
              fprintf(fp, "%d ", *(pu8ResultAddr + k));
            }
            pu8ResultAddr += u32Stride / sizeof(HI_U8);
          }
        }
      }
    } else {
      u32Height = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Height;
      u32Width = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Width;
      u32Chn = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].unShape.stWhc.u32Chn;
      u32Stride = pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Stride;
      ps32ResultAddr = NNIE_CONVERT_64BIT_ADDR(HI_S32, pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u64VirAddr);
      fprintf(fp, "%s 4 1 %d %d %d\n", pst_nnie_param->model_->astSeg[seg_idx_].astSrcNode[node_idx_].szName, u32Height,
              u32Width, u32Chn);
      for (n = 0; n < pst_nnie_param->seg_data_[seg_idx_].src_[node_idx_].u32Num; n++) {
        for (i = 0; i < u32Chn; i++) {
          for (j = 0; j < u32Height; j++) {
            for (k = 0; k < u32Width; k++) {
              fprintf(fp, "%f ", static_cast<float>(*(ps32ResultAddr + k) / NNIE_QUANT_BASE));
            }
            ps32ResultAddr += u32Stride / sizeof(HI_U32);
          }
        }
      }
    }
    fclose(fp);
  }

  return HI_SUCCESS;
}
}  // namespace nnie
}  // namespace mindspore
