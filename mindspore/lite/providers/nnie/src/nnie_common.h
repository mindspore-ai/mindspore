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

#ifndef MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_COMMON_H_
#define MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_COMMON_H_
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "include/mpi_vb.h"
#include "include/hi_comm_svp.h"
#include "include/hi_nnie.h"
#include "include/mpi_nnie.h"
#include "include/ir/dtype/type_id.h"
#include "src/nnie_cfg_parser.h"

namespace mindspore {
namespace nnie {
#define NNIE_ALIGN_16 16
#define NNIE_ALIGN16(u32Num) ((u32Num + NNIE_ALIGN_16 - 1) / NNIE_ALIGN_16 * NNIE_ALIGN_16)

#define NNIE_ALIGN_32 32
#define NNIE_ALIGN32(u32Num) ((u32Num + NNIE_ALIGN_32 - 1) / NNIE_ALIGN_32 * NNIE_ALIGN_32)

#define NNIE_CONVERT_64BIT_ADDR(Type, Addr) reinterpret_cast<Type *>((HI_UL)(Addr))
#define NNIE_QUANT_BASE 4096

#define NNIE_COORDI_NUM 4
#define NNIE_EACH_SEG_STEP_ADDR_NUM 2
#define NNIE_REPORT_NAME_LENGTH 64

typedef struct {
  SVP_NNIE_MODEL_S model_;
  SVP_MEM_INFO_S model_buf_;  // store Model file
} NnieModel;
typedef struct {
  SVP_SRC_BLOB_S src_[SVP_NNIE_MAX_INPUT_NUM];
  SVP_DST_BLOB_S dst_[SVP_NNIE_MAX_OUTPUT_NUM];
} NnieSegData;

typedef struct {
  bool src_node_[SVP_NNIE_MAX_INPUT_NUM];
  bool dst_node_[SVP_NNIE_MAX_OUTPUT_NUM];
} NNIEMemSegInfo;

typedef struct {
  NNIEMemSegInfo seg_[SVP_NNIE_MAX_NET_SEG_NUM];
} NNIEMemCfg;

typedef struct {
  SVP_NNIE_MODEL_S *model_;
  HI_U32 task_buf_size_[SVP_NNIE_MAX_NET_SEG_NUM];
  SVP_MEM_INFO_S task_buf_;
  SVP_MEM_INFO_S tmp_buf_;
  SVP_MEM_INFO_S step_buf_;  // store Lstm step info
  SVP_SRC_BLOB_S rpn_bbox_;
  NnieSegData seg_data_[SVP_NNIE_MAX_NET_SEG_NUM];  // each seg's input and output blob
  SVP_NNIE_FORWARD_CTRL_S forward_ctrl_[SVP_NNIE_MAX_NET_SEG_NUM];
  SVP_NNIE_FORWARD_WITHBBOX_CTRL_S forward_with_bbox_ctrl_[SVP_NNIE_MAX_NET_SEG_NUM];
  NNIEMemCfg mem_cfg_;
  bool get_mem_strong;
} NnieParam;

typedef struct {
  bool pass_align16_io_;
  HI_VOID *data_ptr_;
  HI_U32 max_input_num_;
  HI_U32 max_roi_num_;
  HI_U32 step_;
  HI_U64 step_vir_addr_[NNIE_EACH_SEG_STEP_ADDR_NUM *
                        SVP_NNIE_MAX_NET_SEG_NUM];  // virtual addr of LSTM's or RNN's step buffer
  SVP_NNIE_ID_E nnie_core_id_[SVP_NNIE_MAX_NET_SEG_NUM];
} NnieCfg;

typedef struct {
  HI_U32 seg_idx_;
  HI_U32 node_idx_;
  HI_U32 max_seg_id_;
} NnieDataIndex;

typedef struct {
  HI_U32 src_size_[SVP_NNIE_MAX_INPUT_NUM];
  HI_U32 dst_size_[SVP_NNIE_MAX_OUTPUT_NUM];
} NnieBlobSize;

typedef struct {
  NnieModel model_;
  NnieParam param_;
  NnieCfg cfg_;
  NnieDataIndex run_idx_;
} NnieRunCfg;

int NnieLoadModel(char *model_buf, int size, NnieModel *nnie_model);

int NnieCommCreate(NnieRunCfg *nnie_run_cfg, const std::vector<int64_t> &input_shape);

void NnieCommDelete(NnieParam *pstNnieParamm, NnieModel *nnie_model);

int NnieCommRun(NnieRunCfg *nnie_run_cfg, bool run_box);

int NnieCommFillData(NnieRunCfg *nnie_run_cfg, void *data, HI_U32 input_size, int id);

int NnieCommGetOutputData(NnieRunCfg *nnie_run_cfg, float *data, HI_U32 output_size, int tensor_index);

HI_U32 GetBlobSize(const SVP_SRC_BLOB_S &blob);
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_NNIE_SRC_NNIE_COMMON_H_
