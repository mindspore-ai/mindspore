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

#include "nnie/nnie_interfaces.h"
#include <vector>
#include "src/nnie_print.h"
#include "include/hi_common.h"
#include "include/hi_comm_vb.h"
#include "include/mpi_sys.h"
#include "include/mpi_vb.h"

namespace mindspore {
namespace nnie {
constexpr int kNNIEMaxNameLen = 128;

static int FillRoiPooling(NnieRunCfg *cfg, NnieTensors *inputs, int idx) {
  int *roi_shape = inputs->shape_[idx];
  if (roi_shape[1] != NNIE_COORDI_NUM) {
    LOGE("Roi shape err!");
    return HI_FAILURE;
  }

  if (roi_shape[0] > (int64_t)(cfg->cfg_.max_roi_num_)) {
    LOGE("NNIE_RUNTIME_CONFIG_PATH: The maximum [max_roi_num] value set is less than the actual value: %d < %d.",
         cfg->cfg_.max_roi_num_, (int)(roi_shape[0]));
    return HI_FAILURE;
  }
  cfg->param_.rpn_bbox_.unShape.stWhc.u32Height = roi_shape[0];
  HI_U32 dst_stride = cfg->param_.rpn_bbox_.u32Stride;
  HI_S32 *proposal_result = NNIE_CONVERT_64BIT_ADDR(HI_S32, cfg->param_.rpn_bbox_.u64VirAddr);
  float *float_src_data = reinterpret_cast<float *>(inputs->data_[idx]);
  constexpr int kIndexLeft = 0;
  constexpr int kIndexRight = 1;
  constexpr int kIndexWidth = 2;
  constexpr int kIndexHeight = 3;
  for (size_t j = 0; j < cfg->param_.rpn_bbox_.unShape.stWhc.u32Height; j++) {
    proposal_result[dst_stride / sizeof(HI_U32) * j + kIndexLeft] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + kIndexRight] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + kIndexWidth] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + kIndexHeight] = *(float_src_data++) * NNIE_QUANT_BASE;
  }
  NnieMemFlushCache(cfg->param_.rpn_bbox_.u64PhyAddr,
                    NNIE_CONVERT_64BIT_ADDR(HI_VOID, cfg->param_.rpn_bbox_.u64VirAddr),
                    dst_stride * cfg->param_.rpn_bbox_.unShape.stWhc.u32Height);

  return HI_SUCCESS;
}

int NnieInit(NnieHandle *h, NnieTensors *inputs) {
  NnieModel *model = &(h->cfg_.model_);

  if (inputs->size_ <= 1) {
    LOGE("inputs size need greater than 1!");
    return HI_FAILURE;
  }
  if (NnieLoadModel(h->model_buf_, h->buf_size_, model) != HI_SUCCESS) {
    LOGE("NnieLoadModel failed!");
    return HI_FAILURE;
  }

  std::vector<int64_t> input_shape;
  for (int i = 0; i < inputs->shape_len_[0]; i++) {
    input_shape.push_back(inputs->shape_[0][i]);
  }
  if (NnieCommCreate(&h->cfg_, input_shape) != HI_SUCCESS) {
    NnieCommDelete(&h->cfg_.param_, &h->cfg_.model_);
    return HI_FAILURE;
  }
  return HI_SUCCESS;
}

static size_t GetFillIndex(char **input_name, size_t input_size, const HI_CHAR *name) {
  char prefix[kNNIEMaxNameLen];
  size_t i;
  for (i = 0; i < input_size; ++i) {
    char *post = strrchr(input_name[i], '_');
    if (post && (!strcmp(post + 1, "pre") || !strcmp(post + 1, "post"))) {
      HI_U32 prefix_len = (HI_U32)(post - input_name[i]);
      if (prefix_len >= kNNIEMaxNameLen) return input_size;
      strncpy(prefix, input_name[i], prefix_len);
      prefix[prefix_len] = '\0';
      if (strcmp(prefix, name) == 0) break;
    } else {
      if (strcmp(input_name[i], name) == 0) break;
    }
  }
  if (i == input_size) {
    for (i = 0; i < input_size; ++i) {
      char *post = strrchr(input_name[i], '_');
      if (post && (!strcmp(post + 1, "pre") || !strcmp(post + 1, "post"))) {
        HI_U32 prefix_len = (HI_U32)(post - input_name[i]);
        if (prefix_len >= kNNIEMaxNameLen) return input_size;
        strncpy(prefix, input_name[i], prefix_len);
        prefix[prefix_len] = '\0';
        if (strncmp(prefix, name, prefix_len) == 0) break;
      } else {
        if (strncmp(input_name[i], name, strlen(input_name[i])) == 0) break;
      }
    }
  }
  return i;
}

int NnieFillData(NnieHandle *h, NnieTensors *inputs) {
  SVP_NNIE_MODEL_S *model = h->cfg_.param_.model_;
  unsigned int seg_id = h->cfg_.run_idx_.seg_idx_;
  bool run_box = false;
  size_t i, j;
  if (model->astSeg[seg_id].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    run_box = true;
    for (i = 0; i < static_cast<size_t>(inputs->size_); i++) {
      if (!strcmp(inputs->name_[i], "proposal")) {
        if (FillRoiPooling(&h->cfg_, inputs, i)) {
          return HI_FAILURE;
        }
        break;
      }
    }
    if (i == static_cast<size_t>(inputs->size_)) {
      LOGE("Can't find proposal out!");
      return HI_FAILURE;
    }
  } else if (inputs->size_ != model->astSeg[seg_id].u16SrcNum) {
    LOGE("Input Size Err!");
    return HI_FAILURE;
  }

  for (i = 0; i < model->astSeg[seg_id].u16SrcNum; i++) {
    if (h->cfg_.param_.mem_cfg_.seg_[seg_id].src_node_[i]) {
      continue;
    }
    j = GetFillIndex(inputs->name_, inputs->size_, model->astSeg[seg_id].astSrcNode[i].szName);
    if (j == static_cast<size_t>(inputs->size_)) {
      if (run_box && !strcmp(inputs->name_[i], "proposal")) {
        continue;
      } else {
        j = i;
        LOGW("input tensor name(%s) can't match wk node name(%s).", inputs->name_[i],
             model->astSeg[seg_id].astSrcNode[i].szName);
      }
    }

    auto input_data_type = inputs->dtype_[j];
    SVP_BLOB_TYPE_E src_type = h->cfg_.param_.seg_data_[seg_id].src_[i].enType;
    if (SVP_BLOB_TYPE_U8 <= src_type && src_type <= SVP_BLOB_TYPE_YVU422SP) {
      if (!(input_data_type == NnieDataType::NNIE_INT8 || input_data_type == NnieDataType::NNIE_UINT8)) {
        LOGE("Nnie input node type error!");
        return HI_FAILURE;
      }
    } else {
      if (input_data_type != NnieDataType::NNIE_FLOAT32) {
        LOGE("Nnie input node type error!");
        return HI_FAILURE;
      }
    }
    HI_U32 input_size = 1;
    for (int n = 0; n < inputs->shape_len_[j]; n++) {
      input_size *= inputs->shape_[j][n];
    }
    if (NnieCommFillData(&h->cfg_, inputs->data_[j], input_size, i) != HI_SUCCESS) {
      LOGE("FillData failed!");
      return HI_FAILURE;
    }
  }
  return HI_SUCCESS;
}

int NnieRun(NnieHandle *h, NnieTensors *outputs) {
  SVP_NNIE_MODEL_S *model = h->cfg_.param_.model_;
  unsigned int seg_id = h->cfg_.run_idx_.seg_idx_;
  bool run_box = false;
  int i, j;
  if (model->astSeg[seg_id].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    run_box = true;
  }

  if (NnieCommRun(&h->cfg_, run_box)) {
    LOGE("Nnie Run Fail!");
    return HI_FAILURE;
  }

  // Get output data
  if (outputs->size_ != model->astSeg[seg_id].u16DstNum) {
    LOGE("seg%d: %d output tensors are required, but there are %d outputs.", seg_id, model->astSeg[seg_id].u16DstNum,
         outputs->size_);
    return HI_FAILURE;
  }
  for (i = 0; i < model->astSeg[seg_id].u16DstNum; i++) {
    if (h->cfg_.param_.mem_cfg_.seg_[seg_id].dst_node_[i]) {
      continue;
    }
    j = GetFillIndex(outputs->name_, outputs->size_, model->astSeg[seg_id].astDstNode[i].szName);
    if (j == outputs->size_) {
      j = i;
      LOGW("output tensor name(%s) can't match wk node name(%s).", outputs->name_[j],
           model->astSeg[seg_id].astDstNode[i].szName);
    }
    if (outputs->dtype_[j] == NNIE_FLOAT32) {
      HI_U32 output_size = 1;
      for (int n = 0; n < outputs->shape_len_[j]; n++) {
        output_size *= outputs->shape_[j][n];
      }
      if (NnieCommGetOutputData(&h->cfg_, reinterpret_cast<float *>(outputs->data_[j]), output_size, i) != HI_SUCCESS) {
        return HI_FAILURE;
      }
    } else {
      LOGE("Unsupported DataType!");
      return HI_FAILURE;
    }
  }
  return HI_SUCCESS;
}

void NnieClose(NnieHandle *h) {
  NnieCommDelete(&h->cfg_.param_, &h->cfg_.model_);
  h->load_model_ = 0;
}
}  // namespace nnie
}  // namespace mindspore
