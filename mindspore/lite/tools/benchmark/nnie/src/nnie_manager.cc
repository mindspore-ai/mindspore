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
#include <cstring>
#include "src/nnie_manager.h"
#include "src/nnie_common.h"
#include "src/nnie_print.h"
#include "src/nnie_memory.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
constexpr int kNumInput2 = 2;

namespace mindspore {
namespace nnie {
int NNIEManager::CfgInit(int max_roi_num, int step, const std::vector<int> &core_id) {
  memset(&nnie_cfg_, 0, sizeof(NnieRunCfg));

  nnie_cfg_.cfg_.max_roi_num_ = max_roi_num;

  nnie_cfg_.cfg_.step_ = step;
  for (size_t i = 0; i < SVP_NNIE_MAX_NET_SEG_NUM && i < core_id.size(); i++) {
    if (core_id[i] < SVP_NNIE_ID_BUTT) {
      nnie_cfg_.cfg_.nnie_core_id_[i] = (SVP_NNIE_ID_E)core_id[i];
    } else {
      LOGE("nnie core num toobig.\n");
      return RET_ERROR;
    }
  }
  return RET_OK;
}
void NNIEManager::SetInputNum(int max_input_num) { nnie_cfg_.cfg_.max_input_num_ = max_input_num; }

int NNIEManager::Init(char *model_buf, int size, const std::vector<mindspore::MSTensor> &inputs) {
  if (NnieCommCreate(&nnie_cfg_, model_buf, size, inputs) != RET_OK) {
    NnieCommDelete(&nnie_cfg_.param_, &nnie_cfg_.model_);
    return RET_ERROR;
  }
  return RET_OK;
}

int NNIEManager::Run(std::vector<mindspore::MSTensor> *outputs, unsigned int seg_id,
                     const std::vector<std::vector<int64_t>> &outputs_shape) {
  bool run_box = false;
  nnie_cfg_.run_idx_.seg_idx_ = seg_id;
  if (nnie_cfg_.param_.model_->astSeg[seg_id].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    run_box = true;
  }

  if (NnieCommRun(&nnie_cfg_, run_box)) {
    LOGE("Nnie Run Fail!");
    return RET_ERROR;
  }
  if (GetOutputData(outputs, outputs_shape, run_box)) {
    LOGE("Get Output Data Fail!");
    return RET_ERROR;
  }
  return RET_OK;
}

void NNIEManager::Release() {
  // NniePrintReportResult(&nnie_cfg_.param_);
  NnieCommDelete(&nnie_cfg_.param_, &nnie_cfg_.model_);
}

int NNIEManager::GetOutputData(std::vector<mindspore::MSTensor> *outputs,
                               const std::vector<std::vector<int64_t>> &outputs_shape, bool run_box) {
  int i, j, output_size = outputs->size();
  if (output_size != nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_ - 1].u16DstNum) {
    LOGE("seg%d: %d output tensors are required, but there are %d outputs.", nnie_cfg_.run_idx_.seg_idx_ - 1,
         nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_ - 1].u16DstNum, output_size);
    return RET_ERROR;
  }

  if (run_box) {
    for (i = 0; i < output_size; i++) {
      auto input_data_type = (*outputs)[i].DataType();
      if (input_data_type == DataType::kNumberTypeFloat32) {
        auto ptr_shape = outputs_shape[i];
        int max_roi_num = nnie_cfg_.param_.seg_data_[nnie_cfg_.run_idx_.seg_idx_ - 1].dst_[0].u32Num;
        ptr_shape.insert(ptr_shape.begin(), max_roi_num);
        (*outputs)[i].SetShape(ptr_shape);
      } else {
        LOGE("Unsupported DataType!");
        return RET_ERROR;
      }
    }
  }
  HI_U32 seg_idx = nnie_cfg_.run_idx_.seg_idx_ - 1;
  for (i = 0; i < nnie_cfg_.param_.model_->astSeg[seg_idx].u16DstNum; i++) {
    if (nnie_cfg_.param_.mem_cfg_.seg_[seg_idx].dst_node_[i]) {
      continue;
    }

    j = GetFillIndex(*outputs, output_size, nnie_cfg_.param_.model_->astSeg[seg_idx].astDstNode[i].szName);
    if (j == output_size) {
      j = i;
      LOGI("output tensor name(%s) can't match wk node name(%s).", (*outputs)[j].Name().c_str(),
           nnie_cfg_.param_.model_->astSeg[seg_idx].astDstNode[i].szName);
    }

    auto input_data_type = (*outputs)[j].DataType();
    if (input_data_type == DataType::kNumberTypeFloat32) {
      auto ptr_shape = (*outputs)[j].Shape();
      auto ptr = reinterpret_cast<float *>((*outputs)[j].MutableData());
      if (NnieCommGetOutputData(&nnie_cfg_, ptr, ptr_shape.data(), ptr_shape.size(), i) != RET_OK) {
        return RET_ERROR;
      }
    } else {
      LOGE("Unsupported DataType!");
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int NNIEManager::FillRoiPooling(mindspore::MSTensor *input) {
  auto roi_shape = input->Shape();
  if (roi_shape[1] != NNIE_COORDI_NUM) {
    LOGE("Roi shape err!");
    return RET_ERROR;
  }

  if (roi_shape[0] > static_cast<int64_t>(nnie_cfg_.cfg_.max_roi_num_)) {
    LOGE("NNIE_RUNTIME_CONFIG_PATH: The maximum [max_roi_num] value set is less than the actual value: %d < %d.",
         nnie_cfg_.cfg_.max_roi_num_, static_cast<int>(roi_shape[0]));
    return RET_ERROR;
  }
  nnie_cfg_.param_.rpn_bbox_.unShape.stWhc.u32Height = roi_shape[0];
  HI_U32 dst_stride = nnie_cfg_.param_.rpn_bbox_.u32Stride;
  auto proposal_result = NNIE_CONVERT_64BIT_ADDR(HI_S32, nnie_cfg_.param_.rpn_bbox_.u64VirAddr);
  auto float_src_data = reinterpret_cast<float *>(input->MutableData());

  for (size_t j = 0; j < nnie_cfg_.param_.rpn_bbox_.unShape.stWhc.u32Height; j++) {
    proposal_result[dst_stride / sizeof(HI_U32) * j] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + 1] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + 2] = *(float_src_data++) * NNIE_QUANT_BASE;
    proposal_result[dst_stride / sizeof(HI_U32) * j + 3] = *(float_src_data++) * NNIE_QUANT_BASE;
  }
  NnieMemFlushCache(nnie_cfg_.param_.rpn_bbox_.u64PhyAddr,
                    NNIE_CONVERT_64BIT_ADDR(HI_VOID, nnie_cfg_.param_.rpn_bbox_.u64VirAddr),
                    dst_stride * nnie_cfg_.param_.rpn_bbox_.unShape.stWhc.u32Height);

  return RET_OK;
}

int NNIEManager::FillData(std::vector<mindspore::MSTensor> *inputs, unsigned int seg_id) {
  bool run_box = false;
  size_t i, j;
  size_t input_size = inputs->size();
  if (seg_id >= nnie_cfg_.param_.model_->u32NetSegNum) {
    LOGE("seg num err!");
    return RET_ERROR;
  }

  nnie_cfg_.run_idx_.seg_idx_ = seg_id;

  if (nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    run_box = true;
    for (i = 0; i < (input_size - 1); i++) {
      if ((*inputs)[i].Name() == "proposal") {
        FillRoiPooling(&(*inputs)[i]);
        break;
      }
    }
    if (i == (input_size - 1)) {
      LOGE("Can't find proposal out!");
      return RET_ERROR;
    }
  } else if ((input_size < kNumInput2) ||
             (input_size - 1) != nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_].u16SrcNum) {
    LOGE("Input Size Err!");
    return RET_ERROR;
  }

  for (i = 0; i < nnie_cfg_.param_.model_->astSeg[seg_id].u16SrcNum; i++) {
    if (nnie_cfg_.param_.mem_cfg_.seg_[seg_id].src_node_[i]) {
      continue;
    }
    j = GetFillIndex(*inputs, input_size - 1, nnie_cfg_.param_.model_->astSeg[seg_id].astSrcNode[i].szName);
    if (j == (input_size - 1)) {
      if (run_box && (*inputs)[i].Name() == "proposal") {
        continue;
      } else {
        j = i;
        LOGI("input tensor name(%s) can't match wk node name(%s).", (*inputs)[i].Name().c_str(),
             nnie_cfg_.param_.model_->astSeg[seg_id].astSrcNode[i].szName);
      }
    }

    auto input_data_type = (*inputs)[j].DataType();
    if ((input_data_type == DataType::kNumberTypeFloat32) || (input_data_type == DataType::kNumberTypeUInt8) ||
        (input_data_type == DataType::kNumberTypeInt8)) {
      auto ptr_shape = (*inputs)[j].Shape();
      if (NnieCommFillData(&nnie_cfg_, (*inputs)[j].MutableData(), input_data_type, ptr_shape.data(), ptr_shape.size(),
                           i) != RET_OK) {
        LOGE("FillData failed!");
        return RET_ERROR;
      }
    } else {
      LOGE("Unsupported DataType!");
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace nnie
}  // namespace mindspore
