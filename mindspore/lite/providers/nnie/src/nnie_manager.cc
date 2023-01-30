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
#include <string>
#include <map>
#include <memory>
#include "src/nnie_manager.h"
#include "src/nnie_common.h"
#include "src/nnie_print.h"
#include "src/nnie_memory.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace nnie {
constexpr int kUINT16_MAX = 65535;
constexpr int kNumInput2 = 2;
constexpr int kPreSize = 4;
constexpr int kPostSize = 5;

static size_t GetFillIndex(const std::vector<mindspore::MSTensor> &inputs, size_t input_size, const HI_CHAR *name) {
  size_t j;
  for (j = 0; j < input_size; j++) {
    auto input_str = inputs[j].Name();
    if (input_str.length() > kPreSize) {
      if (input_str.substr(input_str.length() - kPreSize) == "_pre") {
        input_str = input_str.substr(0, input_str.length() - kPreSize);
      } else if (input_str.length() > kPostSize) {
        if (input_str.substr(input_str.length() - kPostSize) == "_post") {
          input_str = input_str.substr(0, input_str.length() - kPostSize);
        }
      }
    }

    if (strcmp(input_str.c_str(), name) == 0) {
      break;
    }
  }
  if (j == input_size) {
    for (j = 0; j < input_size; j++) {
      auto input_str = inputs[j].Name();
      if (input_str.length() > kPreSize) {
        if (input_str.substr(input_str.length() - kPreSize) == "_pre") {
          input_str = input_str.substr(0, input_str.length() - kPreSize);
        } else if (input_str.length() > kPostSize) {
          if (input_str.substr(input_str.length() - kPostSize) == "_post") {
            input_str = input_str.substr(0, input_str.length() - kPostSize);
          }
        }
      }

      if (strncmp(input_str.c_str(), name, input_str.length()) == 0) {
        break;
      }
    }
  }
  return j;
}

int NNIEManager::CfgInit(const Flags &flags, int max_seg_id) {
  memset(&nnie_cfg_, 0, sizeof(NnieRunCfg));

  nnie_cfg_.cfg_.pass_align16_io_ = flags.keep_origin_output_;
  nnie_cfg_.param_.get_mem_strong = false;
  nnie_cfg_.run_idx_.max_seg_id_ = flags.keep_origin_output_ ? max_seg_id + 1 : kUINT16_MAX;
  nnie_cfg_.cfg_.max_roi_num_ = flags.max_roi_num_;
  nnie_cfg_.cfg_.step_ = flags.time_step_;
  if (flags.core_ids_.size() == 1) {
    for (size_t i = 0; i < SVP_NNIE_MAX_NET_SEG_NUM; i++) {
      if (flags.core_ids_[0] < SVP_NNIE_ID_BUTT) {
        nnie_cfg_.cfg_.nnie_core_id_[i] = (SVP_NNIE_ID_E)flags.core_ids_[0];
      } else {
        LOGE("nnie core num toobig.\n");
        return RET_ERROR;
      }
    }
  }
  for (size_t i = 0; i < SVP_NNIE_MAX_NET_SEG_NUM && i < flags.core_ids_.size(); i++) {
    if (flags.core_ids_[i] < SVP_NNIE_ID_BUTT) {
      nnie_cfg_.cfg_.nnie_core_id_[i] = (SVP_NNIE_ID_E)flags.core_ids_[i];
    } else {
      LOGE("nnie core num toobig.\n");
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NNIEManager::MallocBlobData(SVP_SRC_BLOB_S *blob, mindspore::MSTensor *tensor, HI_U32 blob_size) {
  auto ret = NnieMemMallocCached(tensor->Name().c_str(), nullptr, reinterpret_cast<HI_U64 *>(&blob->u64PhyAddr),
                                 reinterpret_cast<void **>(&blob->u64VirAddr), blob_size);
  if (HI_SUCCESS != ret) {
    LOGE("Error,MallocBlobData failed!");
    return RET_ERROR;
  }
  blobs_.push_back(blob);
  tensors_.push_back(tensor);
  return RET_OK;
}

int NNIEManager::SetBlobAddr(SVP_SRC_BLOB_S *blob, HI_U64 virt, mindspore::MSTensor *tensor,
                             std::shared_ptr<Allocator> allocator) {
  HI_U32 blob_size = GetBlobSize(*blob);
  if (virt == 0) {
    auto iter = std::find(blobs_.begin(), blobs_.end(), blob);
    if (iter == blobs_.end()) {
      if (MallocBlobData(blob, tensor, blob_size) != RET_OK) {
        LOGE("Failed to malloc.");
        return RET_ERROR;
      }
    }
    tensor->SetAllocator(allocator);
    tensor->SetData(reinterpret_cast<void *>(blob->u64VirAddr));
    LOGI("\nSet %s allocator!", tensor->Name().c_str());
  } else {
    auto ret = NnieGetVirMemInfo(virt, &blob->u64PhyAddr);
    if (ret == HI_SUCCESS) {
      blob->u64VirAddr = virt;
      LOGI("Get physical address %llu.", blob->u64PhyAddr);
    } else {
      auto iter = std::find(blobs_.begin(), blobs_.end(), blob);
      if (iter == blobs_.end()) {
        if (MallocBlobData(blob, tensor, blob_size) != RET_OK) {
          LOGE("Error, tensor data pointer is not MMZ memory, failed to malloc.");
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int NNIEManager::LoadInputs(std::vector<mindspore::MSTensor> *inputs, std::shared_ptr<Allocator> allocator) {
  size_t input_size = inputs->size();
  if ((input_size < kNumInput2) || (input_size - 1) != nnie_cfg_.param_.model_->astSeg[0].u16SrcNum) {
    LOGE("Input Size Err!");
    return RET_ERROR;
  }

  for (size_t i = 0; i < nnie_cfg_.param_.model_->astSeg[0].u16SrcNum; i++) {
    size_t j = GetFillIndex(*inputs, input_size - 1, nnie_cfg_.param_.model_->astSeg[0].astSrcNode[i].szName);
    if (j == (input_size - 1)) {
      j = i;
      LOGI("input tensor name(%s) can't match wk node name(%s).", (*inputs)[j].Name().c_str(),
           nnie_cfg_.param_.model_->astSeg[0].astSrcNode[i].szName);
    }
    HI_U64 virt = (HI_U64)(HI_UL)((*inputs)[j].Data().get());
    auto blob = &nnie_cfg_.param_.seg_data_[0].src_[i];
    if (SetBlobAddr(blob, virt, &(*inputs)[j], allocator) != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NNIEManager::LoadOutputs(std::vector<mindspore::MSTensor> *outputs, std::shared_ptr<Allocator> allocator) {
  int output_size = outputs->size();
  HI_U32 seg_id = nnie_cfg_.model_.model_.u32NetSegNum - 1;
  if (output_size != nnie_cfg_.param_.model_->astSeg[seg_id].u16DstNum) {
    LOGE("seg%d: %d output tensors are required, but there are %d outputs.", nnie_cfg_.run_idx_.seg_idx_ - 1,
         nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_ - 1].u16DstNum, output_size);
    return RET_ERROR;
  }
  if (nnie_cfg_.param_.model_->astSeg[seg_id].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    LOGE("Unsupported use PassAlign16InOutput!");
    return RET_ERROR;
  }

  for (int i = 0; i < nnie_cfg_.param_.model_->astSeg[seg_id].u16DstNum; i++) {
    int j = GetFillIndex(*outputs, output_size, nnie_cfg_.param_.model_->astSeg[seg_id].astDstNode[i].szName);
    if (j == output_size) {
      j = i;
      LOGI("output tensor name(%s) can't match wk node name(%s).", (*outputs)[j].Name().c_str(),
           nnie_cfg_.param_.model_->astSeg[seg_id].astDstNode[i].szName);
    }

    SVP_SRC_BLOB_S *blob = &nnie_cfg_.param_.seg_data_[seg_id].dst_[i];
    if (SVP_BLOB_TYPE_U8 <= blob->enType && SVP_BLOB_TYPE_YVU422SP >= blob->enType) {
      LOGE("Nnie output type error");
      return RET_ERROR;
    }
    HI_U64 virt = (HI_U64)(HI_UL)((*outputs)[j].Data().get());
    if (SetBlobAddr(blob, virt, &(*outputs)[j], allocator) != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void NNIEManager::SetInputNum(int max_input_num) { nnie_cfg_.cfg_.max_input_num_ = max_input_num; }

int NNIEManager::Init(char *model_buf, int size, const std::vector<mindspore::MSTensor> &inputs) {
  NnieModel *model = &nnie_cfg_.model_;

  if (inputs.size() <= 1) {
    LOGE("inputs size need greater than 1!");
    return RET_ERROR;
  }
  auto ret = NnieLoadModel(model_buf, size, model);
  if (ret != RET_OK) {
    LOGE("NnieLoadModel failed!");
    return RET_ERROR;
  }
  auto j = GetFillIndex(inputs, inputs.size() - 1, model->model_.astSeg[0].astSrcNode[0].szName);
  if (j == (inputs.size() - 1)) {
    j = 0;
    LOGI("input tensor name(%s) can't match wk node name(%s).", inputs[0].Name().c_str(),
         model->model_.astSeg[0].astSrcNode[0].szName);
  }

  if (NnieCommCreate(&nnie_cfg_, inputs[j].Shape()) != RET_OK) {
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

void NNIEManager::Release(bool resize_flag) {
  for (auto &blob : blobs_) {
    NNIE_MEM_FREE(blob->u64PhyAddr, blob->u64VirAddr);
    blob->u64VirAddr = 0;
    blob->u64PhyAddr = 0;
  }
  blobs_.clear();
  if (resize_flag) {
    for (auto &tensor : tensors_) {
      tensor->SetData(nullptr);
      tensor->SetAllocator(nullptr);
    }
  }
  tensors_.clear();
  NnieCommDelete(&nnie_cfg_.param_, &nnie_cfg_.model_);
}

int NNIEManager::GetOutputData(std::vector<mindspore::MSTensor> *outputs,
                               const std::vector<std::vector<int64_t>> &outputs_shape, bool run_box) {
  int output_size = outputs->size();
  if (output_size != nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_ - 1].u16DstNum) {
    LOGE("seg%d: %d output tensors are required, but there are %d outputs.", nnie_cfg_.run_idx_.seg_idx_ - 1,
         nnie_cfg_.param_.model_->astSeg[nnie_cfg_.run_idx_.seg_idx_ - 1].u16DstNum, output_size);
    return RET_ERROR;
  }

  int i;
  int j;
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
      HI_U32 output_element = static_cast<HI_U32>((*outputs)[j].ElementNum());
      auto ptr = reinterpret_cast<float *>((*outputs)[j].MutableData());
      if (NnieCommGetOutputData(&nnie_cfg_, ptr, output_element, i) != RET_OK) {
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

int NNIEManager::SetAllocatorTensor(mindspore::MSTensor *tensor, SVP_SRC_BLOB_S *blob,
                                    std::shared_ptr<Allocator> allocator) {
  int step;
  auto data_type = tensor->DataType();
  if (data_type == DataType::kNumberTypeFloat32) {
    step = sizeof(float);
  } else if ((data_type == DataType::kNumberTypeUInt8) || (data_type == DataType::kNumberTypeInt8)) {
    step = sizeof(unsigned char);
  } else {
    LOGE("Unsupported DataType!");
    return RET_ERROR;
  }
  LOGI("\ninput %s :%d * %d = %d <-> %d", tensor->Name().c_str(), step, blob->unShape.stWhc.u32Width,
       step * blob->unShape.stWhc.u32Width, blob->u32Stride);

  if (blob->unShape.stWhc.u32Width * step == blob->u32Stride) {
    if (((tensor->Data() == nullptr) || tensor->allocator() == allocator) && (blob->u64VirAddr != 0)) {
      tensor->SetAllocator(allocator);
      tensor->SetData(reinterpret_cast<void *>(blob->u64VirAddr));
      LOGI("\nSet input %s allocator!", tensor->Name().c_str());
    }
  }
  return RET_OK;
}

int NNIEManager::SetAllocatorInputs(std::vector<mindspore::MSTensor> *inputs, bool run_box,
                                    std::shared_ptr<Allocator> allocator, unsigned int seg_id) {
  size_t i;
  size_t j;
  size_t input_size = inputs->size();
  if (seg_id >= nnie_cfg_.param_.model_->u32NetSegNum) {
    LOGE("seg num err!");
    return RET_ERROR;
  }

  if (!run_box) {
    if ((input_size < kNumInput2) || (input_size - 1) != nnie_cfg_.param_.model_->astSeg[seg_id].u16SrcNum) {
      LOGE("Input Size Err!");
      return RET_ERROR;
    }
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
    SVP_SRC_BLOB_S *blob = &nnie_cfg_.param_.seg_data_[seg_id].src_[i];
    SVP_BLOB_TYPE_E src_type = blob->enType;

    if (src_type != SVP_BLOB_TYPE_SEQ_S32) {
      SetAllocatorTensor(&(*inputs)[j], blob, allocator);
    }
  }
  return RET_OK;
}

int NNIEManager::SetAllocatorOutputs(std::vector<mindspore::MSTensor> *outputs, bool run_box,
                                     std::shared_ptr<Allocator> allocator, unsigned int seg_id) {
  size_t i;
  size_t j;
  size_t output_size = outputs->size();
  if (output_size != nnie_cfg_.param_.model_->astSeg[seg_id].u16DstNum) {
    LOGE("seg%d: %d output tensors are required.", seg_id, nnie_cfg_.param_.model_->astSeg[seg_id].u16DstNum);
    return RET_ERROR;
  }

  for (i = 0; i < nnie_cfg_.param_.model_->astSeg[seg_id].u16DstNum; i++) {
    if (nnie_cfg_.param_.mem_cfg_.seg_[seg_id].dst_node_[i]) {
      continue;
    }

    j = GetFillIndex(*outputs, output_size, nnie_cfg_.param_.model_->astSeg[seg_id].astDstNode[i].szName);
    if (j == output_size) {
      j = i;
      LOGI("output tensor name(%s) can't match wk node name(%s).", (*outputs)[j].Name().c_str(),
           nnie_cfg_.param_.model_->astSeg[seg_id].astDstNode[i].szName);
    }

    auto output_data_type = (*outputs)[j].DataType();
    if (output_data_type == DataType::kNumberTypeFloat32) {
      SVP_SRC_BLOB_S *blob = &nnie_cfg_.param_.seg_data_[seg_id].dst_[i];
      if (SVP_BLOB_TYPE_U8 <= blob->enType && SVP_BLOB_TYPE_YVU422SP >= blob->enType) {
        LOGE("Nnie output type error");
        return RET_ERROR;
      } else if (SVP_BLOB_TYPE_SEQ_S32 != blob->enType) {
        if ((blob->unShape.stWhc.u32Width * sizeof(float) == blob->u32Stride)) {
          if ((((*outputs)[j].Data() == nullptr) || (*outputs)[j].allocator() == allocator) &&
              (blob->u64VirAddr != 0)) {
            (*outputs)[j].SetAllocator(allocator);
            (*outputs)[j].SetData(reinterpret_cast<void *>(blob->u64VirAddr));
            LOGI("\nSet output %s allocator!", (*outputs)[j].Name().c_str());
          }
        }
      }
    } else {
      LOGE("Unsupported DataType!");
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NNIEManager::SetAllocator(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              std::shared_ptr<Allocator> allocator, unsigned int seg_id) {
  bool run_box = false;
  if (nnie_cfg_.param_.model_->astSeg[seg_id].enNetType == SVP_NNIE_NET_TYPE_ROI) {
    run_box = true;
  }
  if (SetAllocatorInputs(inputs, run_box, allocator, seg_id) != RET_OK) {
    LOGE("SetAllocatorInputs failed!");
    return RET_ERROR;
  }
  if (SetAllocatorOutputs(outputs, run_box, allocator, seg_id) != RET_OK) {
    LOGE("SetAllocatorOutputs failed!");
    return RET_ERROR;
  }
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
    SVP_BLOB_TYPE_E src_type = nnie_cfg_.param_.seg_data_[seg_id].src_[i].enType;
    if (SVP_BLOB_TYPE_U8 <= src_type && src_type <= SVP_BLOB_TYPE_YVU422SP) {
      if (!(input_data_type == DataType::kNumberTypeUInt8 || input_data_type == DataType::kNumberTypeInt8)) {
        LOGE("Nnie input node type error!");
        return RET_ERROR;
      }
    } else {
      if (input_data_type != DataType::kNumberTypeFloat32) {
        LOGE("Nnie input node type error!");
        return RET_ERROR;
      }
    }
    HI_U32 input_element = static_cast<HI_U32>((*inputs)[j].ElementNum());
    if (NnieCommFillData(&nnie_cfg_, (*inputs)[j].MutableData(), input_element, i) != RET_OK) {
      LOGE("FillData failed!");
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace nnie
}  // namespace mindspore
