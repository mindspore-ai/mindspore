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

#include "manager/acl_model_helper.h"
#include <numeric>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>
#include "include/errorcode.h"
#include "common/check_base.h"
#include "src/custom_allocator.h"
#include "common/string_util.h"
#include "common/op_attr.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kByteSize = 8;  // 1 byte = 8 bits
bool Cmp(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs[kClassId] < rhs[kClassId]) {
    return true;
  } else if (lhs[kClassId] == rhs[kClassId]) {
    return lhs[kScore] > rhs[kScore];
  }
  return false;
}
}  // namespace
int GetAclModelType(const schema::Primitive *primitive, AclModelType *acl_model_type) {
  MS_CHECK_TRUE_MSG(primitive != nullptr && acl_model_type != nullptr, RET_ERROR, "input params contain nullptr.");
  auto op = primitive->value_as_Custom();
  MS_CHECK_TRUE_MSG(op != nullptr, RET_ERROR, "custom op is nullptr.");
  auto attrs = op->attr();
  MS_CHECK_TRUE_MSG(attrs != nullptr && attrs->size() >= 1, RET_ERROR, "custom op attr is invalid.");
  std::string acl_model_type_str;
  for (size_t i = 0; i < attrs->size(); i++) {
    auto attr = attrs->Get(i);
    MS_CHECK_TRUE_MSG(attr != nullptr && attr->name() != nullptr, RET_ERROR, "invalid attr.");
    if (attr->name()->str() != kNetType) {
      continue;
    }
    auto data_info = attr->data();
    MS_CHECK_TRUE_MSG(data_info != nullptr, RET_ERROR, "attr data is nullptr");
    int data_size = static_cast<int>(data_info->size());
    for (int j = 0; j < data_size; j++) {
      acl_model_type_str.push_back(static_cast<char>(data_info->Get(j)));
    }
    break;
  }
  if (acl_model_type_str.empty()) {
    *acl_model_type = AclModelType::kCnn;
    return RET_OK;
  }
  if (!IsValidUnsignedNum(acl_model_type_str)) {
    MS_LOG(ERROR) << "net_type attr data is invalid num.";
    return RET_ERROR;
  }
  int acl_model_type_val = stoi(acl_model_type_str);
  if (acl_model_type_val < static_cast<int>(AclModelType::kCnn) ||
      acl_model_type_val > static_cast<int>(AclModelType::kRecurrent)) {
    MS_LOG(ERROR) << "net_type val is invalid. " << acl_model_type_val;
    return RET_ERROR;
  }
  *acl_model_type = static_cast<AclModelType>(acl_model_type_val);
  return RET_OK;
}

int GetAclDataInfo(struct AclDataInfo *acl_data_info, svp_acl_mdl_desc *acl_mdl_desc, int index) {
  MS_CHECK_TRUE_MSG(acl_data_info != nullptr && acl_mdl_desc != nullptr, RET_ERROR, "input params contain nullptr");
  int ret;
  if (acl_data_info->data_mode == AclDataInfo::Input) {
    ret = svp_acl_mdl_get_input_dims(acl_mdl_desc, index, &(acl_data_info->dim_info));
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl get input dims failed.");
    acl_data_info->stride = svp_acl_mdl_get_input_default_stride(acl_mdl_desc, index);
    MS_CHECK_TRUE_MSG(acl_data_info->stride != 0, RET_ERROR, "acl get input default stride failed.");
    acl_data_info->data_size = svp_acl_mdl_get_input_size_by_index(acl_mdl_desc, index);
    MS_CHECK_TRUE_MSG(acl_data_info->data_size != 0, RET_ERROR, "acl get input size by index failed.");
  } else {
    ret = svp_acl_mdl_get_output_dims(acl_mdl_desc, index, &(acl_data_info->dim_info));
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl get output dims failed.");
    acl_data_info->stride = svp_acl_mdl_get_output_default_stride(acl_mdl_desc, index);
    MS_CHECK_TRUE_MSG(acl_data_info->stride != 0, RET_ERROR, "acl get output default stride failed.");
    acl_data_info->data_size = svp_acl_mdl_get_output_size_by_index(acl_mdl_desc, index);
    MS_CHECK_TRUE_MSG(acl_data_info->data_size != 0, RET_ERROR, "acl get output size by index failed.");
  }
  return RET_OK;
}

int AddDatasetBuffer(svp_acl_mdl_dataset *acl_mdl_dataset, size_t data_buffer_size, size_t stride, void *data) {
  MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
  MS_CHECK_TRUE_MSG(acl_mdl_dataset != nullptr, RET_ERROR, "acl_mdl_dataset is nullptr.");
  auto *data_buffer = svp_acl_create_data_buffer(data, data_buffer_size, stride);
  MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "create data buffer failed.");
  int ret = svp_acl_mdl_add_dataset_buffer(acl_mdl_dataset, data_buffer);
  if (ret != SVP_ACL_SUCCESS) {
    MS_LOG(ERROR) << "add dataset buffer failed.";
    ret = svp_acl_destroy_data_buffer(data_buffer);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "destroy data buffer failed.");
    data_buffer = nullptr;
    return RET_ERROR;
  }
  return RET_OK;
}

int DestroyAclDataset(svp_acl_mdl_dataset **acl_mdl_dataset,
                      const std::unordered_map<size_t, bool> &mem_managed_by_tensor, const AllocatorPtr &allocator) {
  if (*acl_mdl_dataset == nullptr) {
    MS_LOG(INFO) << "acl_mdl_dataset is nullptr, no need to destroy";
    return RET_OK;
  }
  int ret;
  auto dataset_buffer_size = svp_acl_mdl_get_dataset_num_buffers(*acl_mdl_dataset);
  MS_CHECK_TRUE_MSG(dataset_buffer_size == mem_managed_by_tensor.size(), RET_ERROR,
                    "dataset_buffer_size:" << dataset_buffer_size << " is not equal to mem_managed_by_tensor.size():"
                                           << mem_managed_by_tensor.size());
  for (size_t i = 0; i < dataset_buffer_size; i++) {
    MS_CHECK_TRUE_MSG(mem_managed_by_tensor.find(i) != mem_managed_by_tensor.end(), RET_ERROR,
                      "invalid dataset buffer index");
    svp_acl_data_buffer *acl_data_buffer = svp_acl_mdl_get_dataset_buffer(*acl_mdl_dataset, i);
    MS_CHECK_TRUE_MSG(acl_data_buffer != nullptr, RET_ERROR, "get acl data buffer failed.");
    if (!mem_managed_by_tensor.at(i)) {
      void *tmp = svp_acl_get_data_buffer_addr(acl_data_buffer);
      ret = AclFree(&tmp);
      MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "AclFree tmp failed");
      tmp = nullptr;
    }
    ret = svp_acl_destroy_data_buffer(acl_data_buffer);
    MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl destroy data buffer failed.");
    acl_data_buffer = nullptr;
  }
  ret = svp_acl_mdl_destroy_dataset(*acl_mdl_dataset);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "acl destroy dataset failed.");
  *acl_mdl_dataset = nullptr;
  return RET_OK;
}

size_t GetDataTypeSize(svp_acl_mdl_desc *acl_mdl_desc, size_t index, AclDataInfo::DataMode data_mode) {
  svp_acl_data_type data_type;
  if (data_mode == AclDataInfo::Input) {
    data_type = svp_acl_mdl_get_input_data_type(acl_mdl_desc, index);
  } else {
    data_type = svp_acl_mdl_get_output_data_type(acl_mdl_desc, index);
  }
  return svp_acl_data_type_size(data_type) / kByteSize;
}

int ComputeValidDetectBoxes(svp_acl_mdl_desc *acl_mdl_desc, svp_acl_mdl_dataset *acl_outputs,
                            std::vector<std::vector<float>> *det_boxes) {
  MS_CHECK_TRUE_MSG(acl_mdl_desc != nullptr && acl_outputs != nullptr && det_boxes != nullptr, RET_ERROR,
                    "input params contain nullptr.");
  // yolo/ssd output 0 is num, output 1 is bbox
  enum InputOutputId { kInputImgId = 0, kOutputNumId = 0, kOutputBboxId = 1 };
  // get valid box number
  svp_acl_mdl_io_dims acl_dims;
  std::vector<int> valid_box_num;
  int ret = svp_acl_mdl_get_output_dims(acl_mdl_desc, kOutputNumId, &acl_dims);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS && acl_dims.dim_count >= 1, RET_ERROR, "svp get output dims failed.");
  svp_acl_data_buffer *data_buffer = svp_acl_mdl_get_dataset_buffer(acl_outputs, kOutputNumId);
  MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "get data buffer failed.");
  auto output_data = reinterpret_cast<float *>(svp_acl_get_data_buffer_addr(data_buffer));
  MS_CHECK_TRUE_MSG(output_data != nullptr, RET_ERROR, "data is nullptr.");
  for (uint32_t i = 0; i < static_cast<uint32_t>(acl_dims.dims[acl_dims.dim_count - 1]); i++) {
    valid_box_num.push_back(*(output_data + i));
  }
  int total_valid_num = std::accumulate(valid_box_num.begin(), valid_box_num.end(), 0);
  if (total_valid_num == 0) {
    MS_LOG(INFO) << "total valid num is zero";
    return RET_OK;
  }
  // get x y score
  data_buffer = svp_acl_mdl_get_dataset_buffer(acl_outputs, kOutputBboxId);
  MS_CHECK_TRUE_MSG(data_buffer != nullptr, RET_ERROR, "get data buffer failed.");
  output_data = reinterpret_cast<float *>(svp_acl_get_data_buffer_addr(data_buffer));
  MS_CHECK_TRUE_MSG(output_data != nullptr, RET_ERROR, "output_data is nullptr.");
  svp_acl_mdl_get_output_dims(acl_mdl_desc, kOutputBboxId, &acl_dims);
  size_t w_stride_offset = svp_acl_mdl_get_output_default_stride(acl_mdl_desc, kOutputBboxId) / sizeof(float);

  // box param include 6 part which is lx, ly, rx, ry, score, class id
  auto bbox_pararm_size = DetectBoxParam::kDetectBoxParamEnd - DetectBoxParam::kDetectBoxParamBegin;
  std::vector<float> bbox(bbox_pararm_size, 0.0f);
  for (int idx = 0; idx < total_valid_num; idx++) {
    float class_id = *(output_data + idx + kClassId * w_stride_offset);
    if (std::fabs(class_id) <= std::numeric_limits<float>::epsilon()) {
      continue;  // skip class 0 back ground
    }
    for (size_t i = 0; i < bbox_pararm_size; i++) {
      bbox[i] = (*(output_data + idx + i * w_stride_offset));
    }
    det_boxes->push_back(bbox);
  }
  std::sort(det_boxes->begin(), det_boxes->end(), Cmp);
  return RET_OK;
}

int WriteDetBoxesToTensorData(const std::vector<std::vector<float>> &det_boxes,
                              mindspore::MSTensor *detect_boxes_tensor) {
  size_t total_box_num = det_boxes.size();
  auto bbox_pararm_size = DetectBoxParam::kDetectBoxParamEnd - DetectBoxParam::kDetectBoxParamBegin;
  MS_CHECK_TRUE_MSG(detect_boxes_tensor != nullptr, RET_ERROR, "detect_boxes_tensor is nullptr.");
  MS_CHECK_TRUE_MSG(static_cast<size_t>(detect_boxes_tensor->ElementNum()) >= total_box_num * bbox_pararm_size,
                    RET_ERROR, "detect box tensor element num is too few");
  auto *bbox_tensor_data = reinterpret_cast<float *>(detect_boxes_tensor->MutableData());
  MS_CHECK_TRUE_MSG(bbox_tensor_data != nullptr, RET_ERROR, "bbox_tensor_data is nullptr");
  MS_CHECK_TRUE_MSG(total_box_num != 0, RET_ERROR, "total_box_num is 0");
  for (size_t i = 0; i < total_box_num; i++) {
    for (size_t bbox_param_idx = 0; bbox_param_idx < bbox_pararm_size; bbox_param_idx++) {
      bbox_tensor_data[bbox_param_idx * total_box_num + i] = det_boxes[i][bbox_param_idx];
    }
  }
  detect_boxes_tensor->SetShape({1, static_cast<int64_t>(total_box_num * bbox_pararm_size)});
  return RET_OK;
}

int AclMalloc(void **buf, size_t size) {
  int ret = svp_acl_rt_malloc(buf, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
  return ret;
}
int AclFree(void **buf) {
  int ret = svp_acl_rt_free(*buf);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp_acl_rt_free failed");
  *buf = nullptr;
  return ret;
}
}  // namespace lite
}  // namespace mindspore
