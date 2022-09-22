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

#include "nnie/nnie_micro.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
#include "nnie/nnie_interfaces.h"
#include "src/proposal.h"
#include "include/ir/dtype/type_id.h"
#include "include/c_api/status_c.h"

namespace mindspore {
namespace {
using nnie::NnieDataType;
using nnie::NnieTensors;
constexpr auto ENV_TIME_STEP = "TIME_STEP";
constexpr auto ENV_MAX_ROI_NUM = "MAX_ROI_NUM";
constexpr int kNumInput2 = 2;
constexpr int kDefaultROINum = 300;
constexpr int kNCHWDims = 4;
constexpr int kNCHWFormatH = 2;
constexpr int kNCHWFormatW = 3;
constexpr int kNCHWFormatC = 1;
static std::map<int, size_t> data_type_size_map = {{mindspore::kNumberTypeInt8, sizeof(int8_t)},
                                                   {mindspore::kNumberTypeUInt8, sizeof(uint8_t)},
                                                   {mindspore::kNumberTypeFloat32, sizeof(float)}};

int MakeTensorList(TensorC *tensors, int tensor_num, NnieTensors *tensor_list) {
  if (tensor_num > SVP_NNIE_MAX_INPUT_NUM) {
    printf("tensors' number is larger than 16\n");
    return kMSStatusLiteError;
  }
  tensor_list->size_ = tensor_num;
  for (int i = 0; i < tensor_num; ++i) {
    tensor_list->data_[i] = tensors[i].data_;
    tensor_list->shape_[i] = tensors[i].shape_;
    tensor_list->shape_len_[i] = tensors[i].shape_size_;
    tensor_list->name_[i] = tensors[i].name_;
    switch (tensors[i].data_type_) {
      case mindspore::kNumberTypeInt8:
        tensor_list->dtype_[i] = NnieDataType::NNIE_INT8;
        break;
      case mindspore::kNumberTypeUInt8:
        tensor_list->dtype_[i] = NnieDataType::NNIE_UINT8;
        break;
      case mindspore::kNumberTypeFloat32:
        tensor_list->dtype_[i] = NnieDataType::NNIE_FLOAT32;
        break;
      default:
        printf("The tensor's data type is unsupported, %d\n", tensors[i].data_type_);
        return kMSStatusLiteError;
    }
  }
  return 0;
}

static bool GetIntCustomAttr(const char *key, int *value, CustomParameter *param) {
  for (int i = 0; i < param->attr_num; ++i) {
    if (!strcmp(param->attr_name[i], key)) {
      *value = atoi(param->attr_data[i]);
      return true;
    }
  }
  return false;
}

static int GetIntEnv(const char *env_key, int default_data) {
  auto *env_data = std::getenv(env_key);
  int result = default_data;
  if (env_data != nullptr) {
    auto iter = std::find_if(env_data, env_data + strlen(env_data), [](char val) { return val < '0' || val > '9'; });
    if (iter != env_data) {
      *iter = '\0';
      result = atoi(env_data);
    } else {
      printf("%s ENV is invalid, now set to default value %d", env_key, default_data);
    }
  } else {
    printf("%s ENV is invalid, now set to default value %d", env_key, default_data);
  }
  return result;
}
}  // namespace
namespace nnie {
static int NnieKernel(TensorC *inputs, int input_num, TensorC *outputs, int output_num, CustomParameter *param) {
  int id;
  if (!GetIntCustomAttr("id", &id, param)) {
    printf("Not find the id attr!\n");
    return kMSStatusLiteError;
  }

  static NnieHandle handle = {
    .load_model_ = 0,
    .roi_used_ = 0,
  };
  handle.model_buf_ = reinterpret_cast<char *>(inputs[input_num - 1].data_);
  if (data_type_size_map.find(inputs[input_num - 1].data_type_) == data_type_size_map.end()) {
    printf("Unsupported data type: %d\n", inputs[input_num - 1].data_type_);
    return kMSStatusLiteError;
  }
  size_t data_type_size = data_type_size_map.at(inputs[input_num - 1].data_type_);
  handle.buf_size_ =
    std::accumulate(inputs[input_num - 1].shape_, inputs[input_num - 1].shape_ + inputs[input_num - 1].shape_size_,
                    data_type_size, std::multiplies<int>());
  handle.cfg_.run_idx_.seg_idx_ = id;
  NnieTensors input_list;
  if (MakeTensorList(inputs, input_num - 1, &input_list)) return kMSStatusLiteError;
  if (!handle.load_model_) {
    handle.cfg_.cfg_.max_roi_num_ = GetIntEnv(ENV_MAX_ROI_NUM, kDefaultROINum);
    handle.cfg_.cfg_.step_ = GetIntEnv(ENV_TIME_STEP, 1);
    if (NnieInit(&handle, &input_list) != HI_SUCCESS) return kMSStatusLiteError;
    handle.load_model_ = 1;
  }
  if (NnieFillData(&handle, &input_list) != HI_SUCCESS) return kMSStatusLiteError;
  NnieTensors output_list;
  if (MakeTensorList(outputs, output_num, &output_list)) return kMSStatusLiteError;
  if (NnieRun(&handle, &output_list) != HI_SUCCESS) return kMSStatusLiteError;
  return 0;
}
}  // namespace nnie

namespace proposal {
static int ProposalKernel(TensorC *inputs, int input_num, TensorC *outputs, int output_num, CustomParameter *param) {
  int ndims, image_height, image_width;
  if (input_num != kNumInput2) {
    printf("inputs tensor num error.\n");
    return kMSStatusLiteError;
  }
  if (output_num != 1) {
    LOGE("outputs tensor num error.");
    return kMSStatusLiteError;
  }
  if (!GetIntCustomAttr("proposal_id", &ndims, param)) {
    printf("Can't find the proposal_id attr!\n");
    return kMSStatusLiteError;
  }
  if (!GetIntCustomAttr("image_height", &image_height, param)) {
    printf("Can't find the image_height attr!\n");
    return kMSStatusLiteError;
  }
  if (!GetIntCustomAttr("image_width", &image_width, param)) {
    printf("Can't find the image_width attr!\n");
    return kMSStatusLiteError;
  }
  int max_roi_num_int = GetIntEnv(ENV_MAX_ROI_NUM, kDefaultROINum);
  ProposalParam pparam;
  memset(&pparam, 0, sizeof(ProposalParam));

  std::vector<std::string> proposal_input{"rpn_cls_score", "rpn_bbox_pred"};
  TensorC *reorder_inputs[kNumInput2];
  for (size_t i = 0; i < proposal_input.size(); ++i) {
    for (int j = 0; j < input_num; ++j) {
      if (proposal_input[i] == inputs[j].name_) {
        reorder_inputs[i] = &inputs[j];
        break;
      }
    }
  }
  for (int i = 0; i < input_num; i++) {
    auto ptr_shape = reorder_inputs[i]->shape_;
    if ((reorder_inputs[i]->shape_size_ == kNCHWDims)) {
      pparam.inputs_height_[i] = ptr_shape[kNCHWFormatH];
      pparam.inputs_width_[i] = ptr_shape[kNCHWFormatW];
      pparam.inputs_channel_[i] = ptr_shape[kNCHWFormatC];
      if (i == 0) {
        pparam.inputs_stride_ = ptr_shape[kNCHWFormatW] * sizeof(float);
      }
    } else {
      printf("proposal only support input shape size == 4.\n");
      return kMSStatusLiteError;
    }
  }
  if (ProposalInit(&pparam, max_roi_num_int, image_height, image_width)) {
    printf("proposal init failed!\n");
    return kMSStatusLiteError;
  }
  for (int i = 0; i < kNumInput2; i++) {
    pparam.inputs_[i] = reinterpret_cast<float *>(reorder_inputs[i]->data_);
  }
  pparam.rpn_bounding_box_.data_ = outputs[0].data_;
  if (ProposalRun(&pparam)) {
    printf("proposal run failed!\n");
    return kMSStatusLiteError;
  }

  ProposalDeInit(&pparam);
  return 0;
}
}  // namespace proposal
}  // namespace mindspore

int CustomKernel(TensorC *inputs, int input_num, TensorC *outputs, int output_num, CustomParameter *param) {
  if (!strcmp(param->type, "NNIE")) {
    return mindspore::nnie::NnieKernel(inputs, input_num, outputs, output_num, param);
  } else if (!strcmp(param->type, "Proposal")) {
    return mindspore::proposal::ProposalKernel(inputs, input_num, outputs, output_num, param);
  } else {
    printf("Unknown custom op type: %s\n", param->type);
    return kMSStatusLiteError;
  }
}
