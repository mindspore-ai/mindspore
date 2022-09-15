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

#include "src/proposal_fp32.h"
#include <memory>
#include <string>
#include <algorithm>
#include "schema/model_generated.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace proposal {
constexpr int kMaxSize = 1024;
constexpr int kNumInput2 = 2;
constexpr int kDecimal = 10;
constexpr auto kMazRoiNum = "MaxROINum";
constexpr int kNCHWDims = 4;
constexpr int kNCHWFormatH = 2;
constexpr int kNCHWFormatW = 3;
constexpr int kNCHWFormatC = 1;
bool IsValidUnsignedNum(const std::string &num_str) {
  return !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit);
}

void PrintInvalidChar(const std::string &key, const std::string &dat) {
  auto message = key + " configuration contains invalid characters: \'" + dat + "\'";
  LOGE(message.c_str());
}

int ProposalCPUKernel::Prepare() {
  if (inputs_.size() < kNumInput2) {
    LOGE("inputs tensor num error.");
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    LOGE("outputs tensor num error.");
    return RET_ERROR;
  }
  std::vector<std::string> inputs_name = {"rpn_cls_score", "rpn_bbox_pred"};
  std::vector<mindspore::MSTensor> inputs;
  for (size_t i = 0; i < inputs_name.size(); i++) {
    bool find_flag = false;
    for (auto &input : inputs_) {
      if (input.Name() == inputs_name[i]) {
        inputs.push_back(input);
        find_flag = true;
        break;
      }
    }
    if (!find_flag) {
      for (auto &input : inputs_) {
        if (std::find(inputs.begin(), inputs.end(), input) != inputs.end()) {
          continue;
        }
        inputs.push_back(input);
        LOGW("input tensor name diff '%s' vs '%s'.", inputs_name[i].c_str(), input.Name().c_str());
        break;
      }
    }
  }
  if (inputs.size() != inputs_name.size()) {
    LOGE("inputs size error.");
    return RET_ERROR;
  }
  this->set_inputs(inputs);
  if (inputs[0].Shape()[0] != 1) {
    LOGE("proposal only support input num == 1.");
    return RET_ERROR;
  }

  outputs_[0].SetTensorName("proposal");

  int max_roi_num_int = 300;
  auto nnie_arg = GetConfig("nnie");
  if (nnie_arg.find(kMazRoiNum) != nnie_arg.end()) {
    if (IsValidUnsignedNum(nnie_arg.at(kMazRoiNum)) == true) {
      max_roi_num_int = stoi(nnie_arg.at(kMazRoiNum));
    } else {
      PrintInvalidChar(kMazRoiNum, nnie_arg.at(kMazRoiNum));
      return RET_ERROR;
    }
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    auto ptr_shape = inputs[i].Shape();
    if ((ptr_shape.size() == kNCHWDims)) {
      proposal_param_.inputs_height_[i] = ptr_shape[kNCHWFormatH];
      proposal_param_.inputs_width_[i] = ptr_shape[kNCHWFormatW];
      proposal_param_.inputs_channel_[i] = ptr_shape[kNCHWFormatC];
      if (i == 0) {
        proposal_param_.inputs_stride_ = ptr_shape[kNCHWFormatW] * sizeof(float);
      }
    } else {
      LOGE("proposal only support input shape size == 4.");
      return RET_ERROR;
    }
  }
  return ProposalInit(&proposal_param_, max_roi_num_int, image_height_, image_weight_);
}

int ProposalCPUKernel::ReSize() {
  if (inputs_[0].Shape()[0] != 1) {
    LOGE("proposal only support input num == 1.");
    return RET_ERROR;
  }
  return RET_OK;
}

int ProposalCPUKernel::Execute() {
  for (int i = 0; i < kNumInput2; i++) {
    proposal_param_.inputs_[i] = reinterpret_cast<float *>(inputs_[i].MutableData());
  }
  if (ProposalRun(&proposal_param_) != RET_OK) {
    LOGE("ProposalRun error.");
    return RET_ERROR;
  }
  std::vector<int64_t> shape{static_cast<int64_t>(proposal_param_.rpn_bounding_box_.height_), COORDI_NUM};
  outputs_[0].SetShape(shape);
  auto output_data = outputs_[0].MutableData();
  memcpy(output_data, proposal_param_.rpn_bounding_box_.data_,
         proposal_param_.rpn_bounding_box_.height_ * COORDI_NUM * sizeof(float));
  return RET_OK;
}

ProposalCPUKernel::~ProposalCPUKernel() { ProposalDeInit(&proposal_param_); }

bool GetCustomAttr(char *buf, int buf_size, const mindspore::schema::Custom *op, const std::string &attr) {
  int attr_size;
  for (size_t i = 0; i < op->attr()->size(); i++) {
    if (op->attr()->Get(i)->name()->str() == attr) {
      auto output_info = op->attr()->Get(i)->data();
      attr_size = static_cast<int>(output_info->size());
      if (attr_size >= buf_size) {
        LOGE("attr size too big");
        return false;
      }
      for (int j = 0; j < attr_size; j++) {
        buf[j] = static_cast<char>(output_info->Get(j));
      }
      buf[attr_size] = 0;
      return true;
    }
  }
  return false;
}

std::shared_ptr<mindspore::kernel::Kernel> ProposalCreateKernel(const std::vector<mindspore::MSTensor> &inputs,
                                                                const std::vector<mindspore::MSTensor> &outputs,
                                                                const mindspore::schema::Primitive *primitive,
                                                                const mindspore::Context *ctx) {
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    LOGE("Primitive type is not PrimitiveType_Custom");
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  if (op->attr()->size() < 1) {
    LOGE("There are at least 1 attribute of Custom");
    return nullptr;
  }
  int64_t ndims;
  int64_t image_height;
  int64_t image_width;

  char *res = nullptr;
  char buf[kMaxSize];
  if (GetCustomAttr(buf, kMaxSize, op, "proposal_id")) {
    res = nullptr;
    ndims = strtol(buf, &res, kDecimal);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have id");
    return nullptr;
  }

  if (GetCustomAttr(buf, kMaxSize, op, "image_height")) {
    res = nullptr;
    image_height = strtol(buf, &res, kDecimal);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have image_height");
    return nullptr;
  }
  if (GetCustomAttr(buf, kMaxSize, op, "image_width")) {
    res = nullptr;
    image_width = strtol(buf, &res, kDecimal);
    if ((*res) != 0) {
      LOGE("Get attr id data fail");
      return nullptr;
    }
  } else {
    LOGE("Proposal Custom op should have image_width");
    return nullptr;
  }

  auto kernel = std::make_shared<ProposalCPUKernel>(inputs, outputs, primitive, ctx, ndims, image_height, image_width);
  if (kernel == nullptr) {
    LOGE("new custom kernel is nullptr");
    return nullptr;
  }
  return kernel;
}
}  // namespace proposal
}  // namespace mindspore

namespace mindspore {
namespace kernel {
namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
}
REGISTER_CUSTOM_KERNEL(CPU, NNIE, kFloat32, Proposal, proposal::ProposalCreateKernel)
}  // namespace kernel
}  // namespace mindspore
