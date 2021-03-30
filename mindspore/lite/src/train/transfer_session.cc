/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/train/transfer_session.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/train/optimizer_kernel.h"
#include "src/sub_graph_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/runtime/runtime_api.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"
#include "nnacl/fp32/pack_fp32.h"

namespace mindspore {
namespace lite {

TransferSession::TransferSession(const char *model_buf_backbone, size_t size_backbone, lite::Context *context)
    : is_valid_(false) {
  lite_model_ = reinterpret_cast<char *>(malloc(size_backbone));
  if (lite_model_ != nullptr) {
    std::copy(model_buf_backbone, model_buf_backbone + size_backbone, lite_model_);
    backbone_session_ =
      reinterpret_cast<lite::LiteSession *>(session::LiteSession::CreateSession(lite_model_, size_backbone, context));
    if (backbone_session_ != nullptr) {
      is_valid_ = true;
    } else {
      MS_LOG(ERROR) << "transfer session: create backbone session failed";
    }
  }
}

std::vector<tensor::MSTensor *> TransferSession::GetInputs() const { return combined_inputs_; }

bool TransferSession::CompileFormatTransform(tensor::MSTensor *out, tensor::MSTensor *in, int *mask, size_t mask_len) {
  MS_ASSERT(out->shape().size() == mask_len);
  for (std::size_t dim = 0; dim != out->shape().size(); ++dim) {
    if (in->shape().at(mask[dim]) != out->shape().at(dim)) {
      return false;
    }
  }
  return true;
}

int TransferSession::CompileTransferGraph() {
  combined_inputs_ = backbone_session_->GetInputs();
  auto outputs_backbone = backbone_session_->GetOutputs();
  auto inputs_head = lite::TrainSession::GetInputs();

  int ret = RET_OK;
  for (auto input : inputs_head) {
    bool match = false;
    mindspore::tensor::MSTensor *output = nullptr;
    for (auto it = outputs_backbone.begin(); it != outputs_backbone.end(); ++it) {
      output = it->second;
      if (output->ElementsNum() == input->ElementsNum() && output->shape().size() == input->shape().size()) {
        match = true;
        for (std::size_t dim = 0; dim != output->shape().size(); ++dim) {
          if (input->shape().at(dim) != output->shape().at(dim)) {
            match = false;
            break;
          }
        }
        if (match == false && input->shape().size() == 4) {
          int nchw2nhwc_mask[4] = {0, 3, 1, 2};
          nchw2nhwc_ = CompileFormatTransform(output, input, nchw2nhwc_mask, 4);
          match = nchw2nhwc_;
        }
        if (true == match) {
          break;
        }
      }
    }
    if (true == match) {
      backbone_head_map_.push_back(std::make_pair(input, output));
    } else {
      combined_inputs_.push_back(input);
    }
  }
  if (0 == backbone_head_map_.size()) {
    ret = RET_ERROR;
  }
  return ret;
}

mindspore::tensor::MSTensor *TransferSession::GetInputsByTensorName(const std::string &tensor_name) const {
  /* First look in backbone netwok */
  auto ret = backbone_session_->GetInputsByTensorName(tensor_name);
  /* If not found look in head network */
  if (nullptr == ret) {
    ret = TrainSession::GetInputsByTensorName(tensor_name);
  }
  return ret;
}

TransferSession::~TransferSession() {
  if (backbone_session_ != nullptr) {
    delete backbone_session_;
    backbone_session_ = nullptr;
  }
  if (lite_model_ != nullptr) {
    free(lite_model_);
    lite_model_ = nullptr;
  }
}
void TransferSession::BindThread(bool if_bind) {
  backbone_session_->BindThread(if_bind);
  TrainSession::BindThread(if_bind);
}

int TransferSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  auto ret = backbone_session_->RunGraph(before, after);
  if (ret != RET_OK) {
    return ret;
  }
  for (auto &backbone_head_pair : backbone_head_map_) {
    auto input = backbone_head_pair.first;
    auto output = backbone_head_pair.second;
    char *input_data = reinterpret_cast<char *>(input->MutableData());
    char *output_data = reinterpret_cast<char *>(output->MutableData());
    if (nchw2nhwc_) {
      int plane = input->shape().at(1) * input->shape().at(2);
      int batch = input->shape().at(0);
      int channel = input->shape().at(3);
      PackNCHWToNHWCFp32(output_data, input_data, batch, plane, channel, 0, 1);
    } else {
      std::copy(output_data, output_data + output->Size(), input_data);
    }
  }
  ret = lite::TrainSession::RunGraph(before, after);
  return ret;
}

}  // namespace lite

session::TrainSession *session::TrainSession::CreateTransferSession(const char *model_buf_backbone,
                                                                    size_t size_backbone, const char *model_buf_head,
                                                                    size_t size_head, lite::Context *context,
                                                                    bool train_mode) {
  auto ValidModelSize = [](size_t size) -> bool {
    constexpr size_t MaxModelSize = 1024 * 1024 * 1024ULL;  // 1G B
    return size < MaxModelSize && size > 0;
  };
  if (!ValidModelSize(size_backbone)) {
    MS_LOG(ERROR) << "size_backbone too large: " << size_backbone;
    return nullptr;
  }
  if (!ValidModelSize(size_head)) {
    MS_LOG(ERROR) << "size_head too large: " << size_head;
    return nullptr;
  }
  auto session = new (std::nothrow) lite::TransferSession(model_buf_backbone, size_backbone, context);
  if (session == nullptr) {
    MS_LOG(ERROR) << "create transfer session failed";
    return nullptr;
  }
  if (!session->is_valid()) {
    MS_LOG(ERROR) << "create transfer session failed";
    delete session;
    return nullptr;
  }

  auto ret = session->Init(context);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "init transfer session failed";
    delete session;
    return nullptr;
  }

  auto model = lite::TrainModel::Import(model_buf_head, size_head);
  if (model == nullptr) {
    MS_LOG(ERROR) << "create model for head train session failed";
    delete session;
    return nullptr;
  }

  ret = session->CompileTrainGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compiling Train Graph failed";
    delete session;
    return nullptr;
  }
  ret = session->CompileTransferGraph();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compiling Transfer Graph failed";
    delete session;
    return nullptr;
  }

  if (train_mode) {
    ret = session->Train();
  } else {
    ret = session->Eval();
  }
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Could not switch to Train Mode " << train_mode;
    delete session;
    return nullptr;
  }
  return session;
}

session::TrainSession *session::TrainSession::CreateTransferSession(const std::string &filename_backbone,
                                                                    const std::string &filename_head,
                                                                    lite::Context *context, bool train_mode) {
  size_t size_head = -1;
  size_t size_backbone = -1;
  auto buf_head = lite::ReadFileToBuf(filename_head, &size_head);
  if (buf_head == nullptr) {
    return nullptr;
  }
  auto buf_backbone = lite::ReadFileToBuf(filename_backbone, &size_backbone);
  if (buf_backbone == nullptr) {
    return nullptr;
  }
  return session::TrainSession::CreateTransferSession(buf_backbone.get(), size_backbone, buf_head.get(), size_head,
                                                      context, train_mode);
}

}  // namespace mindspore
