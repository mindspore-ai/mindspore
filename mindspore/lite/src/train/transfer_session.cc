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
#include "src/common/file_utils.h"
#include "src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/train/optimizer_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/litert/executor.h"
#include "src/train/train_export.h"
#include "src/train/train_utils.h"

namespace mindspore {
namespace lite {
TransferSession::TransferSession(const char *model_buf_backbone, size_t size_backbone,
                                 const std::shared_ptr<lite::InnerContext> &context)
    : is_valid_(false) {
  lite_model_ = reinterpret_cast<char *>(malloc(size_backbone));
  size_backbone_ = size_backbone;
  if (lite_model_ != nullptr) {
    std::copy(model_buf_backbone, model_buf_backbone + size_backbone, lite_model_);
    backbone_session_ = LiteSession::CreateSession(lite_model_, size_backbone, context);
    if (backbone_session_ != nullptr) {
      is_valid_ = true;
    } else {
      MS_LOG(ERROR) << "transfer session: create backbone session failed";
    }
  }
}

std::vector<lite::Tensor *> TransferSession::GetInputs() const { return combined_inputs_; }

bool TransferSession::CompileFormatTransform(lite::Tensor *out, lite::Tensor *in, int *mask, size_t mask_len) {
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
    mindspore::lite::Tensor *output = nullptr;
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
        if (match) {
          break;
        }
      }
    }
    if (match) {
      backbone_head_map_.push_back(std::make_pair(input, output));
    } else {
      combined_inputs_.push_back(input);
    }
  }
  if (backbone_head_map_.size() == 0) {
    ret = RET_ERROR;
  }
  return ret;
}

mindspore::lite::Tensor *TransferSession::GetInputsByTensorName(const std::string &tensor_name) const {
  /* First look in backbone netwok */
  auto ret = backbone_session_->GetInputsByTensorName(tensor_name);
  /* If not found look in head network */
  if (ret == nullptr) {
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
    float *input_data = reinterpret_cast<float *>(input->MutableData());
    float *output_data = reinterpret_cast<float *>(output->MutableData());
    if (nchw2nhwc_) {
      int batch = input->shape().at(0);
      int plane = input->shape().at(1) * input->shape().at(2);
      int channel = input->shape().at(3);
      int img_size = plane * channel;
      for (int b = 0; b < batch; b++) {
        float *in = input_data + b * img_size;
        float *out = output_data + b * img_size;
        for (int p = 0; p < plane; p++) {
          for (int c = 0; c < channel; c++) {
            in[p * channel + c] = out[c * plane + p];
          }
        }
      }
    } else {
      std::copy(output_data, output_data + output->ElementsNum(), input_data);
    }
  }
  ret = lite::TrainSession::RunGraph(before, after);
  return ret;
}

std::unordered_map<size_t, size_t> TransferSession::ConnectionMap() {
  std::unordered_map<size_t, size_t> map;
  for (auto &backbone_head_pair : backbone_head_map_) {
    auto input = backbone_head_pair.first;
    auto output = backbone_head_pair.second;
    auto in_id = TSFindTensorByName(tensors_, input->tensor_name());
    if (in_id == tensors_.size()) {
      MS_LOG(ERROR) << "cannot find input tensor " << input->tensor_name();
      map.clear();
      return map;
    }
    auto out_id = TSFindTensorByName(backbone_session_->tensors_, output->tensor_name());
    if (out_id == backbone_session_->tensors_.size()) {
      MS_LOG(ERROR) << "cannot find input tensor " << output->tensor_name();
      map.clear();
      return map;
    }
    map[in_id] = out_id;
  }
  return map;
}

int TransferSession::Export(const std::string &filename, ModelType model_type, QuantizationType quant_type,
                            FormatType format, std::vector<std::string> out_put_tensor_name) {
  if (format != FT_FLATBUFFERS) {
    MS_LOG(ERROR) << "Currently only flatbuffer format is supported";
    return RET_ERROR;
  }

  if (model_type == MT_TRAIN) {
    return TrainSession::Export(filename, model_type, quant_type, format);
  }

  bool orig_train_state = IsTrain();
  if (Eval() != RET_OK) {
    MS_LOG(ERROR) << "eval failed.";
    return RET_ERROR;
  }
  TrainExport texport(filename);
  int status = texport.LoadModel(lite_model_, size_backbone_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot init export";
    return status;
  }
  auto connect_map = ConnectionMap();
  texport.set_connect(connect_map);
  if (nchw2nhwc_) {
    status = texport.AddTransformNode();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "cannot add transform node";
      return status;
    }
  }
  if (!out_put_tensor_name.empty() && model_type == MT_INFERENCE) {
    std::vector<kernel::KernelExec *> export_kernels = {};
    status = FindExportKernels(&export_kernels, out_put_tensor_name, inference_kernels_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "FindExportKernels failed.";
      return RET_ERROR;
    }
    status = texport.ExportNet(export_kernels, tensors_, out_put_tensor_name, model_.get(), quant_type,
                               backbone_session_->model_);
  } else {
    status = texport.ExportNet(inference_kernels_, tensors_, GetOutputTensorNames(), model_.get(), quant_type,
                               backbone_session_->model_);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot serialize head";
    return status;
  }
  status = texport.SaveToFile();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "failed to save to " << filename;
    return status;
  }
  if (orig_train_state) {
    auto ret = Train();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "train failed.";
      return RET_ERROR;
    }
  }
  return status;
}

lite::LiteSession *CreateTransferSessionInt(const char *model_buf_backbone, size_t size_backbone,
                                            const char *model_buf_head, size_t size_head,
                                            const std::shared_ptr<InnerContext> &context, bool train_mode,
                                            const lite::TrainCfg *cfg) {
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
  auto ret = session->TrainInit(context, cfg);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "init transfer session failed";
    delete session;
    return nullptr;
  }

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(model_buf_head, size_head));
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
}  // namespace lite
}  // namespace mindspore
