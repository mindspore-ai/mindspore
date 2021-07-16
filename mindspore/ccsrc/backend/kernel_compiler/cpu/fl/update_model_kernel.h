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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "fl/worker/fl_worker.h"

namespace mindspore {
namespace kernel {
class UpdateModelKernel : public CPUKernel {
 public:
  UpdateModelKernel() = default;
  ~UpdateModelKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(INFO) << "Launching client UpdateModelKernel";
    if (inputs.size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "Input number of UpdateModelKernel should be " << weight_full_names_.size() << ", but got "
                        << inputs.size();
      return false;
    }

    if (!WeightingData(inputs)) {
      MS_LOG(EXCEPTION) << "Weighting data with data_size failed.";
      return false;
    }

    if (!BuildUpdateModelReq(fbb_, inputs)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return false;
    }

    std::shared_ptr<std::vector<unsigned char>> update_model_rsp_msg = nullptr;
    if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(),
                                                          fbb_->GetSize(), ps::core::TcpUserCommand::kUpdateModel,
                                                          &update_model_rsp_msg)) {
      MS_LOG(EXCEPTION) << "Sending request for UpdateModel to server " << target_server_rank_ << " failed.";
      return false;
    }
    flatbuffers::Verifier verifier(update_model_rsp_msg->data(), update_model_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponseUpdateModel>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponseUpdateModel is invalid.";
      return false;
    }

    const schema::ResponseFLJob *update_model_rsp =
      flatbuffers::GetRoot<schema::ResponseFLJob>(update_model_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(update_model_rsp);
    auto response_code = update_model_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(EXCEPTION) << "Launching start fl job for worker failed. Reason: " << update_model_rsp->reason();
    }
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    MS_LOG(INFO) << "Initializing UpdateModel kernel";
    fbb_ = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);

    MS_EXCEPTION_IF_NULL(kernel_node);
    server_num_ = fl::worker::FLWorker::GetInstance().server_num();
    rank_id_ = fl::worker::FLWorker::GetInstance().rank_id();
    if (rank_id_ == UINT32_MAX) {
      MS_LOG(EXCEPTION) << "Federated worker is not initialized yet.";
      return;
    }
    target_server_rank_ = rank_id_ % server_num_;
    fl_name_ = fl::worker::FLWorker::GetInstance().fl_name();
    fl_id_ = fl::worker::FLWorker::GetInstance().fl_id();
    MS_LOG(INFO) << "Initializing StartFLJob kernel. fl_name: " << fl_name_ << ", fl_id: " << fl_id_
                 << ". Request will be sent to server " << target_server_rank_;

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; i++) {
      auto input_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(kernel_node, i), 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto weight_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(weight_node);
      std::string weight_name = weight_node->fullname_with_scope();
      MS_LOG(INFO) << "Parameter name is " << weight_name;
      weight_full_names_.push_back(weight_name);

      auto weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      size_t weight_size_ =
        std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(float), std::multiplies<float>());
      input_size_list_.push_back(weight_size_);
    }
    output_size_list_.push_back(sizeof(float));
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  bool BuildUpdateModelReq(const std::shared_ptr<fl::FBBuilder> &fbb, const std::vector<AddressPtr> &weights) {
    MS_EXCEPTION_IF_NULL(fbb_);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto fbs_fl_id = fbb->CreateString(fl_id_);
    std::vector<flatbuffers::Offset<schema::FeatureMap>> fbs_feature_maps;
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      const std::string &weight_name = weight_full_names_[i];
      auto fbs_weight_fullname = fbb->CreateString(weight_name);
      auto fbs_weight_data =
        fbb->CreateVector(reinterpret_cast<const float *>(weights[i]->addr), weights[i]->size / sizeof(float));
      auto fbs_feature_map = schema::CreateFeatureMap(*(fbb.get()), fbs_weight_fullname, fbs_weight_data);
      fbs_feature_maps.push_back(fbs_feature_map);
    }
    auto fbs_feature_maps_vector = fbb->CreateVector(fbs_feature_maps);

    schema::RequestUpdateModelBuilder req_update_model_builder(*(fbb.get()));
    req_update_model_builder.add_fl_name(fbs_fl_name);
    req_update_model_builder.add_fl_id(fbs_fl_id);
    iteration_ = fl::worker::FLWorker::GetInstance().fl_iteration_num();
    req_update_model_builder.add_iteration(SizeToInt(iteration_));
    req_update_model_builder.add_feature_map(fbs_feature_maps_vector);
    auto req_update_model = req_update_model_builder.Finish();
    fbb->Finish(req_update_model);
    return true;
  }

  bool WeightingData(const std::vector<AddressPtr> &inputs) {
    data_size_ = fl::worker::FLWorker::GetInstance().data_size();
    for (auto &input : inputs) {
      float *data = reinterpret_cast<float *>(input->addr);
      for (size_t i = 0; i < input->size / sizeof(float); i++) {
        data[i] *= data_size_;
      }
    }
    return true;
  }

  std::shared_ptr<fl::FBBuilder> fbb_;
  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_name_;
  std::string fl_id_;
  int data_size_;
  uint64_t iteration_;
  std::vector<std::string> weight_full_names_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_UPDATE_MODEL_H_
