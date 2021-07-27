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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "fl/worker/fl_worker.h"

namespace mindspore {
namespace kernel {
class GetModelKernel : public CPUKernel {
 public:
  GetModelKernel() = default;
  ~GetModelKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(INFO) << "Launching client GetModelKernel";
    if (!BuildGetModelReq(fbb_, inputs)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
      return false;
    }

    const schema::ResponseGetModel *get_model_rsp = nullptr;
    std::shared_ptr<std::vector<unsigned char>> get_model_rsp_msg = nullptr;
    int response_code = schema::ResponseCode_SucNotReady;
    while (response_code == schema::ResponseCode_SucNotReady) {
      if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(),
                                                            fbb_->GetSize(), ps::core::TcpUserCommand::kGetModel,
                                                            &get_model_rsp_msg)) {
        MS_LOG(EXCEPTION) << "Sending request for GetModel to server " << target_server_rank_ << " failed.";
        return false;
      }
      flatbuffers::Verifier verifier(get_model_rsp_msg->data(), get_model_rsp_msg->size());
      if (!verifier.VerifyBuffer<schema::ResponseGetModel>()) {
        MS_LOG(EXCEPTION) << "The schema of ResponseGetModel is invalid.";
        return false;
      }

      get_model_rsp = flatbuffers::GetRoot<schema::ResponseGetModel>(get_model_rsp_msg->data());
      MS_EXCEPTION_IF_NULL(get_model_rsp);
      response_code = get_model_rsp->retcode();
      if (response_code == schema::ResponseCode_SUCCEED) {
        break;
      } else if (response_code == schema::ResponseCode_SucNotReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        continue;
      } else {
        MS_LOG(EXCEPTION) << "Launching get model for worker failed. Reason: " << get_model_rsp->reason();
      }
    }

    auto feature_map = get_model_rsp->feature_map();
    MS_EXCEPTION_IF_NULL(feature_map);
    if (feature_map->size() == 0) {
      MS_LOG(EXCEPTION) << "Feature map after GetModel is empty.";
      return false;
    }
    for (size_t i = 0; i < feature_map->size(); i++) {
      std::string weight_full_name = feature_map->Get(i)->weight_fullname()->str();
      float *weight_data = const_cast<float *>(feature_map->Get(i)->data()->data());
      size_t weight_size = feature_map->Get(i)->data()->size() * sizeof(float);
      if (weight_name_to_input_idx_.count(weight_full_name) == 0) {
        MS_LOG(EXCEPTION) << "Weight " << weight_full_name << " doesn't exist in FL worker.";
        return false;
      }
      MS_LOG(INFO) << "Cover weight " << weight_full_name << " by the model in server.";
      size_t index = weight_name_to_input_idx_[weight_full_name];
      int ret = memcpy_s(inputs[index]->addr, inputs[index]->size, weight_data, weight_size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
    }
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    MS_LOG(INFO) << "Initializing GetModel kernel";
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
    MS_LOG(INFO) << "Initializing GetModel kernel. fl_name: " << fl_name_ << ". Request will be sent to server "
                 << target_server_rank_;

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; i++) {
      auto input_node = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(kernel_node, i), 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto weight_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(weight_node);
      std::string weight_name = weight_node->fullname_with_scope();
      MS_LOG(INFO) << "Parameter name is " << weight_name;
      weight_name_to_input_idx_.insert(std::make_pair(weight_name, i));

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
  bool BuildGetModelReq(const std::shared_ptr<fl::FBBuilder> &fbb, const std::vector<AddressPtr> &weights) {
    MS_EXCEPTION_IF_NULL(fbb_);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    schema::RequestGetModelBuilder req_get_model_builder(*(fbb.get()));
    req_get_model_builder.add_fl_name(fbs_fl_name);
    iteration_ = fl::worker::FLWorker::GetInstance().fl_iteration_num();
    req_get_model_builder.add_iteration(SizeToInt(iteration_));
    auto req_get_model = req_get_model_builder.Finish();
    fbb->Finish(req_get_model);
    return true;
  }

  std::shared_ptr<fl::FBBuilder> fbb_;
  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_name_;
  uint64_t iteration_;
  std::map<std::string, size_t> weight_name_to_input_idx_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
