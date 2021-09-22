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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "ps/ps_context.h"
#include "fl/worker/fl_worker.h"

namespace mindspore {
namespace kernel {
// The duration between two PushWeights requests when return code is ResponseCode_SucNotReady.
constexpr int kRetryDurationOfPushWeights = 200;
template <typename T>
class FusedPushWeightKernel : public CPUKernel {
 public:
  FusedPushWeightKernel()
      : server_num_(0), indices_({}), weight_full_names_({}), fl_iteration_(0), total_iteration_(0) {}
  ~FusedPushWeightKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(DEBUG) << "Launch FusedPushWeightKernel.";
    if (inputs.size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but FusedPushWeightKernel needs "
                        << weight_full_names_.size() << " weights as inputs.";
    }

    std::shared_ptr<fl::FBBuilder> fbb = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb);

    total_iteration_++;
    uint64_t step_num_per_iteration = fl::worker::FLWorker::GetInstance().worker_step_num_per_iteration();
    if (step_num_per_iteration == 0) {
      MS_LOG(EXCEPTION) << "Step numbers of per iteration should not equal to 0";
    }
    // The worker has to train kWorkerTrainStepNum standalone iterations before it communicates with server.
    MS_LOG(INFO) << "Try to push weights. Local step number: " << total_iteration_
                 << ", step number needs to run per iteration: " << step_num_per_iteration;
    if (step_num_per_iteration != fl::kOneStepPerIteration &&
        total_iteration_ % step_num_per_iteration != fl::kTrainEndStepNum) {
      return true;
    }

    fl_iteration_++;
    MS_LOG(INFO) << "Launching pushing weight for federated learning iteration " << fl_iteration_;
    if (!BuildPushWeightReq(fbb, inputs)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
    }

    // The server number may change after scaling in/out.
    for (uint32_t i = 0; i < fl::worker::FLWorker::GetInstance().server_num(); i++) {
      std::shared_ptr<std::vector<unsigned char>> push_weight_rsp_msg = nullptr;
      const schema::ResponsePushWeight *push_weight_rsp = nullptr;
      int retcode = schema::ResponseCode_SucNotReady;
      while (retcode == schema::ResponseCode_SucNotReady) {
        if (!fl::worker::FLWorker::GetInstance().running()) {
          MS_LOG(WARNING) << "Worker has finished.";
          return true;
        }
        if (!fl::worker::FLWorker::GetInstance().SendToServer(i, fbb->GetBufferPointer(), fbb->GetSize(),
                                                              ps::core::TcpUserCommand::kPushWeight,
                                                              &push_weight_rsp_msg)) {
          MS_LOG(WARNING) << "Sending request for FusedPushWeight to server " << i << " failed.";
          retcode = schema::ResponseCode_SucNotReady;
          std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushWeights));
          continue;
        }
        MS_EXCEPTION_IF_NULL(push_weight_rsp_msg);

        push_weight_rsp = flatbuffers::GetRoot<schema::ResponsePushWeight>(push_weight_rsp_msg->data());
        retcode = push_weight_rsp->retcode();
        if (retcode == schema::ResponseCode_SucNotReady) {
          std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPushWeights));
          fl_iteration_ = push_weight_rsp->iteration();
          MS_LOG(DEBUG) << "Server is not ready for pushing weight yet. Reason: " << push_weight_rsp->reason()->str()
                        << ". Retry later.";
          if (!BuildPushWeightReq(fbb, inputs)) {
            MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
          }
          continue;
        } else if (retcode != schema::ResponseCode_SUCCEED) {
          MS_LOG(EXCEPTION) << "FusedPushWeight failed. Server return code: " << push_weight_rsp->retcode()
                            << ", reason: " << push_weight_rsp->reason()->str();
        } else {
          MS_LOG(DEBUG) << "FusedPushWeight succeed.";
        }
      }
    }

    MS_LOG(INFO) << "Push weights for " << weight_full_names_ << " success. Iteration: " << fl_iteration_;
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; i++) {
      auto weight_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      size_t weight_size_ = std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(T), std::multiplies<T>());
      input_size_list_.push_back(weight_size_);
    }

    server_num_ = ps::PSContext::instance()->server_num();
    indices_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrIndex);
    weight_full_names_ = AnfAlgo::GetNodeAttr<std::vector<std::string>>(kernel_node, kAttrPsKey);
    MS_LOG(INFO) << "Weight full name as key " << weight_full_names_ << ", key index is " << indices_
                 << ", server number is " << server_num_;
    if (server_num_ == 0 || weight_full_names_.empty() || indices_.empty()) {
      MS_LOG(EXCEPTION)
        << "Attributes of FusedPushWeightKernel are invalid: server number is 0 or weight_full_names_ is "
           "empty or indices_ is UINT32_MAX.";
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t i = 0; i < output_num; i++) {
      output_size_list_.push_back(sizeof(size_t));
    }
    return;
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  bool BuildPushWeightReq(std::shared_ptr<fl::FBBuilder> fbb, const std::vector<AddressPtr> &weights) {
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

    schema::RequestPushWeightBuilder req_push_weight_builder(*(fbb.get()));
    req_push_weight_builder.add_iteration(fl_iteration_);
    req_push_weight_builder.add_feature_map(fbs_feature_maps_vector);
    auto req_push_weight = req_push_weight_builder.Finish();
    fbb->Finish(req_push_weight);
    return true;
  }

  uint32_t server_num_;
  std::vector<int64_t> indices_;
  std::vector<std::string> weight_full_names_;
  size_t fl_iteration_;
  uint64_t total_iteration_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PUSH_WEIGHT_KERNEL_H_
