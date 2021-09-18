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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "schema/fl_job_generated.h"
#include "ps/ps_context.h"
#include "fl/worker/fl_worker.h"

namespace mindspore {
namespace kernel {
// The duration between two PullWeights requests when return code is ResponseCode_SucNotReady.
constexpr int kRetryDurationOfPullWeights = 200;
template <typename T>
class FusedPullWeightKernel : public CPUKernel {
 public:
  FusedPullWeightKernel()
      : server_num_(0), indices_({}), weight_full_names_({}), fl_iteration_(0), total_iteration_(0) {}
  ~FusedPullWeightKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(DEBUG) << "Launch FusedPullWeightKernel.";
    if (inputs.size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but FusedPullWeightKernel needs "
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
    MS_LOG(INFO) << "Try to pull weights. Local step number: " << total_iteration_
                 << ", step number needs to run per iteration: " << step_num_per_iteration;
    if (step_num_per_iteration != fl::kOneStepPerIteration &&
        total_iteration_ % step_num_per_iteration != fl::kTrainBeginStepNum) {
      return true;
    }

    fl_iteration_++;
    MS_LOG(INFO) << "Launching pulling weight for federated learning iteration " << fl_iteration_;
    if (!BuildPullWeightReq(fbb)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPullWeight failed.";
    }

    std::shared_ptr<std::vector<unsigned char>> pull_weight_rsp_msg = nullptr;
    const schema::ResponsePullWeight *pull_weight_rsp = nullptr;
    int retcode = schema::ResponseCode_SucNotReady;
    while (retcode == schema::ResponseCode_SucNotReady) {
      if (!fl::worker::FLWorker::GetInstance().running()) {
        MS_LOG(WARNING) << "Worker has finished.";
        return true;
      }
      if (!fl::worker::FLWorker::GetInstance().SendToServer(
            0, fbb->GetBufferPointer(), fbb->GetSize(), ps::core::TcpUserCommand::kPullWeight, &pull_weight_rsp_msg)) {
        MS_LOG(WARNING) << "Sending request for FusedPullWeight to server 0 failed. Retry later.";
        retcode = schema::ResponseCode_SucNotReady;
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPullWeights));
        continue;
      }
      MS_EXCEPTION_IF_NULL(pull_weight_rsp_msg);

      pull_weight_rsp = flatbuffers::GetRoot<schema::ResponsePullWeight>(pull_weight_rsp_msg->data());
      retcode = pull_weight_rsp->retcode();
      if (retcode == schema::ResponseCode_SucNotReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryDurationOfPullWeights));
        fl_iteration_ = pull_weight_rsp->iteration();
        MS_LOG(DEBUG) << "Server is not ready for downloading yet. Reason: " << pull_weight_rsp->reason()->str()
                      << ". Retry later.";
        // Recreate fbb to avoid memory leak of FlatBuffers.
        fbb = std::make_shared<fl::FBBuilder>();
        if (!BuildPullWeightReq(fbb)) {
          MS_LOG(EXCEPTION) << "Building request for FusedDownloadWeightsByKeys failed.";
        }
        continue;
      } else if (retcode != schema::ResponseCode_SUCCEED) {
        MS_LOG(EXCEPTION) << "FusedPullWeight failed. Server return code: " << pull_weight_rsp->retcode()
                          << ", reason: " << pull_weight_rsp->reason()->str();
      } else {
        MS_LOG(DEBUG) << "FusedPullWeight succeed.";
      }
    }

    auto feature_map = ParseFeatureMap(pull_weight_rsp);
    for (size_t i = 0; i < weight_full_names_.size(); i++) {
      const std::string &weight_name = weight_full_names_[i];
      if (feature_map.count(weight_name) == 0) {
        MS_LOG(EXCEPTION) << "The weights for " << weight_name << " is not pulled from server.";
      }
      int ret =
        memcpy_s(inputs[i]->addr, inputs[i]->size, feature_map[weight_name].addr, feature_map[weight_name].size);
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      }
    }
    MS_LOG(INFO) << "Pull weights for " << weight_full_names_ << " success. Iteration: " << fl_iteration_;
    fl::worker::FLWorker::GetInstance().SetIterationRunning();
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
        << "Attributes of FusedPullWeightKernel are invalid: server number is 0 or weight_full_names_ is "
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
  bool BuildPullWeightReq(std::shared_ptr<fl::FBBuilder> fbb) {
    MS_EXCEPTION_IF_NULL(fbb);
    std::vector<flatbuffers::Offset<flatbuffers::String>> fbs_weight_names;
    for (const std::string &weight_name : weight_full_names_) {
      auto fbs_weight_name = fbb->CreateString(weight_name);
      fbs_weight_names.push_back(fbs_weight_name);
    }
    auto fbs_weight_names_vector = fbb->CreateVector(fbs_weight_names);

    schema::RequestPullWeightBuilder req_pull_weight_builder(*(fbb.get()));
    req_pull_weight_builder.add_iteration(fl_iteration_);
    req_pull_weight_builder.add_weight_names(fbs_weight_names_vector);
    auto req_pull_weight = req_pull_weight_builder.Finish();
    fbb->Finish(req_pull_weight);
    return true;
  }

  std::map<std::string, Address> ParseFeatureMap(const schema::ResponsePullWeight *pull_weight_rsp) {
    MS_EXCEPTION_IF_NULL(pull_weight_rsp);
    auto fbs_feature_map = pull_weight_rsp->feature_map();
    if (fbs_feature_map->size() != weight_full_names_.size()) {
      MS_LOG(EXCEPTION) << "FusedPullWeightKernel should get " << weight_full_names_.size() << " weights, but got "
                        << fbs_feature_map->size() << " weights.";
    }

    std::map<std::string, Address> feature_map;
    for (size_t i = 0; i < fbs_feature_map->size(); i++) {
      std::string weight_full_name = fbs_feature_map->Get(i)->weight_fullname()->str();
      float *weight_data = const_cast<float *>(fbs_feature_map->Get(i)->data()->data());
      size_t weight_size = fbs_feature_map->Get(i)->data()->size() * sizeof(float);
      feature_map[weight_full_name] = {weight_data, weight_size};
    }
    return feature_map;
  }

  uint32_t server_num_;
  std::vector<int64_t> indices_;
  std::vector<std::string> weight_full_names_;
  uint64_t fl_iteration_;
  uint64_t total_iteration_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PS_FUSED_PULL_WEIGHT_KERNEL_H_
