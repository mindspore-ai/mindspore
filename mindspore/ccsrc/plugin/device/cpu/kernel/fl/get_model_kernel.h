/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "fl/worker/fl_cloud_worker.h"
#include "ps/core/comm_util.h"

namespace mindspore {
namespace kernel {
constexpr size_t kRetryTimesOfGetModel = 512;
constexpr size_t kSleepMillisecondsOfGetModel = 1000;
class GetModelKernelMod : public NativeCpuKernelMod {
 public:
  GetModelKernelMod() = default;
  ~GetModelKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(INFO) << "Launching client GetModelKernelMod";
    if (!BuildGetModelReq(fbb_)) {
      MS_LOG(EXCEPTION) << "Building request for FusedPushWeight failed.";
    }
    bool get_model_success = false;
    std::map<std::string, size_t> weight_name_to_input_idx_tmp = weight_name_to_input_idx_;
    fl::worker::FLCloudWorker::GetInstance().RegisterMessageCallback(
      kernel_path_, [&](const std::shared_ptr<std::vector<unsigned char>> &response_msg) {
        flatbuffers::Verifier verifier(response_msg->data(), response_msg->size());
        if (!verifier.VerifyBuffer<schema::ResponseGetModel>()) {
          MS_LOG(DEBUG) << "The schema of response message is invalid.";
          return;
        }
        const schema::ResponseGetModel *get_model_rsp =
          flatbuffers::GetRoot<schema::ResponseGetModel>(response_msg->data());
        MS_ERROR_IF_NULL_WO_RET_VAL(get_model_rsp);
        auto response_code = get_model_rsp->retcode();
        if (response_code == schema::ResponseCode_SUCCEED) {
          MS_LOG(INFO) << "Get model from server successful.";
          get_model_success = true;
        } else if (response_code == schema::ResponseCode_SucNotReady) {
          MS_LOG(INFO) << "Get model response code from server is not ready.";
          return;
        } else {
          MS_LOG(ERROR) << "Launching get model for worker failed. Reason: " << get_model_rsp->reason();
        }

        auto feature_map = get_model_rsp->feature_map();
        MS_EXCEPTION_IF_NULL(feature_map);
        if (feature_map->size() == 0) {
          MS_LOG(ERROR) << "Feature map after GetModel is empty.";
          return;
        }
        MS_LOG(INFO) << "weight_name_to_input_idx_tmp size is " << weight_name_to_input_idx_tmp.size();
        for (size_t i = 0; i < feature_map->size(); i++) {
          std::string weight_full_name = feature_map->Get(i)->weight_fullname()->str();
          float *weight_data = const_cast<float *>(feature_map->Get(i)->data()->data());
          size_t weight_size = feature_map->Get(i)->data()->size() * sizeof(float);
          if (weight_name_to_input_idx_tmp.count(weight_full_name) == 0) {
            MS_LOG(ERROR) << "Weight " << weight_full_name << " doesn't exist in FL worker.";
            return;
          }
          MS_LOG(INFO) << "Cover weight " << weight_full_name << " by the model in server.";
          size_t index = weight_name_to_input_idx_tmp[weight_full_name];
          int ret = memcpy_s(inputs[index]->addr, inputs[index]->size, weight_data, weight_size);
          if (ret != 0) {
            MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
          }
        }
        return;
      });

    size_t retryTimes = 0;
    while (!get_model_success && retryTimes < kRetryTimesOfGetModel) {
      if (!fl::worker::FLCloudWorker::GetInstance().SendToServerSync(kernel_path_, HTTP_CONTENT_TYPE_URL_ENCODED,
                                                                     fbb_->GetBufferPointer(), fbb_->GetSize())) {
        MS_LOG(WARNING) << "Sending request for GetModel to server failed.";
        break;
      }
      retryTimes += 1;
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMillisecondsOfGetModel));
    }
    if (!get_model_success) {
      MS_LOG(WARNING) << "Get model from server failed.";
    }
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    fbb_ = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);

    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_path_ = "/getModel";
    fl_name_ = fl::worker::FLCloudWorker::GetInstance().fl_name();
    MS_LOG(INFO) << "Initializing GetModel kernel. fl_name: " << fl_name_;

    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; i++) {
      auto input_node =
        common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(kernel_node, i), 0).first;
      MS_EXCEPTION_IF_NULL(input_node);
      auto weight_node = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(weight_node);
      std::string weight_name = weight_node->fullname_with_scope();
      MS_LOG(INFO) << "Parameter name is " << weight_name;
      weight_name_to_input_idx_.insert(std::make_pair(weight_name, i));

      auto weight_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      size_t weight_size_ =
        std::accumulate(weight_shape.begin(), weight_shape.end(), sizeof(float), std::multiplies<float>());
      input_size_list_.push_back(weight_size_);
    }
    output_size_list_.push_back(sizeof(float));
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

  void InitSizeLists() { return; }

 private:
  bool BuildGetModelReq(const std::shared_ptr<fl::FBBuilder> &fbb) {
    MS_EXCEPTION_IF_NULL(fbb_);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto time = ps::core::CommUtil::GetNowTime();
    MS_LOG(INFO) << "now time: " << time.time_str_mill;
    auto fbs_timestamp = fbb->CreateString(std::to_string(time.time_stamp));
    schema::RequestGetModelBuilder req_get_model_builder(*(fbb.get()));
    req_get_model_builder.add_fl_name(fbs_fl_name);
    req_get_model_builder.add_timestamp(fbs_timestamp);
    iteration_ = fl::worker::FLCloudWorker::GetInstance().fl_iteration_num();
    req_get_model_builder.add_iteration(SizeToInt(iteration_));
    auto req_get_model = req_get_model_builder.Finish();
    fbb->Finish(req_get_model);
    return true;
  }

  std::shared_ptr<fl::FBBuilder> fbb_;
  std::string kernel_path_;
  std::string fl_name_;
  uint64_t iteration_;
  std::map<std::string, size_t> weight_name_to_input_idx_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_MODEL_H_
