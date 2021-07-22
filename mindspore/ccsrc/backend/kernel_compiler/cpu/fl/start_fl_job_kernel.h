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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_START_FL_JOB_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_START_FL_JOB_H_

#include <vector>
#include <string>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "fl/worker/fl_worker.h"

namespace mindspore {
namespace kernel {
class StartFLJobKernel : public CPUKernel {
 public:
  StartFLJobKernel() = default;
  ~StartFLJobKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    MS_LOG(INFO) << "Launching client StartFLJobKernel";
    if (!BuildStartFLJobReq(fbb_)) {
      MS_LOG(EXCEPTION) << "Building request for StartFLJob failed.";
      return false;
    }

    std::shared_ptr<std::vector<unsigned char>> start_fl_job_rsp_msg = nullptr;
    if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(),
                                                          fbb_->GetSize(), ps::core::TcpUserCommand::kStartFLJob,
                                                          &start_fl_job_rsp_msg)) {
      MS_LOG(EXCEPTION) << "Sending request for StartFLJob to server " << target_server_rank_ << " failed.";
      return false;
    }
    flatbuffers::Verifier verifier(start_fl_job_rsp_msg->data(), start_fl_job_rsp_msg->size());
    if (!verifier.VerifyBuffer<schema::ResponseFLJob>()) {
      MS_LOG(EXCEPTION) << "The schema of ResponseFLJob is invalid.";
      return false;
    }

    const schema::ResponseFLJob *start_fl_job_rsp =
      flatbuffers::GetRoot<schema::ResponseFLJob>(start_fl_job_rsp_msg->data());
    MS_EXCEPTION_IF_NULL(start_fl_job_rsp);
    auto response_code = start_fl_job_rsp->retcode();
    switch (response_code) {
      case schema::ResponseCode_SUCCEED:
      case schema::ResponseCode_OutOfTime:
        break;
      default:
        MS_LOG(EXCEPTION) << "Launching start fl job for worker failed. Reason: " << start_fl_job_rsp->reason();
    }

    uint64_t iteration = IntToSize(start_fl_job_rsp->iteration());
    fl::worker::FLWorker::GetInstance().set_fl_iteration_num(iteration);
    MS_LOG(INFO) << "Start fl job for iteration " << iteration;
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
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
    data_size_ = LongToInt(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "data_size"));
    fl::worker::FLWorker::GetInstance().set_data_size(data_size_);
    MS_LOG(INFO) << "Initializing StartFLJob kernel. fl_name: " << fl_name_ << ", fl_id: " << fl_id_
                 << ", data_size: " << data_size_ << ". Request will be sent to server " << target_server_rank_;

    fbb_ = std::make_shared<fl::FBBuilder>();
    MS_EXCEPTION_IF_NULL(fbb_);

    input_size_list_.push_back(sizeof(int));
    output_size_list_.push_back(sizeof(float));
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  bool BuildStartFLJobReq(const std::shared_ptr<fl::FBBuilder> &fbb) {
    MS_EXCEPTION_IF_NULL(fbb);
    auto fbs_fl_name = fbb->CreateString(fl_name_);
    auto fbs_fl_id = fbb->CreateString(fl_id_);
    schema::RequestFLJobBuilder req_fl_job_builder(*(fbb.get()));
    req_fl_job_builder.add_fl_name(fbs_fl_name);
    req_fl_job_builder.add_fl_id(fbs_fl_id);
    req_fl_job_builder.add_data_size(data_size_);
    auto req_fl_job = req_fl_job_builder.Finish();
    fbb->Finish(req_fl_job);
    return true;
  }

  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_name_;
  std::string fl_id_;
  int data_size_;
  std::shared_ptr<fl::FBBuilder> fbb_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_START_FL_JOB_H_
