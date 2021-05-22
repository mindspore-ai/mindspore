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

#ifndef MINDSPORE_LITE_SRC_LITE_MINDRT_H_
#define MINDSPORE_LITE_SRC_LITE_MINDRT_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "actor/op_actor.h"
#include "src/lite_kernel.h"
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/sub_graph_kernel.h"

namespace mindspore::lite {

typedef enum { GRAPH, OP_BY_OP } MindRTMode;
const constexpr int kSwitchInputsSize = 3;
const constexpr int kSwitchCondInputIndex = 0;
const constexpr int kSwitchTruePartialInputIndex = 1;
const constexpr int kSwitchFalsePartialInputIndex = 2;

class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::LiteKernel *kernel) : OpActor<lite::Tensor>(kernel->name()), kernel_(kernel) {}
  ~LiteOpActor() override = default;
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override {
    auto op_uuid = context->sequential_num_;
    input_op_datas_[op_uuid].push_back(inputs);
    inputs_data_.push_back(inputs->data_);
    if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
      return;
    }

    CpuBindMode cpu_bind_mode = kernel_->Context()->device_list_.front().device_info_.cpu_device_info_.cpu_bind_mode_;
    BindThreads(static_cast<const lite::InnerContext *>(kernel_->Context())->thread_pool_, true, cpu_bind_mode);

    int ret = CheckInputData();
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }

    ret = SetInputData();
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }
    for (auto &arrow : output_data_arrows_) {
      kernel_->out_tensors().at(arrow->from_output_index_)->IncRefCount();
    }

    ret = RunKernel(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
                    *(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_)));
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }
    input_op_datas_.erase(op_uuid);
    inputs_data_.clear();
    AsyncOutput(context);

    BindThreads(static_cast<const lite::InnerContext *>(kernel_->Context())->thread_pool_, false, cpu_bind_mode);
    SetOutputData(context);

    for (auto &input_data : inputs_data_) {
      input_data->DecRefCount();
    }
  }
  int CastTensorData(Tensor *dst, Tensor *src);
  void Init() override {
    auto ret = CompileArrow();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CompileArrow failed, name: " << kernel_->name();
      // do not support return error
    }

    ret = PrepareOutputData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "PrepareOutputData failed, name: " << kernel_->name();
      // do not support return error
    }
  }
  virtual int CompileArrow();
  int RunKernel(const KernelCallBack &before, const KernelCallBack &after) {
    auto ret = kernel_->Execute(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
      return ret;
    }
    return ret;
  }

 public:
  void AddResultIndex(size_t index);
  void SetPartialMap(const std::unordered_map<size_t, AID> &partial_map) { subgraph_index_to_actor = partial_map; }

 protected:
  int CheckInputData();
  int SetInputData();
  void MoveInputData(Tensor *dst_tensor, Tensor *src_tensor);
  void CopyInputData(Tensor *dst_tensor, Tensor *src_tensor);
  void SetOutputData(OpContext<Tensor> *context);
  void AsyncOutput(OpContext<Tensor> *context);
  int CompileArrowThroughPartialCall();
  int CompileArrowThroughOutputKernels();
  virtual int PrepareOutputData();

  kernel::LiteKernel *kernel_;
  std::vector<size_t> results_index_{};
  std::unordered_map<size_t, AID> subgraph_index_to_actor{};
  std::vector<OpDataPtr<Tensor>> outputs_data_{};
  std::vector<Tensor *> inputs_data_{};

 private:
  kernel::LiteKernel *partial_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
};

class LiteSwitchOpActor : public LiteOpActor {
 public:
  explicit LiteSwitchOpActor(kernel::LiteKernel *kernel) : LiteOpActor(kernel) {}
  ~LiteSwitchOpActor() override = default;
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override {
    auto op_uuid = context->sequential_num_;
    input_op_datas_[op_uuid].push_back(inputs);
    inputs_data_.push_back(inputs->data_);
    if (input_op_datas_[op_uuid].size() < kernel_->in_tensors().size()) {
      return;
    }

    int ret = CheckInputData();
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }

    ret = SetInputData();
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }

    ret = RunKernel(*(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_before_)),
                    *(reinterpret_cast<const KernelCallBack *>(context->kernel_call_back_after_)));
    if (ret != RET_OK) {
      input_op_datas_.erase(op_uuid);
      context->SetFailed(ret);
      return;
    }
    input_op_datas_.erase(op_uuid);
    inputs_data_.clear();

    bool *cond = reinterpret_cast<bool *>(output_tensors_[0]->data());
    if (*cond) {
      for (auto &arrow : true_branch_output_data_arrows_) {
        kernel_->out_tensors().at(arrow->from_output_index_)->IncRefCount();
      }
      AsyncTrueBranchOutput(context);
    } else {
      for (auto &arrow : false_branch_output_data_arrows_) {
        kernel_->out_tensors().at(arrow->from_output_index_)->IncRefCount();
      }
      AsyncFalseBranchOutput(context);
    }
  }

  void Init() override {
    auto ret = CompileArrow();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CompileArrow failed, name: " << kernel_->name();
      // do not support return error
    }

    ret = PrepareOutputData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "PrepareOutputData failed, name: " << kernel_->name();
      // do not support return error
    }
  }
  int CompileArrow() override;

 private:
  void AsyncTrueBranchOutput(OpContext<Tensor> *context);
  void AsyncFalseBranchOutput(OpContext<Tensor> *context);

  int GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel);
  void AppendOutputTensors();
  int CompileTrueBranchArrow();
  int CompileFalseBranchArrow();
  int CompileArrowThroughSwitchCall();
  int PrepareOutputData() override;

  std::vector<DataArrowPtr> true_branch_output_data_arrows_;
  std::vector<DataArrowPtr> false_branch_output_data_arrows_;

  kernel::LiteKernel *bool_node_ = nullptr;
  kernel::LiteKernel *true_partial_node_ = nullptr;
  kernel::LiteKernel *false_partial_node_ = nullptr;
  kernel::LiteKernel *switch_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
  std::vector<lite::Tensor *> output_tensors_{};

  std::vector<OpDataPtr<Tensor>> true_branch_outputs_data_;
  std::vector<OpDataPtr<Tensor>> false_branch_outputs_data_;
};

int MindrtInit();
void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &);

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels);

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
