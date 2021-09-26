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
#include "src/cpu_info.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif

namespace mindspore::lite {

typedef enum { GRAPH, OP_BY_OP } MindRTMode;
const constexpr int kSwitchMaxInputKernelSize = 3;
const constexpr int kSwitchMinInputKernelSize = 2;
const constexpr int kSwitchTruePartialInputIndex = 1;
const constexpr int kSwitchFalsePartialInputIndex = 2;
const constexpr int kSwitchMinInputTensorSize = 3;
const constexpr int kSwitchCondTensorIndex = 0;

class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::LiteKernel *kernel) : OpActor<lite::Tensor>(kernel->name()), kernel_(kernel) {
    inputs_data_.resize(kernel_->in_tensors().size());
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    CpuInfo cpu_info;
    support_fp16_ = cpu_info.ArmIsSupportFp16();
#endif
  }
  ~LiteOpActor() override {
    for (auto map : isolate_input_map_) {
      auto isolate_input_tensor = map.first;
      isolate_input_tensor->set_data(nullptr);
      delete isolate_input_tensor;
    }
    delete call_node_;
    delete partial_node_;
  }
  void RunOpData(OpData<lite::Tensor> *input_data, OpContext<lite::Tensor> *context = nullptr) override;
  virtual int CompileArrow();
  int RunKernel(const KernelCallBack &before, const KernelCallBack &after) {
    auto ret = kernel_->Execute(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
      return ret;
    }
    return ret;
  }
  int LiteActorInit(std::vector<std::shared_ptr<LiteOpActor>> *actors);
  int ResizeGraphInput(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                       const std::vector<std::vector<int>> &dims);

 public:
  void AddResultIndex(size_t index);
  void SetSubgraphAIDMap(const std::unordered_map<kernel::LiteKernel *, AID> &partial_map) {
    subgraph_to_actor_ = partial_map;
  }

 protected:
  void SetInputShape();
  int InitInputData();
  void SetOutputData(OpContext<Tensor> *context);
  void AsyncOutput(OpContext<Tensor> *context);
  int CompileArrowThroughPartialCall();
  int CompileArrowThroughOutputKernels();
  virtual int PrepareOutputData();

  kernel::LiteKernel *kernel_;
  std::vector<size_t> results_index_{};
  std::unordered_map<kernel::LiteKernel *, AID> subgraph_to_actor_{};
  std::vector<OpDataPtr<Tensor>> outputs_data_{};
  std::vector<Tensor *> inputs_data_{};
  std::unordered_map<Tensor *, Tensor *> isolate_input_map_{}; /* <calculate-tensor,  src-input-tensor> */

 private:
  void ReplaceNodeInTensor(kernel::LiteKernel *kernel, Tensor *old_tensor, Tensor *new_tensor);
  int IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors);
  void MoveTensorInputData(Tensor *dst_tensor, Tensor *src_tensor);
  void MoveInputData(Tensor *dst_tensor, Tensor *src_tensor);
  void SetInputData(Tensor *dst_tensor, Tensor *src_tensor);
  int CastInputData(Tensor *dst_tensor, Tensor *src_tensor);
  bool NeedCastData(Tensor *dst_tensor, Tensor *src_tensor);
  int CastTensorInputData(Tensor *dst_tensor, Tensor *src_tensor);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  void MoveTensorListInputData(TensorList *dst_tensor, TensorList *src_tensor);
  int CastTensorListInputData(TensorList *dst_tensor, TensorList *src_tensor);
#endif

 private:
  kernel::LiteKernel *partial_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  bool support_fp16_ = false;
#endif
};

#ifndef CONTROLFLOW_TENSORLIST_CLIP
class LiteSwitchOpActor : public LiteOpActor {
 public:
  explicit LiteSwitchOpActor(kernel::LiteKernel *kernel) : LiteOpActor(kernel) {}
  ~LiteSwitchOpActor() override {
    delete call_node_;
    delete switch_node_;
    delete true_partial_node_;
    delete false_partial_node_;
  };
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override;
  int CompileArrow() override;
  int PrepareOutputData() override;

 private:
  void AsyncTrueBranchOutput(OpContext<Tensor> *context);
  void AsyncFalseBranchOutput(OpContext<Tensor> *context);
  void DecreaseTrueBranchInputTensor();
  void DecreaseFalseBranchInputTensor();
  int GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel);
  void AppendOutputTensors();
  int CompileTrueBranchArrow();
  int CompileFalseBranchArrow();
  int CompileArrowThroughSwitchCall();

  std::vector<DataArrowPtr> true_branch_output_data_arrows_;
  std::vector<DataArrowPtr> false_branch_output_data_arrows_;

  kernel::LiteKernel *true_partial_node_ = nullptr;
  kernel::LiteKernel *false_partial_node_ = nullptr;
  kernel::LiteKernel *switch_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
  std::vector<lite::Tensor *> output_tensors_{};

  std::vector<OpDataPtr<Tensor>> true_branch_outputs_data_;
  std::vector<OpDataPtr<Tensor>> false_branch_outputs_data_;
};
#endif

int MindrtInit();
void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &);

static std::atomic_int64_t actor_count = 0;
std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels,
                                                        const lite::InnerContext *ctx);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
