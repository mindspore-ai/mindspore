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
class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::LiteKernel *kernel, lite::InnerContext *ctx)
      : OpActor<lite::Tensor>(kernel->name()), kernel_(kernel), ctx_(ctx) {
    inputs_data_.resize(kernel_->in_tensors().size());
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
    CpuInfo cpu_info;
    support_fp16_ = cpu_info.ArmIsSupportFp16();
#endif
  }
  ~LiteOpActor() override {
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
  int LiteActorInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                    std::unordered_map<Tensor *, Tensor *> *input_map);
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
  std::unordered_map<Tensor *, Tensor *> *isolate_input_map_ = nullptr; /* real obj in session */
  lite::InnerContext *ctx_ = nullptr;

 private:
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
  explicit LiteSwitchOpActor(kernel::LiteKernel *kernel, lite::InnerContext *ctx) : LiteOpActor(kernel, ctx) {}
  ~LiteSwitchOpActor() override {
    delete call_node_;
    delete switch_type_node_;
    for (auto &partial_node : partial_nodes_) {
      delete partial_node;
    }
  };
  void RunOpData(OpData<Tensor> *inputs, OpContext<Tensor> *context = nullptr) override;
  int CompileArrow() override;
  int PrepareOutputData() override;

 private:
  void AsyncBranchOutput(const size_t &index, OpContext<Tensor> *context);
  void DecreaseOtherBranchInputTensor(const size_t &index);
  int GetSwitchAndCallNode(kernel::SubGraphKernel *subgraph_kernel);
  void AppendOutputTensors();
  int CompileBranchArrow();
  int CompileArrowThroughSwitchCall();
  int SetSwitchPartialNodes();
  int SetSwitchLayerPartialNodes();

  // each element is a set of data arrow sent to the next target actor.
  std::vector<std::vector<DataArrowPtr>> all_branch_output_data_arrows_;

  std::vector<kernel::LiteKernel *> partial_nodes_{};
  kernel::LiteKernel *switch_type_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
  std::vector<lite::Tensor *> output_tensors_{};

  // each element is a set of output data which is going to be send to the next target actor.
  std::vector<std::vector<OpDataPtr<Tensor>>> all_branchs_output_data_;
};
#endif

int MindrtInit();
void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &);

static std::atomic_int64_t actor_count = 0;
std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels,
                                                        lite::InnerContext *ctx);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
