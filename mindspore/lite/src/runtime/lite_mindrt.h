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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_LITE_MINDRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_LITE_MINDRT_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <set>
#include <utility>
#include "actor/op_actor.h"
#include "src/runtime/kernel_exec.h"
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/runtime/sub_graph_kernel.h"
#include "src/runtime/cpu_info.h"
#include "src/tensorlist.h"

namespace mindspore::lite {

typedef enum { GRAPH, OP_BY_OP } MindRTMode;
class LiteOpActor : public OpActor<lite::Tensor> {
 public:
  explicit LiteOpActor(kernel::KernelExec *kernel, lite::InnerContext *ctx)
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
  virtual int CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map);
  int RunKernel(KernelCallBack before, KernelCallBack after) {
    auto ret = kernel_->Execute(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel_->name();
      return ret;
    }
    return ret;
  }
  virtual int PreInit(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                      std::unordered_map<Tensor *, Tensor *> *input_map);
  virtual int PostInit();
  int ResizeGraphInput(const std::vector<mindspore::lite::Tensor *> &inputs, const std::vector<std::vector<int>> &dims);

 public:
  void AddResultIndex(size_t index, size_t tensor_index);
  const kernel::KernelExec *GetKernel() { return kernel_; }
  // call this function after CompileArrow
  virtual std::set<kernel::KernelExec *> GetPartialKernels() const {
    if (partial_node_ == nullptr) {
      return {};
    }
    std::set<kernel::KernelExec *> ret{partial_node_};
    return ret;
  }

 protected:
  virtual bool NeedResize();
  virtual int SetInputShape();
  virtual int InitInputData();
  virtual int AssignInputData();
  void SetOutputData(OpContext<Tensor> *context);
  virtual void AsyncOutput(OpContext<Tensor> *context);

  int CompileArrowThroughOutputTensors(
    const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map);
  int IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                       std::unordered_map<Tensor *, Tensor *> *input_map);
  virtual int PrepareOutputData();
  virtual int UpdateActorOutput();

  kernel::KernelExec *kernel_;
  std::vector<size_t> results_index_{};
  std::vector<size_t> results_tensor_index_{};
  std::vector<OpDataPtr<Tensor>> outputs_data_{};
  std::vector<Tensor *> inputs_data_{};
  std::unordered_map<Tensor *, Tensor *> *isolate_input_map_ = nullptr; /* real obj in session */
  lite::InnerContext *ctx_ = nullptr;

 private:
  int CreateCommonArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
                        const std::set<void *> &receiver_tensors, const size_t &output_index,
                        std::unordered_map<AID, std::set<size_t>> *receiver_index_set);
  int CreateEmptyArrow(const size_t &output_index);
  bool ArrowHasCompiled(const AID &actor_name, size_t to_index,
                        const std::unordered_map<AID, std::set<size_t>> &receiver_index_set);
  void MarkArrowAsCompiled(const AID *actor_name, size_t to_index,
                           std::unordered_map<AID, std::set<size_t>> *receiver_index_set);

 private:
  kernel::KernelExec *partial_node_ = nullptr;
  kernel::KernelExec *call_node_ = nullptr;
  bool support_fp16_ = false;
};

int MindrtInit();
void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &,
                     const std::shared_ptr<ActorMgr> &actor_mgr = nullptr);
static std::atomic_int64_t actor_count = 0;

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::KernelExec *> &kernels,
                                                        lite::InnerContext *ctx, const std::shared_ptr<ActorMgr> &mgr);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_LITE_MINDRT_H_
