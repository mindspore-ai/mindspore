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
#include <set>
#include <utility>
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
  virtual int CompileArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map);
  int RunKernel(const KernelCallBack &before, const KernelCallBack &after) {
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
  int ResizeGraphInput(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                       const std::vector<std::vector<int>> &dims);

 public:
  void AddResultIndex(size_t index);
  const kernel::LiteKernel *GetKernel() { return kernel_; }
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  // call this function after CompileArrow
  virtual std::set<kernel::LiteKernel *> GetPartialKernels() const {
    if (partial_node_ == nullptr) {
      return {};
    }
    std::set<kernel::LiteKernel *> ret{partial_node_};
    return ret;
  }
#endif

 protected:
  virtual void SetInputShape();
  virtual void InitInputData();
  void SetOutputData(OpContext<Tensor> *context);
  virtual void AsyncOutput(OpContext<Tensor> *context);
  void SetTensorShape(Tensor *dst, Tensor *src);

  int CompileArrowThroughOutputTensors(
    const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map);
  std::set<void *> PartialSubgraphInputTensors(kernel::LiteKernel *partial_node);
  int IsolateInputData(std::vector<std::shared_ptr<LiteOpActor>> *actors,
                       std::unordered_map<Tensor *, Tensor *> *input_map);
  virtual int PrepareOutputData();
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  virtual int UpdateActorOutput();
  void SetTensorListShape(Tensor *dst, Tensor *src);
#endif
  int CastTensorData(Tensor *dst_tensor, Tensor *src_tensor);
  bool NeedCastData(Tensor *dst_tensor, Tensor *src_tensor);
  int CastCommonTensorData(Tensor *dst_tensor, Tensor *src_tensor);
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  int CastTensorListTensorData(TensorList *dst_tensor, TensorList *src_tensor);
#endif

  kernel::LiteKernel *kernel_;
  std::vector<size_t> results_index_{};
  std::vector<OpDataPtr<Tensor>> outputs_data_{};
  std::vector<Tensor *> inputs_data_{};
  std::unordered_map<Tensor *, Tensor *> *isolate_input_map_ = nullptr; /* real obj in session */
  lite::InnerContext *ctx_ = nullptr;

 private:
  int CreateCommonArrow(const std::unordered_map<void *, std::set<std::pair<AID, size_t>>> &receivers_map,
                        const std::set<void *> &subgraph_inputs_set, const std::set<void *> &receiver_tensors,
                        const size_t &output_index, std::unordered_map<AID, std::set<size_t>> *receiver_index_set);
  int CreateEmptyArrow(const size_t &output_index);
  bool ArrowHasCompiled(const AID &actor_name, const size_t &to_index,
                        const std::unordered_map<AID, std::set<size_t>> &receiver_index_set);
  void MarkArrowAsCompiled(const AID *actor_name, const size_t *to_index,
                           std::unordered_map<AID, std::set<size_t>> *receiver_index_set);

 private:
  kernel::LiteKernel *partial_node_ = nullptr;
  kernel::LiteKernel *call_node_ = nullptr;
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  bool support_fp16_ = false;
#endif
};

int MindrtInit();
void MindrtTerminate(const std::vector<std::shared_ptr<LiteOpActor>> &);
static std::atomic_int64_t actor_count = 0;

std::vector<std::shared_ptr<LiteOpActor>> CreateOpActor(const std::vector<kernel::LiteKernel *> &kernels,
                                                        lite::InnerContext *ctx);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_LITE_MINDRT_H_
