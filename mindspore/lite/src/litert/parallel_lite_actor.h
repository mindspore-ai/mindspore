/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_PARALLEL_LITE_ACTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PARALLEL_LITE_ACTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <set>
#include <utility>
#include "actor/op_actor.h"
#include "src/litert/kernel_exec.h"
#include "actor/actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/litert/sub_graph_kernel.h"
#include "src/litert/cpu_info.h"
#include "src/tensorlist.h"
#include "src/litert/lite_mindrt.h"

namespace mindspore::lite {
class KernelsActor;
class ParallelLiteActor : public LiteOpActor {
 public:
  explicit ParallelLiteActor(kernel::KernelExec *kernel, lite::InnerContext *ctx) : LiteOpActor(kernel, ctx) {}
  ~ParallelLiteActor() override;
  void RunOpData(OpData<lite::Tensor> *input_data, mindspore::OpContext<lite::Tensor> *context = nullptr) override;
  int PostInit() override;
  mindspore::OpContext<lite::Tensor> *OpContext() const { return op_context_; }
  inline void SetOpContext(mindspore::OpContext<lite::Tensor> *op_context) { op_context_ = op_context; }
  void AddKernelsActor(const std::shared_ptr<KernelsActor> &kernels_actor) { kernels_actors_.push_back(kernels_actor); }
  void SetBeginReadlyIndexs(const std::vector<size_t> &readly_indexs) { begin_readly_indexs_ = readly_indexs; }
  void CheckReadyActors(const std::vector<size_t> &indices);
  void AddOutputDataCount();
  int KernelActorInit();

 private:
  void DelKernelsActors();

 private:
  std::vector<std::shared_ptr<KernelsActor>> kernels_actors_;
  mindspore::OpContext<lite::Tensor> *op_context_ = nullptr;
  std::vector<size_t> begin_readly_indexs_{};
  std::atomic<int> output_data_count_ = 0;
  bool finish_ = true;
};

class KernelsActor : public ActorBase {
 public:
  explicit KernelsActor(ParallelLiteActor *parallel_lite_actor, const std::string &op_name,
                        const std::vector<kernel::KernelExec *> &nodes)
      : ActorBase(op_name), parallel_lite_actor_(parallel_lite_actor), nodes_(nodes) {}
  ~KernelsActor() override = default;
  void Run();
  void AddOutputDataArrows(const DataArrowPtr &data_arrow) { output_data_arrows_.push_back(data_arrow); }
  void AddOutputData(const OpDataPtr<Tensor> &data) { outputs_data_.push_back(data); }
  void AddResultsIndex(size_t result) { results_index_.push_back(result); }
  void SetInActorIndexs(const std::vector<size_t> &in_indexs) {
    in_actors_indexs_ = in_indexs;
    if (in_indexs.size() <= 1) {
      in_actors_num_ = 0;
      is_single_in_ = true;
    } else {
      in_actors_num_ = in_indexs.size() - 1;
      is_single_in_ = false;
    }
  }
  void SetOutActorIndexs(const std::vector<size_t> &out_indexs) { out_actors_indexs_ = out_indexs; }
  bool GetReady() {
    if (is_single_in_) {
      return true;
    } else if (ready_.fetch_add(1) == in_actors_num_) {
      ready_ = 0;
      return true;
    }
    return false;
  }
  inline void ClearReady() { ready_ = 0; }
  void SetIsSignleIn(bool flag) { is_single_in_ = flag; }
  void SetHaveOutput(bool flag) { have_output_ = true; }

 private:
  // The op data.
  ParallelLiteActor *parallel_lite_actor_ = nullptr;
  std::vector<kernel::KernelExec *> nodes_{};
  std::vector<size_t> out_actors_indexs_{};
  std::vector<size_t> in_actors_indexs_{};

  std::vector<size_t> results_index_{};
  std::vector<DataArrowPtr> output_data_arrows_;
  std::vector<OpDataPtr<Tensor>> outputs_data_{};

  std::atomic<int> ready_ = 0;  // This flag is used to reduce message communication
  bool is_single_in_ = false;
  int in_actors_num_ = 0;

  bool have_output_ = false;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PARALLEL_LITE_ACTOR_H_
