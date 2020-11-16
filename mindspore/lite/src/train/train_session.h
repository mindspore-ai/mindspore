/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include "src/ops/primitive_c.h"
#include "include/train_session.h"
#include "include/train_model.h"
#include "src/lite_session.h"

/*
                 Inheritance Diagram

            +-------------------------------+
            |     session::LiteSession      |
            +--------+------------+---------+
                    /              \
 +-----------------+-----+  +-------+------------+
 | session::TrainSession |  | lite::LiteSession  |
 +-----------------+-----+  +-------+------------+
                    \              /
            +--------+------------+---------+
            |       lite::TrainSession      |
            +-------------------------------+
*/

namespace mindspore {
namespace lite {

using CreatorOp = std::tuple<mindspore::kernel::KernelKey, mindspore::kernel::KernelCreator>;
class TrainSession : virtual public session::TrainSession, virtual public lite::LiteSession {
 public:
  TrainSession();
  ~TrainSession();

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  int CompileGraph(lite::Model *model) override;
  int CompileTrainGraph(lite::TrainModel *model) override;

  void *ExportToBuf(char *buf, size_t *len) const override;
  int SaveToFile(const std::string &filename) const override;

  int Train() override;
  int Eval() override;

  void BindThread(bool if_bind) override { return lite::LiteSession::BindThread(if_bind); }
  std::vector<tensor::MSTensor *> GetInputs() const override { return lite::LiteSession::GetInputs(); }
  mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &tensor_name) const override {
    return lite::LiteSession::GetInputsByTensorName(tensor_name);
  }
  std::vector<tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const override {
    return lite::LiteSession::GetOutputsByNodeName(node_name);
  }
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> GetOutputs() const override {
    return lite::LiteSession::GetOutputs();
  }

  std::vector<std::string> GetOutputTensorNames() const override { return lite::LiteSession::GetOutputTensorNames(); }
  mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const override {
    return lite::LiteSession::GetOutputByTensorName(tensor_name);
  }
  int Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) override {
    return lite::LiteSession::Resize(inputs, dims);
  }

  void UpdateOutputMapByInKernel(const kernel::LiteKernel *kernel);
  void UpdateOutputMapByLossKernel(const kernel::LiteKernel *kernel);

 protected:
  void AllocWorkSpace();
  bool IsLossKernel(const kernel::LiteKernel *kernel) const;
  bool IsOptimizer(kernel::LiteKernel *kernel) const;
  virtual void MarkOptimizedKernels();
  virtual std::vector<CreatorOp> ReplaceOps();
  virtual void RestoreOps(const std::vector<CreatorOp> &restore);
  virtual void BuildInferenceKernelsMap();
  virtual void BuildInferenceKernelsRecursive(kernel::LiteKernel *ker, std::vector<kernel::LiteKernel *> *req_kernels);
  virtual void CompileTrainKernels();
  TrainModel *model_ = nullptr;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> orig_output_map_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> orig_output_tensor_map_;
  std::vector<kernel::LiteKernel *> inference_kernels_;
  std::vector<kernel::LiteKernel *> train_kernels_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
