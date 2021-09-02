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
#include <memory>
#include <map>
#include "include/train/train_cfg.h"
#include "include/train/train_session.h"
#include "src/lite_session.h"

/*
       Inheritance Diagram

  +-------------------------------+
  |     session::LiteSession      |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |        lite::LiteSession      |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |       lite::TrainSession      |
  +-------------------------------+
*/

namespace mindspore {
namespace lite {
using CreatorOp = std::tuple<mindspore::kernel::KernelKey, mindspore::kernel::KernelCreator>;
class TrainSession : virtual public lite::LiteSession {
 public:
  TrainSession();
  ~TrainSession();

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  int CompileGraph(lite::Model *model) override;
  virtual int CompileTrainGraph(std::shared_ptr<Model> model);

  virtual int Init(InnerContext *context, const TrainCfg *train_cfg);

  int Train() override;
  int Eval() override;
  bool IsTrain() override { return train_mode_; }
  bool IsEval() override { return !train_mode_; }
  int SetLearningRate(float learning_rate) override;
  float GetLearningRate() override;
  std::vector<tensor::MSTensor *> GetGradients() const override;
  std::vector<tensor::MSTensor *> GetOptimizerParams() const override;
  int SetOptimizerParams(const std::vector<tensor::MSTensor *> &params) override;
  int ApplyGradients(const std::vector<tensor::MSTensor *> &gradients) override;
  int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) override;

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
  int Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) override;

  std::vector<tensor::MSTensor *> GetPredictions() const override {
    std::vector<tensor::MSTensor *> outputs;
    for (auto it = eval_output_tensor_map_.begin(); it != eval_output_tensor_map_.end(); ++it) {
      outputs.push_back(it->second);
    }
    return outputs;
  }
  int Export(const std::string &fb_name, ModelType model_type, QuantizationType quant_type, FormatType,
             std::vector<std::string> out_put_tensor_name = {}) override;

  std::vector<tensor::MSTensor *> GetFeatureMaps() const override;

  int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features_map) override;
  int FindUseInTensorKernel(std::vector<kernel::LiteKernel *> *use_in_tensor_kernels,
                            const std::vector<lite::Tensor *> &kernel_in_tensors,
                            const std::vector<kernel::LiteKernel *> &inference_kernels);
  int FindExportKernels(std::vector<kernel::LiteKernel *> *export_kernels,
                        const std::vector<std::string> &export_output_tensor_names,
                        const std::vector<kernel::LiteKernel *> &inference_kernels);

 protected:
  int AllocWorkSpace();
  bool IsLossKernel(const kernel::LiteKernel *kernel) const;
  bool IsGradKernel(const kernel::LiteKernel *kernel) const;
  bool IsOptimizer(kernel::LiteKernel *kernel) const;
  bool IsMaskOutput(kernel::LiteKernel *kernel) const;
  bool IsBN(kernel::LiteKernel *kernel) const;

  virtual std::vector<CreatorOp> ReplaceOps();
  virtual void RestoreOps(const std::vector<CreatorOp> &restore);
  virtual void CompileTrainKernels();
  virtual int CompileInferenceKernels();
  virtual void CompileOptimizedKernels();
  virtual void CompileTrainOutputs();
  virtual void CompileEvalOutputs();
  virtual int InitCallBack();
  std::shared_ptr<Model> model_ = nullptr;
  // TrainCfg train_cfg_;
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> orig_output_node_map_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> orig_output_tensor_map_;
  std::vector<std::string> orig_output_tensor_names_;

  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> eval_output_node_map_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> eval_output_tensor_map_;
  std::vector<std::string> eval_output_tensor_names_;

  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> train_output_node_map_;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> train_output_tensor_map_;
  std::vector<std::string> train_output_tensor_names_;

  std::vector<kernel::LiteKernel *> inference_kernels_;
  std::vector<kernel::LiteKernel *> train_kernels_;
  TrainCfg cfg_;

 private:
  std::string get_loss_name() const { return cfg_.loss_name_; }
  void BuildInferenceKernelsRecursive(kernel::LiteKernel *ker, std::vector<kernel::LiteKernel *> *req_kernels);
  int AdminSetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum);
  int OptimizerStep();
  int ExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                  const std::vector<kernel::LiteKernel *> &run_kernel);
  int MixPrecisionExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                              const std::vector<kernel::LiteKernel *> &run_kernel);
  int MixPrecisionPreProcess(kernel::LiteKernel *kernel, float scale);
  int MixPrecisionPostProcess(kernel::LiteKernel *kernel);
  bool IsLossTensor(Tensor *tensor);
  void RestoreTensorData();
  void FreeRestoreTensors();
  bool AllInputsNeedScale(kernel::LiteKernel *kernel);
  void FreeWorkSpace();
  int AllocTensors(const std::vector<kernel::LiteKernel *> &kernels);

  std::map<Tensor *, Tensor *> restored_origin_tensors_;
  int virtual_batch_idx_ = 0;
  int virtual_batch_multiplier_ = 0;
  uint32_t num_of_not_nan_iter_ = 0;
  void *workspace_ = nullptr;
  SchedCallBack sched_mix_precision_callback_;
  bool train_mode_ = false;
  void *tensors_data_ = nullptr;
  std::shared_ptr<Allocator> allocator_;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
