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
#include "src/litert/lite_session.h"

/*
       Inheritance Diagram

  +--------------+----------------+
  |        lite::LiteSession      |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |       lite::TrainSession      |
  +-------------------------------+
*/

#define TRAIN_SESSION_CHECK_FALSE_MSG(value, errcode, msg) \
  do {                                                     \
    if ((value)) {                                         \
      MS_LOG(ERROR) << #msg;                               \
      return errcode;                                      \
    }                                                      \
  } while (0)

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

  virtual int TrainInit(const std::shared_ptr<InnerContext> &context, const TrainCfg *train_cfg);

  int Train() override;
  int Eval() override;
  bool IsTrain() override { return train_mode_; }
  bool IsEval() override { return !train_mode_; }
  int SetLearningRate(float learning_rate) override;
  float GetLearningRate() override;
  std::vector<lite::Tensor *> GetGradients() const override;
  std::vector<lite::Tensor *> GetOptimizerParams() const override;
  int SetOptimizerParams(const std::vector<lite::Tensor *> &params) override;
  int ApplyGradients(const std::vector<lite::Tensor *> &gradients) override;
  int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) override;

  void BindThread(bool if_bind) override { return lite::LiteSession::BindThread(if_bind); }
  std::vector<lite::Tensor *> GetInputs() const override { return lite::LiteSession::GetInputs(); }
  mindspore::lite::Tensor *GetInputsByTensorName(const std::string &tensor_name) const override {
    return lite::LiteSession::GetInputsByTensorName(tensor_name);
  }
  std::vector<lite::Tensor *> GetOutputsByNodeName(const std::string &node_name) const override {
    return lite::LiteSession::GetOutputsByNodeName(node_name);
  }
  std::unordered_map<std::string, mindspore::lite::Tensor *> GetOutputs() const override {
    return lite::LiteSession::GetOutputs();
  }

  std::vector<std::string> GetOutputTensorNames() const override { return lite::LiteSession::GetOutputTensorNames(); }
  mindspore::lite::Tensor *GetOutputByTensorName(const std::string &tensor_name) const override {
    return lite::LiteSession::GetOutputByTensorName(tensor_name);
  }
  int Resize(const std::vector<lite::Tensor *> &inputs, const std::vector<std::vector<int>> &dims) override;

  std::vector<lite::Tensor *> GetPredictions() const override {
    std::vector<lite::Tensor *> outputs;
    for (auto it = eval_output_tensor_map_.begin(); it != eval_output_tensor_map_.end(); ++it) {
      outputs.push_back(it->second);
    }
    return outputs;
  }
  int Export(const std::string &fb_name, ModelType model_type, QuantizationType quant_type, FormatType,
             std::vector<std::string> out_put_tensor_name = {}) override;
  int Export(Buffer *model_buffer, ModelType model_type, QuantizationType quant_type, FormatType,
             std::vector<std::string> out_put_tensor_name = {}) override;
  int ExportWeightsCollaborateWithMicro(const std::string &file_name, lite::ModelType model_type, FormatType,
                                        bool enable_fp16,
                                        const std::vector<std::string> &changeable_weights_name) override;
  std::vector<lite::Tensor *> GetFeatureMaps() const override;
  std::vector<lite::Tensor *> GetTrainableParams() const override;

  int UpdateFeatureMaps(const std::vector<lite::Tensor *> &features_map) override;
  int FindUseInTensorKernel(std::vector<kernel::KernelExec *> *use_in_tensor_kernels,
                            const std::vector<lite::Tensor *> &kernel_in_tensors,
                            const std::vector<kernel::KernelExec *> &inference_kernels);
  int FindExportKernels(std::vector<kernel::KernelExec *> *export_kernels,
                        const std::vector<std::string> &export_output_tensor_names,
                        const std::vector<kernel::KernelExec *> &inference_kernels);

 protected:
  int AllocWorkSpace();
  bool IsLossKernel(const kernel::KernelExec *kernel) const;
  bool IsGradKernel(const kernel::KernelExec *kernel) const;
  bool IsOptimizer(kernel::KernelExec *kernel) const;
  bool IsMaskOutput(kernel::KernelExec *kernel) const;
  bool IsBN(kernel::KernelExec *kernel) const;

  virtual std::vector<CreatorOp> ReplaceOps();
  virtual void RestoreOps(const std::vector<CreatorOp> &restore);
  virtual void CompileTrainKernels();
  virtual int CompileInferenceKernels();
  virtual void CompileOptimizedKernels();
  virtual void CompileTrainableParams();
  virtual void CompileTrainOutputs();
  virtual void CompileEvalOutputs();
  virtual int InitCallBack();
  virtual int FindConstFoldedKernels();
  std::shared_ptr<Model> model_ = nullptr;
  std::unordered_map<std::string, std::vector<mindspore::lite::Tensor *>> orig_output_node_map_;
  std::unordered_map<std::string, mindspore::lite::Tensor *> orig_output_tensor_map_;
  std::vector<std::string> orig_output_tensor_names_;

  std::unordered_map<std::string, std::vector<mindspore::lite::Tensor *>> eval_output_node_map_;
  std::unordered_map<std::string, mindspore::lite::Tensor *> eval_output_tensor_map_;
  std::vector<std::string> eval_output_tensor_names_;

  std::unordered_map<std::string, std::vector<mindspore::lite::Tensor *>> train_output_node_map_;
  std::unordered_map<std::string, mindspore::lite::Tensor *> train_output_tensor_map_;
  std::vector<std::string> train_output_tensor_names_;

  std::vector<kernel::KernelExec *> inference_kernels_;
  std::vector<kernel::KernelExec *> train_kernels_;
  std::vector<kernel::KernelExec *> const_fold_kernels_;
  std::vector<lite::Tensor *> const_output_tensors_;
  TrainCfg cfg_;

 private:
  std::vector<std::string> get_loss_name() const { return cfg_.loss_name_; }
  void BuildInferenceKernelsRecursive(kernel::KernelExec *ker, std::vector<kernel::KernelExec *> *req_kernels);
  int AdminSetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum);
  int OptimizerStep();
  int ExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                  const std::vector<kernel::KernelExec *> &run_kernel);
  int MixPrecisionExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                              const std::vector<kernel::KernelExec *> &run_kernel);
  int MixPrecisionPreProcess(kernel::KernelExec *kernel, float scale);
  int MixPrecisionPostProcess(kernel::KernelExec *kernel);
  bool IsLossTensor(Tensor *tensor);
  void RestoreTensorData();
  void FreeRestoreTensors();
  bool AllInputsNeedScale(kernel::KernelExec *kernel);
  void FreeWorkSpace();
  int AllocTensors(const std::vector<kernel::KernelExec *> &kernels);
  bool IsInPlaceKernel(kernel::KernelExec *kernel);
  bool IsInPlaceTensor(kernel::KernelExec *kernel, uint32_t idx,
                       const std::unordered_map<lite::Tensor *, int> &ref_count, uint32_t *input_idx);
  size_t GetInplaceTensorOffset(kernel::KernelExec *kernel,
                                const std::unordered_map<lite::Tensor *, size_t> &offset_map,
                                std::unordered_map<lite::Tensor *, int> *ref_count, uint32_t input_idx);
  template <typename DestType>
  int ExportByDifferentType(DestType destination, ModelType model_type, QuantizationType quant_type,
                            bool orig_train_state, std::vector<std::string> output_tensor_name = {});
  template <typename DestType>
  int ExportInner(DestType destination, ModelType model_type, QuantizationType quant_type, FormatType,
                  std::vector<std::string> out_put_tensor_name = {});
  lite::Tensor *FindObfTensor();
  int ChangeObfWeight(std::string tensor_name, float obf_ratio);
  float ModelRecoverObfuscate();
  int ModelDeObfuscate(float obf_ratio);
  std::map<Tensor *, Tensor *> restored_origin_tensors_;
  std::vector<Tensor *> trainable_parameters_;
  int virtual_batch_idx_ = 0;
  int virtual_batch_multiplier_ = 0;
  uint32_t num_of_not_nan_iter_ = 0;
  void *workspace_ = nullptr;
  SchedCallBack sched_mix_precision_callback_;
  bool train_mode_ = false;
  bool model_buff_changed_ = false;
  void *tensors_data_ = nullptr;
  size_t tensors_data_size_ = 0;
  std::shared_ptr<Allocator> allocator_;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_SESSION_H_
