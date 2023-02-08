/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_LITE_SESSION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_LITE_SESSION_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <atomic>
#include "src/litert/kernel_exec.h"
#include "src/litert/lite_model.h"
#include "src/litert/inner_context.h"
#include "src/litert/runtime_allocator.h"
#include "schema/model_generated.h"
#include "src/litert/executor.h"
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "src/common/dynamic_library_loader.h"
#include "include/api/delegate.h"
#if GPU_OPENCL
#include "src/litert/kernel/gpu/opencl/opencl_runtime.h"
#endif
#include "src/litert/scheduler_cb.h"

#ifdef ENABLE_LITE_HELPER
#include "src/common/helper/infer_helpers.h"
#endif

namespace mindspore {
namespace lite {
class MS_API LiteSession {
 public:
  LiteSession();
  virtual ~LiteSession();
  static LiteSession *CreateSession(const std::shared_ptr<InnerContext> &context);
  static LiteSession *CreateSession(const char *model_buf, size_t size, const std::shared_ptr<InnerContext> &context);
#ifdef ENABLE_LITE_HELPER
  int LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type, const size_t &buf_size,
                               mindspore::infer::helper::InferHelpers *infer_helpers = nullptr);
#else
  int LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type, const size_t &buf_size);
#endif
  int LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type);
  mindspore::ModelType LoadModelByBuff(const char *model_buf, const size_t &buf_size, char **lite_buf, size_t *size,
                                       mindspore::ModelType model_type);
  const char *LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size);
  virtual int Init(const std::shared_ptr<InnerContext> &context);
  virtual void BindThread(bool if_bind);
  virtual int CompileGraph(Model *model);
  virtual std::vector<mindspore::lite::Tensor *> GetInputs() const;
  virtual mindspore::lite::Tensor *GetInputsByTensorName(const std::string &name) const;
  virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr);
  virtual std::vector<mindspore::lite::Tensor *> GetOutputsByNodeName(const std::string &node_name) const;
  virtual std::vector<std::string> GetOutputTensorNames() const;
  virtual mindspore::lite::Tensor *GetOutputByTensorName(const std::string &tensor_name) const;
  virtual std::unordered_map<std::string, mindspore::lite::Tensor *> GetOutputs() const;
  virtual int BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture);
  virtual int Resize(const std::vector<mindspore::lite::Tensor *> &inputs, const std::vector<std::vector<int>> &dims);
  void InitExecutionConfig(std::map<std::string, TypeId> *config) { execution_plan_ = config; }
  void set_model(Model *model) { this->model_ = model; }
  const std::vector<kernel::KernelExec *> &get_kernels() const { return this->kernels_; }
  const Delegate *get_delegate() const { return this->delegate_.get(); }
  void SetConfigInfo(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
    config_info_ = config_info;
  }
  void SetPrepareSessionFlag(bool is_prepare_session) { is_prepare_session_ = is_prepare_session; }
  const std::vector<Tensor *> &GetTensors() const { return this->tensors_; }

  virtual int Train() { return mindspore::lite::RET_ERROR; }
  virtual bool IsTrain() { return false; }
  virtual int Eval() { return mindspore::lite::RET_OK; }
  virtual bool IsEval() { return true; }
  virtual int SetLearningRate(float learning_rate) { return mindspore::lite::RET_ERROR; }
  virtual float GetLearningRate() { return 0.0; }
  virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) {
    return mindspore::lite::RET_ERROR;
  }
  virtual std::vector<lite::Tensor *> GetPredictions() const {
    std::vector<lite::Tensor *> outputs;
    return outputs;
  }
  virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFERS,
                     std::vector<std::string> out_put_tensor_name = {}) {
    return mindspore::lite::RET_ERROR;
  }
  virtual int Export(Buffer *model_buffer, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFERS,
                     std::vector<std::string> out_put_tensor_name = {}) {
    return mindspore::lite::RET_ERROR;
  }
  virtual int UpdateWeights(std::vector<lite::Tensor *> new_weights) { return mindspore::lite::RET_ERROR; }
  virtual std::vector<lite::Tensor *> GetFeatureMaps() const {
    std::vector<lite::Tensor *> features;
    return features;
  }
  virtual std::vector<lite::Tensor *> GetTrainableParams() const {
    std::vector<lite::Tensor *> train_params;
    return train_params;
  }
  virtual int UpdateFeatureMaps(const std::vector<lite::Tensor *> &features) { return mindspore::lite::RET_ERROR; }
  virtual std::vector<lite::Tensor *> GetGradients() const {
    std::vector<lite::Tensor *> gradients;
    return gradients;
  }
  virtual int ApplyGradients(const std::vector<lite::Tensor *> &gradients) { return mindspore::lite::RET_ERROR; }
  virtual std::vector<lite::Tensor *> GetOptimizerParams() const {
    std::vector<lite::Tensor *> params;
    return params;
  }
  virtual int SetOptimizerParams(const std::vector<lite::Tensor *> &params) { return mindspore::lite::RET_ERROR; }

  bool GetKeepModelBuf() { return keep_model_buf_; }

  void SetKeepModelBuf(bool keep_model_buf) { keep_model_buf_ = keep_model_buf; }

  void SetModelId(std::string id) { model_id_ = id; }

 protected:
  static void ConvertTensorsQuantParam(const schema::Tensor *src_tensor, lite::Tensor *dst_tensor);
  int CheckTensorValid(lite::Tensor *dst_tensor);
  int ConvertTensorsData(const lite::LiteModel *model, size_t tensor_index, lite::Tensor *dst_tensor);
  lite::Tensor *ConvertTensor(const schema::Tensor &src_tensor);
  int ConvertTensors(const lite::Model *model);
  void InitGraphInOutTensorsMap(const lite::Model *model);
  void InitGraphInputTensors(const lite::Model *model);
  void InitGraphInputMSTensors();
  void InitGraphOutputTensors(const lite::Model *model);
  void InitGraphInputMap(const lite::Model *model);
  void InitGraphOutputNodeMap(const lite::Model *model);
  void InitGraphOutputTensorMap(const lite::Model *model);
  int UpdateInputShapeMap();
  int ResizeInputs(const std::vector<mindspore::lite::Tensor *> &inputs, const std::vector<std::vector<int>> &dims);
  int SetAllocatorForDelegateKernels(const kernel::KernelExec *kernel);
  int PrepareKernels(const Model *model);
  int SetTensorInitRefCount();
  int SetNonTaiCallSubgraphOutputInitRefCount();
  static int ReSizeKernels(
    const std::vector<kernel::KernelExec *> &kernels,
    const std::unordered_map<Tensor *, Tensor *> &isolate_input_map = std::unordered_map<Tensor *, Tensor *>());
  static void FreePackOpWeight(const std::vector<kernel::KernelExec *> &kernels);
  static void MarkSharedWeight(const std::vector<kernel::KernelExec *> &kernels);
  std::string ParseWeightPath();

 private:
  int PreCheck(Model *model);
  int InitExecutor();
  void ResetInputsShape(const std::vector<std::vector<int>> &dims);
  int ContextInit(const std::shared_ptr<InnerContext> &context);
  int CreateTensorRTDelegate();
  int CreateNPUDelegate();
  int CreateNNAPIDelegate();
  int CreateCoreMLDelegate();
  int DelegateInit();
  int InitGPURuntime();
  int InitSharedThreadPool();

 private:
  int IsolateOutputTensor();
  bool IsIsolatedSubGraph(const kernel::KernelExec *kernel);
  void UpdateGraphOutputMap(const std::vector<kernel::KernelExec *> &kernel);
  void UpdateLinkInfoForIsolateOutput();
  void SynIsolateInOutputDataType();
  std::unordered_map<Tensor *, Tensor *> isolate_graph_output_map_; /* <calculate-tensor,  graph-output-tensor> */
  std::unordered_map<Tensor *, Tensor *> isolate_input_map_;        /* <calculate-tensor,  src-subgraph-input-tensor> */

 private:
  int RuntimeAllocatorInit();
  int RuntimeAllocatorSetData();
  void RuntimeAllocatorInitGraphOutput();
  void RuntimeAllocatorInitSubgraph();
  virtual int RuntimeAllocatorValid();
  RuntimeAllocatorPtr runtime_allocator_ = nullptr;

 private:
  int AscendInit(const std::shared_ptr<InnerContext> &context);

 protected:
  std::shared_ptr<InnerContext> context_ = nullptr;
  mindspore::Context *ms_context_ = nullptr;
  std::vector<kernel::KernelExec *> kernels_;
  std::vector<Tensor *> tensors_;
  // graph input tensors
  std::vector<Tensor *> inputs_;
  // graph output tensors
  std::vector<Tensor *> outputs_;
  // graph input MSTensors
  std::vector<mindspore::lite::Tensor *> input_vec_;
  // graph input tensor name -- input tensors
  std::unordered_map<std::string, mindspore::lite::Tensor *> input_map_;
  // graph input tensor -- input tensor shape
  std::unordered_map<Tensor *, std::vector<int>> input_shape_map_;
  // graph output node name -- output tensors
  std::unordered_map<std::string, std::vector<mindspore::lite::Tensor *>> output_node_map_;

  std::vector<std::string> output_tensor_names_;
  // graph output tensor name -- output tensor
  std::unordered_map<std::string, mindspore::lite::Tensor *> output_tensor_map_;

  Executor *executor_ = nullptr;
  Model *model_ = nullptr;
  std::atomic<bool> is_running_ = {false};
  bool is_train_session_ = false;
  bool is_prepare_session_ = false;
  friend class TransferSession;
#if GPU_OPENCL
  opencl::OpenCLRuntimeInnerWrapper *opencl_runtime_wrapper_{nullptr};
#endif

  // In the dynamic shape scene, the flag is to indicate when to do shape-infer for kernel. If true, the shape-infer
  // will not be called when calling 'Resize', but be done along with running. And we will decide whether to call
  // shape-infer by judging whether existing input has changed. If false, the shape-infer will be pre-called when
  // calling 'Resize'. And we will judge the outputs to decide whether to call shape-infer when running. Currently, the
  // value is true only in the pure CPU scenario, at the meantime, both of 'is_control_flow_' and 'is_train_session_'
  // are false and 'runtime_allocator_' is a nullptr.
  bool infer_along_running_{true};

  int is_infershape_{RET_ERROR};
  bool is_control_flow_ = false;
  bool keep_model_buf_ = false;
  std::unique_ptr<SchedulerCb> sched_cb_;
  std::shared_ptr<Delegate> delegate_ = nullptr;
  int delegate_device_type_ = -1;  // -1: not specified; 0: CPU; 1: GPU; 2: NPU
  std::map<std::string, TypeId> *execution_plan_ = nullptr;
  const std::map<std::string, std::map<std::string, std::string>> *config_info_ = nullptr;
  std::vector<kernel::KernelExec *> non_tail_call_kernels_;
  std::string model_id_;
  std::string runner_id_;
  int worker_id_;
  bool is_shared_weight_ = false;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_LITE_SESSION_H_
