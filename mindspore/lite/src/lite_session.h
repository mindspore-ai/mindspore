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

#ifndef MINDSPORE_LITE_SRC_LITE_SESSION_H_
#define MINDSPORE_LITE_SRC_LITE_SESSION_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <atomic>
#include "src/lite_kernel.h"
#include "include/ms_tensor.h"
#include "include/lite_session.h"
#include "src/lite_model.h"
#include "src/inner_context.h"
#include "src/runtime/runtime_allocator.h"
#include "schema/model_generated.h"
#include "src/executor.h"
#include "src/tensor.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif
#ifndef DELEGATE_CLIP
#include "include/api/delegate.h"
#endif
#if GPU_OPENCL
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#endif
#include "src/scheduler_cb.h"

namespace mindspore {
namespace lite {
class LiteSession : public session::LiteSession {
 public:
  LiteSession();
  ~LiteSession() override;
  static session::LiteSession *CreateSession(const std::string &model_path, const lite::Context *context);
  int LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type, const size_t &buf_size);
  int LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type, const size_t &buf_size,
                               const std::shared_ptr<mindspore::Context> &ms_context);

  int LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type);
  int LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type,
                                const std::shared_ptr<mindspore::Context> &ms_context);
  static mindspore::ModelType LoadModelByBuff(const char *model_buf, const size_t &buf_size, char **lite_buf,
                                              size_t *size, mindspore::ModelType model_type);
  static mindspore::ModelType LoadModelByBuff(const char *model_buf, const size_t &buf_size, char **lite_buf,
                                              size_t *size, mindspore::ModelType model_type,
                                              const std::shared_ptr<mindspore::Context> &ms_context);
  static const char *LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size);
  static const char *LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size,
                                     const std::shared_ptr<mindspore::Context> &ms_context);
  virtual int Init(InnerContext *context);
  void BindThread(bool if_bind) override;
  int CompileGraph(Model *model) override;
  std::vector<mindspore::tensor::MSTensor *> GetInputs() const override;
  mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &name) const override;
  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;
  std::vector<mindspore::tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const override;
  std::vector<std::string> GetOutputTensorNames() const override;
  mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const override;
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> GetOutputs() const override;
#ifdef ENABLE_OPENGL_TEXTURE
  int BindGLTexture2DMemory(const std::map<std::string, GLuint> &inputGLTexture,
                            std::map<std::string, GLuint> *outputGLTexture) override;
#endif
  int Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs,
             const std::vector<std::vector<int>> &dims) override;
  void InitExecutionConfig(std::map<std::string, TypeId> *config) { execution_plan_ = config; }
  void set_model(Model *model) { this->model_ = model; }
  const std::vector<kernel::LiteKernel *> &get_kernels() const { return this->kernels_; }
  const Delegate *get_delegate() const { return this->delegate_.get(); }
  void SetConfigInfo(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
    config_info_ = config_info;
  }
  const std::vector<Tensor *> &GetTensors() const { return this->tensors_; }

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
  void AdjustModelOutputTensorInitRefCount(const lite::Model *model);
  int UpdateInputShapeMap();
  int ResizeInputs(const std::vector<mindspore::tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims);
  int SetAllocatorForDelegateKernels(const kernel::LiteKernel *kernel);
  int PrepareKernels(const Model *model);
  int SetTensorInitRefCount(const Model *model);
#ifdef ENABLE_V0
  void TensorNameCompatibleWithV0(const lite::Model *model);
#endif
#ifndef CONTROLFLOW_TENSORLIST_CLIP
  int SetNonTaiCallSubgraphOutputInitRefCount(const std::vector<kernel::LiteKernel *> &non_tail_call_kernels);
#endif
  static int ReSizeKernels(
    const std::vector<kernel::LiteKernel *> &kernels,
    const std::unordered_map<Tensor *, Tensor *> isolate_input_map = std::unordered_map<Tensor *, Tensor *>());
  static void FreePackOpWeight(const std::vector<kernel::LiteKernel *> &kernels);
#ifdef SERVER_INFERENCE
  int IniPackWeightData(Model *model);
#endif

 private:
  int PreCheck(Model *model);
  int InitExecutor();
  void ResetInputsShape(const std::vector<std::vector<int>> &dims);
  int ContextInit(InnerContext *context);
  int CreateTensorRTDelegate();
  int CreateNPUDelegate();
  int DelegateInit();
  int InitGPURuntime();

 private:
  int IsolateOutputTensor();
  bool IsIsolatedSubGraph(const kernel::LiteKernel *kernel);
  void UpdateGraphOutputMap(const std::vector<kernel::LiteKernel *> &kernel);
  void UpdateLinkInfoForIsolateOutput();
  std::unordered_map<Tensor *, Tensor *> isolate_graph_output_map_; /* <calculate-tensor,  graph-output-tensor> */
  std::unordered_map<Tensor *, Tensor *> isolate_input_map_;        /* <calculate-tensor,  src-subgraph-input-tensor> */

 private:
  int RuntimeAllocatorInit();
  int RuntimeAllocatorSetData();
  void RuntimeAllocatorInitGraphOutput();
  void RuntimeAllocatorInitSubgraph();
  virtual int RuntimeAllocatorValid();
  RuntimeAllocatorPtr runtime_allocator_ = nullptr;

 protected:
  InnerContext *context_ = nullptr;
  mindspore::Context *ms_context_ = nullptr;
  std::vector<kernel::LiteKernel *> kernels_;
  std::vector<Tensor *> tensors_;
  // graph input tensors
  std::vector<Tensor *> inputs_;
  // graph output tensors
  std::vector<Tensor *> outputs_;
  // graph input MSTensors
  std::vector<mindspore::tensor::MSTensor *> input_vec_;
  // graph input tensor name -- input tensors
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> input_map_;
  // graph input tensor -- input tensor shape
  std::unordered_map<Tensor *, std::vector<int>> input_shape_map_;
  // graph output node name -- output tensors
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> output_node_map_;

  std::vector<std::string> output_tensor_names_;
  // graph output tensor name -- output tensor
  std::unordered_map<std::string, mindspore::tensor::MSTensor *> output_tensor_map_;

  Executor *executor_ = nullptr;
  Model *model_ = nullptr;
  std::atomic<bool> is_running_ = {false};
  bool is_train_session_ = false;
  friend class TransferSession;
#if GPU_OPENCL
  opencl::OpenCLRuntimeInnerWrapper *opencl_runtime_wrapper_{nullptr};
#endif
  int is_infershape_{RET_ERROR};
  bool is_control_flow_ = false;
  std::unique_ptr<SchedulerCb> sched_cb_;
  std::shared_ptr<Delegate> delegate_ = nullptr;
  int delegate_device_type_ = -1;  // -1: not specified; 0: CPU; 1: GPU; 2: NPU
  std::map<std::string, TypeId> *execution_plan_ = nullptr;
  const std::map<std::string, std::map<std::string, std::string>> *config_info_ = nullptr;
  std::vector<kernel::LiteKernel *> non_tail_call_kernels_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITE_SESSION_H_
