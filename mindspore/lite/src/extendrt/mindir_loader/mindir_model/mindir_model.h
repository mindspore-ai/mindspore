/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_H_

#include <vector>
#include <string>

#include "extendrt/mindir_loader/abstract_base_model.h"
#include "src/litert/schema_tensor_wrapper.h"
#include "proto/mind_ir.pb.h"

namespace mindspore::infer::mindir {
class TensorProtoWrap {
 public:
  TensorProtoWrap(std::string name, const mind_ir::TensorProto &tensor_proto)
      : name_(name), tensor_proto_(tensor_proto) {}
  ~TensorProtoWrap() = default;

  const mind_ir::TensorProto &tensor_proto() { return tensor_proto_; }

  std::string name() { return name_; }

 private:
  std::string name_;
  mind_ir::TensorProto tensor_proto_;
};

class MindirModel : public AbstractBaseModel {
 public:
  MindirModel() {}
  ~MindirModel() { Destroy(); }

  bool ModelVerify() const override;
  // virtual SchemaTensorWrapper *GetSchemaTensor(const size_t &tensor_index) const override;
  int ConvertTensors(std::vector<mindspore::lite::Tensor *> *lite_tensors) override;
  std::string GetModelPath() const override;
  virtual mindspore::kernel::KernelExec *FindBackendKernel(const std::vector<mindspore::lite::Tensor *> &in_tensors,
                                                           const std::vector<mindspore::lite::Tensor *> &out_tensors,
                                                           const LiteGraph::Node *node, lite::InnerContext *context,
                                                           TypeId prefer_data_type);

  void Free() override;
  void Destroy() override;

  void SetModelPath(const std::string &model_path) { this->model_path_ = model_path; }

 private:
  mindspore::lite::Tensor *ConvertTensor(TensorProtoWrap mindir_tensor);
  int LoadTensorData(mindspore::lite::Tensor *lite_tensor, const mind_ir::TensorProto &mindir_tensor);
  int CheckTensorValid(lite::Tensor *dst_tensor);
  mindspore::kernel::KernelExec *FindLiteKernel(const std::vector<mindspore::lite::Tensor *> &in_tensors,
                                                const std::vector<mindspore::lite::Tensor *> &out_tensors,
                                                const LiteGraph::Node *node, lite::InnerContext *context,
                                                TypeId prefer_data_type);

 public:
  std::vector<TensorProtoWrap> all_mindir_tensors_;
  mind_ir::ModelProto mindir_model_proto_;

 private:
  std::string model_path_;
  bool select_lite_kernel_ = true;
};
}  // namespace mindspore::infer::mindir

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_MINDIR_MODEL_MINDIR_MODEL_H_
