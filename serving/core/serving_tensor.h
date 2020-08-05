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

#ifndef MINDSPORE_SERVING_TENSOR_H_
#define MINDSPORE_SERVING_TENSOR_H_

#include <utility>
#include <vector>
#include <memory>
#include "include/infer_tensor.h"
#include "serving/ms_service.pb.h"

namespace mindspore {
namespace serving {

class MS_API ServingTensor : public inference::InferTensorBase {
 public:
  // the other's lifetime must longer than this object
  explicit ServingTensor(ms_serving::Tensor &other);
  ~ServingTensor();

  inference::DataType data_type() const override;
  void set_data_type(inference::DataType type) override;
  std::vector<int64_t> shape() const override;
  void set_shape(const std::vector<int64_t> &shape) override;
  const void *data() const override;
  size_t data_size() const override;
  bool resize_data(size_t data_len) override;
  void *mutable_data() override;

 private:
  // if tensor_ is reference from other ms_serving::Tensor, the other's lifetime must
  // longer than this object
  ms_serving::Tensor &tensor_;
};

class ServingImages : public inference::InferImagesBase {
 public:
  explicit ServingImages(const ms_serving::Images &images);

  size_t batch_size() const override;
  bool get(size_t index, const void *&pic_buffer, uint32_t &pic_size) const override;
  size_t input_index() const override;

 private:
  const ms_serving::Images &images_;
};

class ServingRequest : public inference::RequestBase {
 public:
  explicit ServingRequest(const ms_serving::PredictRequest &request);

  size_t size() const override;
  const inference::InferTensorBase *operator[](size_t index) const override;

 private:
  const ms_serving::PredictRequest &request_;
  std::vector<ServingTensor> cache_;
};

class ServingReply : public inference::ReplyBase {
 public:
  explicit ServingReply(ms_serving::PredictReply &reply) : reply_(reply) {}

  size_t size() const override;
  inference::InferTensorBase *operator[](size_t index) override;
  const inference::InferTensorBase *operator[](size_t index) const override;
  inference::InferTensorBase *add() override;
  void clear() override;

 private:
  ms_serving::PredictReply &reply_;
  std::vector<ServingTensor> cache_;
};

class ServingImagesRequest : public inference::ImagesRequestBase {
 public:
  explicit ServingImagesRequest(const ms_serving::PredictRequest &request);

  size_t size() const override;
  const inference::InferImagesBase *operator[](size_t index) const override;

 private:
  const ms_serving::PredictRequest &request_;
  std::vector<ServingImages> cache_;
};

}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVING_TENSOR_H_
