/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_KERNELS_TENSOR_OP_H_
#define DATASET_KERNELS_TENSOR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/core/tensor_row.h"
#include "dataset/util/status.h"

#define IO_CHECK(input, output)                             \
  do {                                                      \
    if (input == nullptr || output == nullptr) {            \
      RETURN_STATUS_UNEXPECTED("input or output is null."); \
    }                                                       \
  } while (false)

#define IO_CHECK_VECTOR(input, output)              \
  do {                                              \
    if (output == nullptr) {                        \
      RETURN_STATUS_UNEXPECTED("output is null.");  \
    }                                               \
    for (auto &_i : input) {                        \
      if (_i == nullptr) {                          \
        RETURN_STATUS_UNEXPECTED("input is null."); \
      }                                             \
    }                                               \
  } while (false)

#define BOUNDING_BOX_CHECK(input)                                                           \
  do {                                                                                      \
    if (input.size() != 2) {                                                                \
      return Status(StatusCode::kBoundingBoxInvalidShape, __LINE__, __FILE__,               \
                    "Requires Image and Bounding Boxes, likely missed bounding boxes.");    \
    }                                                                                       \
    if (input[1]->shape().Size() < 2) {                                                     \
      return Status(StatusCode::kBoundingBoxInvalidShape, __LINE__, __FILE__,               \
                    "Bounding boxes shape should have at least two dimensions.");           \
    }                                                                                       \
    uint32_t num_of_features = input[1]->shape()[1];                                        \
    if (num_of_features < 4) {                                                              \
      return Status(StatusCode::kBoundingBoxInvalidShape, __LINE__, __FILE__,               \
                    "Bounding boxes should be have at least 4 features.");                  \
    }                                                                                       \
    uint32_t num_of_boxes = input[1]->shape()[0];                                           \
    uint32_t img_h = input[0]->shape()[0];                                                  \
    uint32_t img_w = input[0]->shape()[1];                                                  \
    for (uint32_t i = 0; i < num_of_boxes; i++) {                                           \
      uint32_t min_x = 0;                                                                   \
      uint32_t min_y = 0;                                                                   \
      uint32_t b_w = 0;                                                                     \
      uint32_t b_h = 0;                                                                     \
      input[1]->GetItemAt<uint32_t>(&min_x, {i, 0});                                        \
      input[1]->GetItemAt<uint32_t>(&min_y, {i, 1});                                        \
      input[1]->GetItemAt<uint32_t>(&b_w, {i, 2});                                          \
      input[1]->GetItemAt<uint32_t>(&b_h, {i, 3});                                          \
      if ((min_x + b_w > img_w) || (min_y + b_h > img_h)) {                                 \
        return Status(StatusCode::kBoundingBoxOutOfBounds, __LINE__, __FILE__,              \
                      "At least one of the bounding boxes is out of bounds of the image."); \
      }                                                                                     \
      if (static_cast<int>(min_x) < 0 || static_cast<int>(min_y) < 0) {                     \
        return Status(StatusCode::kBoundingBoxOutOfBounds, __LINE__, __FILE__,              \
                      "At least one of the bounding boxes has negative min_x or min_y.");   \
      }                                                                                     \
    }                                                                                       \
  } while (false)

namespace mindspore {
namespace dataset {
// A class that does a computation on  a Tensor
class TensorOp {
 public:
  TensorOp() = default;

  virtual ~TensorOp() = default;

  // A function that prints info about the tensor operation
  // @param out
  virtual void Print(std::ostream &out) const;

  // Provide stream operator for displaying it
  // @param output stream
  // @param so the TensorOp object to be printed
  // @return output stream
  friend std::ostream &operator<<(std::ostream &out, const TensorOp &so) {
    so.Print(out);
    return out;
  }

  // Perform an operation on one Tensor and produce one Tensor. This is for 1-to-1 column MapOp
  // @param input  shares the ownership of the Tensor (increase the ref count).
  // @param output the address to a shared_ptr where the result will be placed.
  // @return Status
  virtual Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output);

  // Perform an operation on Tensors from multiple columns, and produce multiple Tensors.
  // This is for m-to-n column MapOp.
  // @param input is a vector of shared_ptr to Tensor (pass by const reference).
  // @param output is the address to an empty vector of shared_ptr to Tensor.
  // @return Status
  virtual Status Compute(const TensorRow &input, TensorRow *output);

  // Returns true oif the TensorOp takes one input and returns one output.
  // @return true/false
  bool OneToOne() { return NumInput() == 1 && NumOutput() == 1; }

  // Function to determine the number of inputs the TensorOp can take. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumInput() { return 1; }

  // Function to determine the number of output the TensorOp generates. 0: means undefined.
  // @return uint32_t
  virtual uint32_t NumOutput() { return 1; }

  // Function to determine the shapes of the output tensor given the input tensors' shapes.
  // If a subclass did not override this function, it means that the shape does not change.
  // @param inputs in: vector of the shapes of the input tensors.
  // @param outputs out: vector of the shapes of the output tensors to be filled.
  // @return Status
  virtual Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs);

  // Function to determine the types of the output tensor given the input tensor's types.
  // If a subclass did not override this function, it means that the type does not change.
  // @param inputs in: vector of the types of the input tensors.
  // @param outputs out: vector of the types of the output tensors to be filled.
  // @return Status
  virtual Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_KERNELS_TENSOR_OP_H_
