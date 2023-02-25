/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_INFER_TENSOR_H_
#define MINDSPORE_LITE_INFER_TENSOR_H_

#include <vector>
#include <string>
#include <memory>

#include "core/type_id.h"
#include "src/tensor.h"

namespace mindspore::infer::abstract {
using Tensor = mindspore::lite::Tensor;
// struct LiteQuantParam {
//   double scale;
//   int32_t zeroPoint;
//   float var_corr{1};
//   float mean_corr{0};
//   bool inited{false};
//   std::vector<float> clusters{};
//   int bitNum{8};
//   int roundType{1};
//   int multiplier{1};
//   int dstDtype{32};
//   // dynamic range
//   double min{-255.0};
//   double max{255.0};
// };

// enum CompressType {
//   kNoCompression = 0,
//   kIndexing = 1,
//   kSparse = 2,
//   kFSE = 3,
//   kBitPacking = 4,
//   kFSEInt = 5,
//   kFSEInfer = 6
// };

// enum Category {
//   CONST_TENSOR,  // weight tensor
//   CONST_SCALAR,  // weight scalar
//   VAR,           // activation tensor
//   GRAPH_INPUT,
//   GRAPH_OUTPUT,
// };

// class mindspore::Allocator;
// using AllocatorPtr = std::shared_ptr<mindspore::Allocator>;

// class Tensor : public std::enable_shared_from_this<Tensor> {
//  public:
//   virtual ~Tensor() = default;

//   virtual bool operator==(const Tensor &tensor) = 0;

//   virtual void set_tensor_name(const std::string &name) = 0;

//   virtual std::string tensor_name() const = 0;

//   virtual TypeId data_type() const = 0;

//   virtual void set_data_type(TypeId data_type) = 0;

//   virtual std::vector<int> shape() const = 0;

//   virtual void set_shape(const std::vector<int> &shape) = 0;

//   virtual size_t Size() const = 0;

//   virtual void set_allocator(AllocatorPtr allocator) = 0;

//   virtual AllocatorPtr allocator() const = 0;

//   virtual int MallocData(const AllocatorPtr allocator = nullptr) = 0;

//   virtual void FreeData() = 0;

//   virtual void *MutableData() = 0;

//   virtual void *ReallocData() = 0;

//   virtual void *data() = 0;

//   virtual void *data() const = 0;

//   // note: in the case of that old_data is valid, set_data just releases the ownership of it but not frees it. Of
//   //       course, you can call FreeData before calling set_data to ensure the data can be freed by current tensor.
//   virtual void set_data(void *data, bool own_data = true) = 0;

//   virtual void set_device_data(void *data) = 0;

//   virtual void *device_data() const = 0;

//   virtual Category category() const = 0;

//   virtual void set_category(Category category) = 0;

//   virtual void set_format(mindspore::Format format) = 0;

//   virtual mindspore::Format format() const = 0;
//   virtual int ref_count() const = 0;

//   virtual int init_ref_count() const = 0;

//   virtual void set_ref_count(int ref_count) = 0;

//   virtual void set_init_ref_count(int ref_count) = 0;

//   virtual void ResetRefCount() = 0;

//   virtual void IncRefCount() = 0;

//   virtual void DecRefCount() = 0;

//   virtual std::string ToString() const = 0;

//   virtual void AddQuantParam(const LiteQuantParam &quant_param) = 0;

//   virtual void ClearQuantParam() = 0;

//   virtual std::vector<LiteQuantParam> quant_params() const = 0;

//   virtual void set_quant_params(std::vector<LiteQuantParam>) = 0;

//   virtual std::vector<float> quant_clusters() const = 0;

//   virtual void set_quant_clusters(const std::vector<float> &clusters) = 0;

//   virtual bool IsConst() const = 0;

//   virtual bool IsScalar() const = 0;

//   virtual bool IsGraphInput() const = 0;

//   virtual bool IsGraphOutput() const = 0;

//   virtual void Prepare() = 0;

//   virtual bool IsReady() const = 0;

//   virtual bool own_data() const = 0;

//   virtual void set_own_data(bool own_data) = 0;

//   // template <typename T>
//   // int Scale(float scale) {
//   //   T cast_scale = static_cast<T>(scale);
//   //   auto data = reinterpret_cast<T *>(data_);
//   //   if (data == nullptr) {
//   //     return RET_ERROR;
//   //   }
//   //   int length = ElementsNum();
//   //   for (int i = 0; i < length; i++) {
//   //     data[i] *= cast_scale;
//   //   }
//   //   scale_ *= scale;
//   //   return RET_OK;
//   // }

//   virtual float get_scale() const = 0;

//   virtual void set_scale(float scale) = 0;

//   virtual CompressType get_compress_type() const = 0;

//   virtual void set_compress_type(CompressType compression_type) = 0;

//   virtual void set_compressed_size(size_t compressed_size) = 0;

//   virtual bool IsScale() const = 0;
// };
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_TENSOR_H_
