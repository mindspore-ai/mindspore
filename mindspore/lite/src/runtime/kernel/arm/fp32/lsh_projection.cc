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
#include "src/runtime/kernel/arm/fp32/lsh_projection.h"

#include "include/errorcode.h"
#include "src/common/string_util.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LshProjection;

namespace mindspore::kernel {
int LshProjectionCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LshProjectionCPUKernel::ReSize() { return RET_OK; }

int LshProjectionCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }

  auto input_tensor0 = in_tensors_.at(0);
  auto input_tensor1 = in_tensors_.at(1);
  auto out_tensor0 = out_tensors_.at(0);

  hash = reinterpret_cast<float *>(input_tensor0->MutableData());
  in_data = reinterpret_cast<char *>(input_tensor1->MutableData());
  weight = in_tensors_.size() == 2 ? nullptr : reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
  output = reinterpret_cast<int32_t *>(out_tensor0->MutableData());

  const size_t seed_size = sizeof(float);
  const size_t input_item_size =
    input_tensor1->ElementsNum() * sizeof(input_tensor1->data_type()) / input_tensor1->DimensionSize(0);
  const size_t key_size = seed_size + input_item_size;
  lsh_param_->seed_size_ = seed_size;
  lsh_param_->in_item_size_ = input_item_size;
  lsh_param_->key_size_ = key_size;
  lsh_param_->in_item_num_ = input_tensor1->DimensionSize(0);
  memcpy(lsh_param_->hash_shape_, input_tensor0->shape().data(), sizeof(int) * input_tensor0->shape().size());

  elements_num_ = input_tensor0->DimensionSize(0);
  count_unit_ = thread_num_ > 1 ? UP_DIV(elements_num_, thread_num_) : elements_num_;
  ret = ParallelLaunch(this->context_->thread_pool_, LshProjectionRun, this, thread_num_);
  return ret;
}

int LshProjectionRun(void *cdata, int task_id) {
  auto lsh_projection = reinterpret_cast<LshProjectionCPUKernel *>(cdata);
  lsh_projection->DoExecute(task_id);
  return RET_OK;
}

int LshProjectionCPUKernel::DoExecute(int task_id) {
  int64_t real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  lsh_param_->real_dst_count = real_dst_count;
  lsh_param_->task_id_ = task_id;
  lsh_param_->count_unit_ = count_unit_;
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }

  switch (lsh_param_->lsh_type_) {
    case schema::LshProjectionType_SPARSE:
      LshProjectionSparse(hash, in_data, weight, output, lsh_param_);
      break;
    case schema::LshProjectionType_DENSE:
      LshProjectionDense(hash, in_data, weight, output, lsh_param_);
      break;
    default:
      return RET_ERROR;
  }
  return RET_OK;
}

int LshProjectionCPUKernel::GetSignBit(char *in_data, float *weight, float seed, LshProjectionParameter *para) {
  double score = 0.0;
  for (int i = 0; i < para->in_item_num_; i++) {
    char *key = static_cast<char *>(context_->allocator->Malloc(lsh_param_->key_size_));
    if (key == nullptr) {
      MS_LOG(ERROR) << "malloc key failed.";
      return RET_ERROR;
    }
    memcpy(key, &seed, para->seed_size_);
    memcpy(key + para->seed_size_, in_data, para->in_item_size_);
    in_data += para->in_item_size_;
    int64_t hash_i = static_cast<int64_t>(mindspore::lite::StringHash64(key, para->key_size_));
    double hash_d = static_cast<double>(hash_i);
    if (weight == nullptr) {
      score += hash_d;
    } else {
      score += weight[i] * hash_d;
    }
    context_->allocator->Free(key);
  }
  return (score > 0) ? 1 : 0;
}

void LshProjectionCPUKernel::LshProjectionSparse(float *hash, char *in_data, float *weight, int32_t *output,
                                                 LshProjectionParameter *para) {
  int start = para->task_id_ * para->count_unit_;
  int end = start + para->real_dst_count;
  for (int i = start; i < end; i++) {
    int32_t hash_sign = 0;
    for (int j = 0; j < para->hash_shape_[1]; j++) {
      int bit = GetSignBit(in_data, weight, hash[i * para->hash_shape_[1] + j], para);
      hash_sign = (hash_sign << 1) | bit;
    }
    output[i] = hash_sign + i * (1 << para->hash_shape_[1]);
  }
}

void LshProjectionCPUKernel::LshProjectionDense(float *hash, char *in_data, float *weight, int32_t *output,
                                                LshProjectionParameter *para) {
  int start = para->task_id_ * para->count_unit_;
  int end = start + para->real_dst_count;
  for (int i = start; i < end; i++) {
    for (int j = 0; j < para->hash_shape_[1]; j++) {
      output[i * para->hash_shape_[1] + j] = GetSignBit(in_data, weight, hash[i * para->hash_shape_[1] + j], para);
    }
  }
}

kernel::LiteKernel *CpuLshProjectionFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                      const std::vector<lite::Tensor *> &outputs,
                                                      OpParameter *op_parameter, const lite::InnerContext *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    free(op_parameter);
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_LshProjection);
  auto *kernel = new (std::nothrow) LshProjectionCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new LshProjectionCPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LshProjection, CpuLshProjectionFp32KernelCreator)

}  // namespace mindspore::kernel
