/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "src/litert/infer_manager.h"
#include <algorithm>
#include <set>
#include <string>
#include <memory>
#include "src/common/prim_util.h"
#include "src/common/tensor_util.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"
#include "src/tensorlist.h"
#include "include/registry/register_kernel_interface.h"
#include "src/litert/kernel_registry.h"

namespace mindspore {
namespace lite {
namespace {
static const size_t kNumMaxMallocSize = GetMaxMallocSize();
}

int KernelInferShape(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                     const void *primitive, std::set<std::string> &&providers, int schema_version,
                     const kernel::Kernel *kernel) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  if (primitive == nullptr && kernel == nullptr) {
    return RET_NOT_SUPPORT;
  }
  std::shared_ptr<kernel::KernelInterface> kernel_interface = nullptr;
  bool is_custom_node = false;
  if (kernel == nullptr) {
    if (IsCustomNode(primitive, schema_version)) {
      is_custom_node = true;
    }
  } else if (kernel->type() == schema::PrimitiveType_Custom) {
    is_custom_node = true;
  }
  if (is_custom_node) {
    kernel_interface = registry::RegisterKernelInterface::GetKernelInterface(
      "", static_cast<const schema::Primitive *>(primitive), kernel);
  } else {
    for (auto &&provider : providers) {
      kernel_interface = registry::RegisterKernelInterface::GetKernelInterface(
        provider, static_cast<const schema::Primitive *>(primitive), kernel);
      if (kernel_interface != nullptr) {
        break;
      }
    }
  }

  if (kernel_interface == nullptr) {
    return RET_NOT_SUPPORT;
  }
  std::vector<mindspore::MSTensor> in_tensors;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(in_tensors), [](lite::Tensor *tensor) {
    return mindspore::MSTensor(std::make_shared<LiteTensorImpl>(tensor));
  });
  std::vector<mindspore::MSTensor> out_tensors;
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(out_tensors), [](lite::Tensor *tensor) {
    return mindspore::MSTensor(std::make_shared<LiteTensorImpl>(tensor));
  });
  auto ret =
    kernel_interface->Infer(&in_tensors, &out_tensors, static_cast<const schema::Primitive *>(primitive), kernel);
  if (ret == kLiteInferInvalid) {
    for (auto output : outputs) {
      output->set_shape({-1});
    }
    return RET_INFER_INVALID;
  }
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "op_type: " << GetPrimitiveTypeName(primitive, schema_version) << " infer fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
#endif
  return lite::RET_NOT_SUPPORT;
}

int CheckInfershapeResult(int result, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, OpParameter *parameter) {
  if (result == NNACL_INFER_INVALID) {
    return RET_INFER_INVALID;
  } else if (result != NNACL_OK) {
    if (result == NNACL_FORMAT_ERROR) {
      MS_LOG(ERROR) << "Unexpected input format " << inputs[0]->format();
    }
    return RET_INFER_ERR;
  }

  for (auto output : outputs) {
    auto element_num = output->ElementsNum();
    if (element_num < 0) {
      continue;
    }
    if (static_cast<size_t>(element_num) >= kNumMaxMallocSize / sizeof(int64_t)) {
      MS_LOG(ERROR) << "The size of output tensor is too big, output size: " << output->ElementsNum();
      return RET_INFER_ERR;
    }
  }

  parameter->is_zero_shape_ = true;
  size_t zero_shape_num = 0;
  for (auto tensor : outputs) {
    for (size_t i = 0; i < tensor->shape().size(); i++) {
      if (tensor->shape()[i] == 0) {
        zero_shape_num++;
        break;
      }
    }
  }
  if (zero_shape_num != outputs.size()) {
    parameter->is_zero_shape_ = false;
  }
  return RET_OK;
}

int KernelInferShape(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                     OpParameter *parameter, std::shared_ptr<Allocator> allocator) {
  MS_CHECK_TRUE_MSG(parameter != nullptr, RET_ERROR, "parameter is nullptr.");
  if (inputs.empty()) {
    MS_LOG(ERROR) << "No input!";
    return RET_ERROR;
  }
  std::vector<TensorC *> in_tensors;
  std::vector<TensorC *> out_tensors;
  if (parameter->type_ == static_cast<int>(schema::PrimitiveType_PartialFusion) ||
      parameter->type_ == static_cast<int>(schema::PrimitiveType_Switch) ||
      parameter->type_ == static_cast<int>(schema::PrimitiveType_Call) ||
      parameter->type_ == static_cast<int>(schema::PrimitiveType_SwitchLayer)) {
    MS_LOG(INFO) << "no need infer shape.";
    return RET_OK;
  }

  int ret = GenerateInTensorC(inputs, &in_tensors, allocator);
  if (ret != RET_OK) {
    FreeInTensorC(&in_tensors, allocator);
    return RET_ERROR;
  }
  ret = GenerateOutTensorC(parameter, outputs, &out_tensors);
  if (ret != RET_OK) {
    FreeInTensorC(&in_tensors, allocator);
    FreeOutTensorC(&out_tensors, allocator);
    return RET_ERROR;
  }
  auto infer_shape_func = GetInferFunc(parameter->type_);
  if (infer_shape_func == nullptr) {
    MS_LOG(ERROR) << "Get infershape func failed! type:" << PrimitiveCurVersionTypeName(parameter->type_);
    FreeInTensorC(&in_tensors, allocator);
    FreeOutTensorC(&out_tensors, allocator);
    return RET_ERROR;
  }
  ret = infer_shape_func(static_cast<TensorC **>(in_tensors.data()), in_tensors.size(), out_tensors.data(),
                         out_tensors.size(), parameter);
  FreeInTensorC(&in_tensors, allocator);
  for (size_t i = 0; i < out_tensors.size(); i++) {
    if (out_tensors.at(i) == nullptr) {
      continue;
    }
    if (reinterpret_cast<TensorListC *>(out_tensors.at(i))->data_type_ == TypeIdC::kObjectTypeTensorType) {
      auto *tensor_list_c = reinterpret_cast<TensorListC *>(out_tensors.at(i));
      auto tensor_list = MallocTensorListDataAccordingToTensorListC(outputs.at(i), tensor_list_c);
      if (tensor_list == nullptr) {
        MS_LOG(ERROR) << "get as tensorlist failed";
        FreeOutTensorC(&out_tensors, allocator);
        return RET_ERROR;
      }
      auto tensor_ret = TensorListC2TensorList(tensor_list_c, tensor_list);
      if (tensor_ret != RET_OK) {
        MS_LOG(ERROR) << "TensorCList2TensorList failed";
        FreeOutTensorC(&out_tensors, allocator);
        return tensor_ret;
      }
    } else {
      if (out_tensors.at(i)->data_ != nullptr) {
        outputs.at(i)->set_own_data(true);
        outputs.at(i)->set_category(CONST_TENSOR);
      }
    }

    if (ret == NNACL_INFER_INVALID) {
      outputs.at(i)->set_shape({-1});
    }
  }
  FreeOutTensorC(&out_tensors, allocator);

  return CheckInfershapeResult(ret, inputs, outputs, parameter);
}
}  // namespace lite
}  // namespace mindspore
