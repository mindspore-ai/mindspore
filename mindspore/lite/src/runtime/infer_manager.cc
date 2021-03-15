/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/runtime/infer_manager.h"
#include "src/common/tensor_util.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "nnacl/errorcode.h"
#include "src/tensorlist.h"

namespace mindspore {
namespace lite {
int KernelInferShape(const std::vector<lite::Tensor *> &inputs, std::vector<lite::Tensor *> *outputs,
                     OpParameter *parameter) {
  std::vector<TensorC *> in_tensors;
  std::vector<TensorC *> out_tensors;
  int ret = 0;

  ret = GenerateInTensorC(parameter, inputs, outputs, &in_tensors);
  if (ret != RET_OK) {
    FreeAllTensorC(&in_tensors);
    return RET_ERROR;
  }

  ret = GenerateOutTensorC(parameter, inputs, outputs, &out_tensors);
  if (ret != RET_OK) {
    FreeAllTensorC(&in_tensors);
    FreeAllTensorC(&out_tensors);
    return RET_ERROR;
  }

  auto infer_shape_func = GetInferFunc(parameter->type_);
  if (infer_shape_func == nullptr) {
    MS_LOG(ERROR) << "Get infershape func failed! type:" << PrimitiveCurVersionTypeName(parameter->type_);
    return RET_ERROR;
  }
  ret = infer_shape_func(static_cast<TensorC **>(in_tensors.data()), in_tensors.size(), out_tensors.data(),
                         out_tensors.size(), parameter);

  if (ret == RET_OK) {
    for (size_t i = 0; i < out_tensors.size(); i++) {
      if (reinterpret_cast<TensorListC *>(out_tensors.at(i))->data_type_ == TypeIdC::kObjectTypeTensorType) {
        auto *tensor_list_c = reinterpret_cast<TensorListC *>(out_tensors.at(i));
        auto *tensor_list = reinterpret_cast<TensorList *>(outputs->at(i));
        tensor_list->set_shape({static_cast<int>(tensor_list_c->element_num_)});
        auto tensor_shape = std::vector<std::vector<int>>(
          tensor_list_c->element_num_,
          std::vector<int>(tensor_list_c->element_shape_,
                           tensor_list_c->element_shape_ + tensor_list_c->element_shape_size_));
        tensor_list->MallocTensorListData(static_cast<TypeId>(tensor_list_c->data_type_), tensor_shape);
        TensorListC2TensorList(tensor_list_c, tensor_list);
      } else {
        TensorC2Tensor(out_tensors.at(i), outputs->at(i));
      }
    }
  } else {
    SetOutputTensorAttr(out_tensors, outputs);
  }

  FreeAllTensorC(&in_tensors);
  FreeAllTensorC(&out_tensors);
  if (ret == NNACL_INFER_INVALID) {
    return RET_INFER_INVALID;
  } else if (ret != NNACL_OK) {
    return RET_INFER_ERR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
