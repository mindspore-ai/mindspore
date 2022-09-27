/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_UTIL_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_UTIL_H_

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "securec/include/securec.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "include/transform/graph_ir/types.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace transform {
class TransformUtil {
 public:
  /*
   * Parameters:
   *     type: [MeDataType] the data type for ME tensor
   * Return：
   *     [GeDataType] the data type for ge tensor
   * */
  static std::vector<int64_t> ConvertIntToList(int64_t data, int size);

  /*
   * Parameters:
   *     type: [MeDataType] the data type for ME tensor
   * Return：
   *     [GeDataType] the data type for ge tensor
   * */
  static GeDataType ConvertDataType(const MeDataType &type);

  /*
   * Parameters:
   *     type: [string] the data format in ME op
   * Return：
   *     [GeFormat] the data format for ge tensor
   * */
  static GeFormat ConvertFormat(const string &format, const size_t shape_size);

  /*
   * Parameters:
   *     type: [MeDataType] the data type for ME tensor
   * Return：
   *     [size_t] the buff size for the type in ME
   * */
  static size_t GetDataTypeSize(const MeDataType &type);

  /*
   * Parameters:
   *     tensor: [MeTensorPtr] the me tensor to get description from
   *     format: [string] the data format in ME
   *     is_input: [bool] whether the tensor is used as input, default:false
   * Return：
   *     [shared_ptr<GeTensorDesc>] the shared pointer of ge tensor description
   * */
  static std::shared_ptr<GeTensorDesc> GetGeTensorDesc(const ShapeVector &shape, const MeDataType &me_type,
                                                       const std::string &format, const ShapeVector &ori_shape = {},
                                                       const std::string &ori_format = {});

  /*
   * Parameters:
   *     tensor: [MeTensor] the data tensor in ME
   *     format: [string] the data format in ME op
   *     is_input: [bool] whether the tensor is used as input, default:false
   * Return：
   *     [GeTensor] the data tensor in GE
   * */
  static GeTensorPtr ConvertTensor(const MeTensorPtr &tensor, const std::string &format);

  /*
   * Parameters:
   *     me_tensors: [vector<MeTensorPtr>] the data tensors in ME
   *     format: [string] the data format in ME op
   * Return：
   *     [std::vector<GeTensorPtr>] the data tensors in GE
   * */
  static std::vector<GeTensorPtr> ConvertInputTensors(const std::vector<MeTensorPtr> &me_tensors,
                                                      const std::string &format);

  /*
   * Parameters:
   *     tensor: [GeTensor] the data tensor in GE
   * Return：
   *     [MeTensor] the data tensor in ME
   * */
  static MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor);

  /*
   * Parameters:
   *     tensor: [GeTensor] the data tensor in GE
   *     me_type: [TypeId] the type of created Me tensor
   * Return：
   *     [MeTensor] the data tensor in ME
   * */
  static MeTensorPtr ConvertGeTensor(const GeTensorPtr &tensor, const TypeId &me_type);

  /*
   * Parameters:
   *     tensor: [GeTensor] the data tensor in GE
   *     request_dims [ShapeVector] the output Me tensors must adjust to this shapes
   * Return：
   *     [MeTensor] the data tensor in ME
   * */
  static MeTensorPtr ConvertGeTensor(GeTensorPtr ge_tensor, const ShapeVector &request_dims);
  /*
   * Parameters:
   *     ge_tensors: [std::vector<GeTensorPtr>] the data tensor in GE
   *     request_dims [std::vector<ShapeVector>] the output Me tensors must adjust to this shapes
   * Return：
   *     [std::vector<MeTensorPtr>] the data tensor in ME
   * */
  static std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors,
                                                   const std::vector<ShapeVector> &request_dims);
  /*
   * Parameters:
   *     ge_tensors: [std::vector<GeTensorPtr>] the data tensor in GE
   * Return：
   *     [std::vector<MeTensorPtr>] the data tensor in ME
   * */
  static std::vector<MeTensorPtr> ConvertGeTensors(const std::vector<GeTensorPtr> &ge_tensors);
  /*
   * Parameters:
   *     ge_tensor: [GeTensor] the data tensor in GE
   *     me_dims: [ShapeVector] the shape of created Me tensor
   *     me_type: [TypeId] the type of created Me tensor
   * Return：
   *     [MeTensor] the data tensor in ME
   * */
  static MeTensorPtr GenerateMeTensor(const GeTensorPtr &ge_tensor, const ShapeVector &me_dims, const TypeId &me_type);
  /*
   * Parameters:
   *     type: [GeDataType] the ge tensor data type
   * Return：
   *     [MeDataType] the me tensor data type
   * */
  static MeDataType ConvertGeDataType(const GeDataType &type);

  /*
   * Parameters:
   *     me_dims: [ShapeVector] the me shape
   * Return：
   *     [GeShape] the ge shape
   * */
  static GeShape ConvertMeShape(const ShapeVector &me_dims);

  /*
   * Parameters:
   *     ge_shape: [GeShape] the ge shape
   * Return：
   *     [vector<int>] the me shape
   * */
  static ShapeVector ConvertGeShape(const GeShape &ge_shape);

  /* Function:
   *     Convert GeShape to Me request shape, Support pattern:
   *         {1, x, 1, 1} --> {x}
   *         {x, 1, 1, 1} --> {x}
   *         {x, x, 1, 1} --> {x, x}
   *         {x, x, x, 1} --> {x, x, x}
   *         {x, x, x, x} --> {x, x, x, x}
   *      If unmatch upon patterns, return original ge dims
   * Parameters:
   *     ge_shape: [GeShape] the ge shape
   *     request_dims: [vector<int>] request dims
   * Return：
   *     [vector<int>] the me shape
   * */
  static ShapeVector ConvertGeShape(const GeShape &ge_shape, const ShapeVector &request_dims);

  /*
   * Parameters:
   *     vec: [ShapeVector] the vector to print
   * Return：
   *     [string] value string
   * */
  template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  static std::string PrintVector(const std::vector<T> &vec) {
    const int MAX_PRINT_NUM = 100;
    std::stringstream ss;
    ss << "{ ";
    int i = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
      ss << std::to_string(*it) << ", ";
      i++;
      if (i >= MAX_PRINT_NUM) {
        break;
      }
    }

    if (i >= MAX_PRINT_NUM) {
      ss << "... to be continue}";
    } else {
      ss << "}";
    }
    return ss.str();
  }

  /*
   * Parameters:
   *     ge_tensor: [GeTensorPtr] the ge tensor
   * Return：
   *     [stringstream] value string
   * */
  static std::string PrintGeTensor(const GeTensorPtr ge_tensor);

  /*
   * Parameters:
   *     data: [uint8_t *] the ge tensor data pointer
   *     size: [size_t] the ge tensor data bytes
   * Return：
   *     [shared_ptr<std::vector<T>]  vector pointer
   * */
  template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  static std::vector<T> MakeVector(const uint8_t *const data, size_t size) {
    auto dest = std::vector<T>(size / sizeof(T));
    if (data == nullptr) {
      return dest;
    }

    errno_t ret = memcpy_s(dest.data(), dest.size() * sizeof(T), data, size);
    if (EOK != ret) {
      return std::vector<T>();
    }
    return dest;
  }

  /*
   * Parameters:
   *     anf_name: [string] the anf node name
   * Return：
   *     [string] operator name
   * */
  static std::string NormOpName(const std::string &anf_name);
};
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_UTIL_H_
