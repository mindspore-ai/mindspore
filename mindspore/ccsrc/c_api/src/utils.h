/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_C_API_SRC_UTILS_H_
#define MINDSPORE_CCSRC_C_API_SRC_UTILS_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <nlohmann/json.hpp>
#include "base/base.h"
#include "base/base_ref.h"
#include "ir/dtype/type_id.h"
#include "c_api/src/resource_manager.h"
#include "c_api/include/context.h"
#include "c_api/include/node.h"
#include "c_api/src/common.h"

const std::map<DTypeFormat, std::vector<std::string>> kDTypeFmtEnumToStrMap = {
  {None_None, {"", ""}},
  {None_Default, {"", "DefaultFormat"}},
  {BOOL_None, {"bool", ""}},
  {BOOL_Default, {"bool", "DefaultFormat"}},
  {BOOL_5HD, {"bool", "NC1HWC0"}},
  {BOOL_FracZ, {"bool", "FRACTAL_Z"}},
  {BOOL_FracNZ, {"bool", "FRACTAL_NZ"}},
  {BOOL_C1HWNCoC0, {"bool", "C1HWNCoC0"}},
  {BOOL_NCHW, {"bool", "NCHW"}},
  {BOOL_NHWC, {"bool", "NHWC"}},
  {BOOL_HWCN, {"bool", "HWCN"}},
  {BOOL_NDHWC, {"bool", "NDHWC"}},
  {BOOL_ChannelLast, {"bool", "ChannelLast"}},
  {BOOL_Default_Tuple, {"bool", "DefaultFormat", "tuple"}},
  {BOOL_Default_List, {"bool", "DefaultFormat", "list"}},
  {I8_None, {"int8", ""}},
  {I8_Default, {"int8", "DefaultFormat"}},
  {I8_5HD, {"int8", "NC1HWC0"}},
  {I8_FracZ, {"int8", "FRACTAL_Z"}},
  {I8_FracNZ, {"int8", "FRACTAL_NZ"}},
  {I8_C1HWNCoC0, {"int8", "C1HWNCoC0"}},
  {I8_NCHW, {"int8", "NCHW"}},
  {I8_NHWC, {"int8", "NHWC"}},
  {I8_HWCN, {"int8", "HWCN"}},
  {I8_NDHWC, {"int8", "NDHWC"}},
  {I8_NCDHW, {"int8", "NCDHW"}},
  {I8_ChannelLast, {"int8", "ChannelLast"}},
  {I8_NDC1HWC0, {"int8", "NDC1HWC0"}},
  {I8_NC1HWC0, {"int8", "NC1HWC0"}},
  {I8_Default_Tuple, {"int8", "DefaultFormat", "tuple"}},
  {I8_Default_List, {"int8", "DefaultFormat", "list"}},
  {U8_None, {"uint8", ""}},
  {U8_Default, {"uint8", "DefaultFormat"}},
  {U8_5HD, {"uint8", "NC1HWC0"}},
  {U8_FracZ, {"uint8", "FRACTAL_Z"}},
  {U8_FracNZ, {"uint8", "FRACTAL_NZ"}},
  {U8_C1HWNCoC0, {"uint8", "C1HWNCoC0"}},
  {U8_NCHW, {"uint8", "NCHW"}},
  {U8_NHWC, {"uint8", "NHWC"}},
  {U8_HWCN, {"uint8", "HWCN"}},
  {U8_NDHWC, {"uint8", "NDHWC"}},
  {U8_NCDHW, {"uint8", "NCDHW"}},
  {U8_ChannelLast, {"uint8", "ChannelLast"}},
  {U8_NDC1HWC0, {"uint8", "NDC1HWC0"}},
  {U8_NC1HWC0, {"uint8", "NC1HWC0"}},
  {U8_Default_Tuple, {"uint8", "DefaultFormat", "tuple"}},
  {U8_Default_List, {"uint8", "DefaultFormat", "list"}},
  {I16_None, {"int16", ""}},
  {I16_Default, {"int16", "DefaultFormat"}},
  {I16_5HD, {"int16", "NC1HWC0"}},
  {I16_FracZ, {"int16", "FRACTAL_Z"}},
  {I16_FracNZ, {"int16", "FRACTAL_NZ"}},
  {I16_C1HWNCoC0, {"int16", "C1HWNCoC0"}},
  {I16_NCHW, {"int16", "NCHW"}},
  {I16_NHWC, {"int16", "NHWC"}},
  {I16_HWCN, {"int16", "HWCN"}},
  {I16_NDHWC, {"int16", "NDHWC"}},
  {I16_ChannelLast, {"int16", "ChannelLast"}},
  {I16_Default_Tuple, {"int16", "DefaultFormat", "tuple"}},
  {I16_Default_List, {"int16", "DefaultFormat", "list"}},
  {U16_None, {"uint16", ""}},
  {U16_Default, {"uint16", "DefaultFormat"}},
  {U16_5HD, {"uint16", "NC1HWC0"}},
  {U16_FracZ, {"uint16", "FRACTAL_Z"}},
  {U16_FracNZ, {"uint16", "FRACTAL_NZ"}},
  {U16_C1HWNCoC0, {"uint16", "C1HWNCoC0"}},
  {U16_NCHW, {"uint16", "NCHW"}},
  {U16_NHWC, {"uint16", "NHWC"}},
  {U16_HWCN, {"uint16", "HWCN"}},
  {U16_NDHWC, {"uint16", "NDHWC"}},
  {U16_ChannelLast, {"uint16", "ChannelLast"}},
  {U16_Default_Tuple, {"uint16", "DefaultFormat", "tuple"}},
  {U16_Default_List, {"uint16", "DefaultFormat", "list"}},
  {I32_None, {"int32", ""}},
  {I32_Default, {"int32", "DefaultFormat"}},
  {I32_5HD, {"int32", "NC1HWC0"}},
  {I32_FracZ, {"int32", "FRACTAL_Z"}},
  {I32_FracNZ, {"int32", "FRACTAL_NZ"}},
  {I32_C1HWNCoC0, {"int32", "C1HWNCoC0"}},
  {I32_NCHW, {"int32", "NCHW"}},
  {I32_NHWC, {"int32", "NHWC"}},
  {I32_HWCN, {"int32", "HWCN"}},
  {I32_NDHWC, {"int32", "NDHWC"}},
  {I32_NDC1HWC0, {"int32", "NDC1HWC0"}},
  {I32_NCDHW, {"int32", "NCDHW"}},
  {I32_ChannelLast, {"int32", "ChannelLast"}},
  {I32_Default_Tuple, {"int32", "DefaultFormat", "tuple"}},
  {I32_Default_List, {"int32", "DefaultFormat", "list"}},
  {U32_None, {"uint32", ""}},
  {U32_Default, {"uint32", "DefaultFormat"}},
  {U32_5HD, {"uint32", "NC1HWC0"}},
  {U32_FracZ, {"uint32", "FRACTAL_Z"}},
  {U32_FracNZ, {"uint32", "FRACTAL_NZ"}},
  {U32_C1HWNCoC0, {"uint32", "C1HWNCoC0"}},
  {U32_NCHW, {"uint32", "NCHW"}},
  {U32_NHWC, {"uint32", "NHWC"}},
  {U32_HWCN, {"uint32", "HWCN"}},
  {U32_NDHWC, {"uint32", "NDHWC"}},
  {U32_ChannelLast, {"uint32", "ChannelLast"}},
  {U32_Default_Tuple, {"uint32", "DefaultFormat", "tuple"}},
  {U32_Default_List, {"uint32", "DefaultFormat", "list"}},
  {I64_None, {"int64", ""}},
  {I64_Default, {"int64", "DefaultFormat"}},
  {I64_5HD, {"int64", "NC1HWC0"}},
  {I64_FracZ, {"int64", "FRACTAL_Z"}},
  {I64_FracNZ, {"int64", "FRACTAL_NZ"}},
  {I64_C1HWNCoC0, {"int64", "C1HWNCoC0"}},
  {I64_NCHW, {"int64", "NCHW"}},
  {I64_NHWC, {"int64", "NHWC"}},
  {I64_HWCN, {"int64", "HWCN"}},
  {I64_NDHWC, {"int64", "NDHWC"}},
  {I64_ChannelLast, {"int64", "ChannelLast"}},
  {I64_Default_Tuple, {"int64", "DefaultFormat", "tuple"}},
  {I64_Default_List, {"int64", "DefaultFormat", "list"}},
  {U64_None, {"uint64", ""}},
  {U64_Default, {"uint64", "DefaultFormat"}},
  {U64_5HD, {"uint64", "NC1HWC0"}},
  {U64_FracZ, {"uint64", "FRACTAL_Z"}},
  {U64_FracNZ, {"uint64", "FRACTAL_NZ"}},
  {U64_C1HWNCoC0, {"uint64", "C1HWNCoC0"}},
  {U64_NCHW, {"uint64", "NCHW"}},
  {U64_NHWC, {"uint64", "NHWC"}},
  {U64_HWCN, {"uint64", "HWCN"}},
  {U64_NDHWC, {"uint64", "NDHWC"}},
  {U64_ChannelLast, {"uint64", "ChannelLast"}},
  {U64_Default_Tuple, {"uint64", "DefaultFormat", "tuple"}},
  {U64_Default_List, {"uint64", "DefaultFormat", "list"}},
  {F16_None, {"float16", ""}},
  {F16_Default, {"float16", "DefaultFormat"}},
  {F16_5HD, {"float16", "NC1HWC0"}},
  {F16_FracZ, {"float16", "FRACTAL_Z"}},
  {F16_FracNZ, {"float16", "FRACTAL_NZ"}},
  {F16_C1HWNCoC0, {"float16", "C1HWNCoC0"}},
  {F16_NCHW, {"float16", "NCHW"}},
  {F16_NHWC, {"float16", "NHWC"}},
  {F16_HWCN, {"float16", "HWCN"}},
  {F16_NDHWC, {"float16", "NDHWC"}},
  {F16_NCDHW, {"float16", "NCDHW"}},
  {F16_DHWCN, {"float16", "DHWCN"}},
  {F16_NDC1HWC0, {"float16", "NDC1HWC0"}},
  {F16_FRACTAL_Z_3D, {"float16", "FRACTAL_Z_3D"}},
  {F16_FracZNLSTM, {"float16", "FRACTAL_ZN_LSTM"}},
  {F16_FracZNRNN, {"float16", "FRACTAL_ZN_RNN"}},
  {F16_ND_RNNBIAS, {"float16", "ND_RNN_BIAS"}},
  {F16_ChannelLast, {"float16", "ChannelLast"}},
  {F16_Default_Tuple, {"float16", "DefaultFormat", "tuple"}},
  {F16_Default_List, {"float16", "DefaultFormat", "list"}},
  {F32_None, {"float32", ""}},
  {F32_Default, {"float32", "DefaultFormat"}},
  {F32_5HD, {"float32", "NC1HWC0"}},
  {F32_FracZ, {"float32", "FRACTAL_Z"}},
  {F32_FracNZ, {"float32", "FRACTAL_NZ"}},
  {F32_C1HWNCoC0, {"float32", "C1HWNCoC0"}},
  {F32_NCHW, {"float32", "NCHW"}},
  {F32_NHWC, {"float32", "NHWC"}},
  {F32_HWCN, {"float32", "HWCN"}},
  {F32_NDHWC, {"float32", "NDHWC"}},
  {F32_NCDHW, {"float32", "NCDHW"}},
  {F32_DHWCN, {"float32", "DHWCN"}},
  {F32_NDC1HWC0, {"float32", "NDC1HWC0"}},
  {F32_FRACTAL_Z_3D, {"float32", "FRACTAL_Z_3D"}},
  {F32_FracZNLSTM, {"float32", "FRACTAL_ZN_LSTM"}},
  {F32_FracZNRNN, {"float32", "FRACTAL_ZN_RNN"}},
  {F32_ND_RNNBIAS, {"float32", "ND_RNN_BIAS"}},
  {F32_ChannelLast, {"float32", "ChannelLast"}},
  {F32_Default_Tuple, {"float32", "DefaultFormat", "tuple"}},
  {F32_Default_List, {"float32", "DefaultFormat", "list"}},
  {F64_None, {"float64", ""}},
  {F64_Default, {"float64", "DefaultFormat"}},
  {F64_5HD, {"float64", "NC1HWC0"}},
  {F64_FracZ, {"float64", "FRACTAL_Z"}},
  {F64_FracNZ, {"float64", "FRACTAL_NZ"}},
  {F64_C1HWNCoC0, {"float64", "C1HWNCoC0"}},
  {F64_NCHW, {"float64", "NCHW"}},
  {F64_NHWC, {"float64", "NHWC"}},
  {F64_HWCN, {"float64", "HWCN"}},
  {F64_NDHWC, {"float64", "NDHWC"}},
  {F64_ChannelLast, {"float64", "ChannelLast"}},
  {F64_Default_Tuple, {"float64", "DefaultFormat", "tuple"}},
  {F64_Default_List, {"float64", "DefaultFormat", "list"}},
  {C64_Default, {"complex64", "DefaultFormat"}},
  {C128_Default, {"complex128", "DefaultFormat"}},
};

void ConvertConstScalarInputToTensor(const AnfNodePtr &input_node);

std::vector<TensorPtr> ConvertOutputToTensor(const mindspore::BaseRef &output);

AbstractBasePtr GetAbstract(const TypePtr &type, const int64_t shape[], size_t shape_size, bool is_param = false);

STATUS CheckCustomOpInfo(const CustomOpInfo &info);

nlohmann::json ConvertOpInfoToJson(const CustomOpInfo &info);

size_t GetMaxMallocSize();

#define MS_ERROR_IF_FALSE_W_RET_N_LOG(condition, val, message) \
  do {                                                         \
    if (!(condition)) {                                        \
      MS_LOG(ERROR) << message;                                \
      return val;                                              \
    }                                                          \
  } while (0)

#define MS_ERROR_IF_TRUE_W_RET_N_LOG(condition, val, message) \
  do {                                                        \
    if ((condition)) {                                        \
      MS_LOG(ERROR) << message;                               \
      return val;                                             \
    }                                                         \
  } while (0)
#endif  // MINDSPORE_CCSRC_C_API_SRC_UTILS_H_
