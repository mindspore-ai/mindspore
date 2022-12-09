/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of types
 */

#ifndef CPU_KERNEL_TYPES_H
#define CPU_KERNEL_TYPES_H

#include <map>

namespace aicpu {
#ifdef VISIBILITY
#define AICPU_VISIBILITY __attribute__((visibility("default")))
#else
#define AICPU_VISIBILITY
#endif

enum DataType {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT8 = 4,            // uint8 type
  DT_INT32 = 3,            //
  DT_INT64 = 9,            // int64 type
  DT_UINT32 = 8,           // unsigned int32
  DT_UINT64 = 10,          // unsigned int64
  DT_BOOL = 12,            // bool type
  DT_DOUBLE = 11,          // double type
  DT_STRING = 13,          // string type
  DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  DT_COMPLEX64 = 16,       // complex64 type
  DT_COMPLEX128 = 17,      // complex128 type
  DT_QINT8 = 18,           // qint8 type
  DT_QINT16 = 19,          // qint16 type
  DT_QINT32 = 20,          // qint32 type
  DT_QUINT8 = 21,          // quint8 type
  DT_QUINT16 = 22,         // quint16 type
  DT_RESOURCE = 23,        // resource type
  DT_STRING_REF = 24,      // string ref type
  DT_DUAL = 25,            // dual output type
  DT_UNDEFINED             // Used to indicate a DataType field has not been set.
};

AICPU_VISIBILITY inline int GetSizeByDataType(DataType dataType) {
  const std::map<DataType, int> sizeMap = {
    {DT_FLOAT, 4},     {DT_FLOAT16, 2},     {DT_INT8, 1},      {DT_INT16, 2},         {DT_UINT16, 2},
    {DT_UINT8, 1},     {DT_INT32, 4},       {DT_INT64, 8},     {DT_UINT32, 4},        {DT_UINT64, 8},
    {DT_BOOL, 1},      {DT_DOUBLE, 8},      {DT_STRING, -1},   {DT_DUAL_SUB_INT8, 1}, {DT_DUAL_SUB_UINT8, 1},
    {DT_COMPLEX64, 8}, {DT_COMPLEX128, 16}, {DT_QINT8, 1},     {DT_QINT16, 2},        {DT_QINT32, 4},
    {DT_QUINT8, 1},    {DT_QUINT16, 2},     {DT_RESOURCE, -1}, {DT_STRING_REF, -1},   {DT_DUAL, 5}};
  auto iter = sizeMap.find(dataType);
  if (iter == sizeMap.end()) {
    return -1;
  }
  return iter->second;
}

enum Format {
  FORMAT_NCHW = 0,   // NCHW
  FORMAT_NHWC,       // NHWC
  FORMAT_ND,         // Nd Tensor
  FORMAT_NC1HWC0,    // NC1HWC0
  FORMAT_FRACTAL_Z,  // FRACTAL_Z
  FORMAT_NC1C0HWPAD,
  FORMAT_NHWC1C0,
  FORMAT_FSR_NCHW,
  FORMAT_FRACTAL_DECONV,
  FORMAT_C1HWNC0,
  FORMAT_FRACTAL_DECONV_TRANSPOSE,
  FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
  FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 =4
  FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 =4
  FORMAT_CHWN,
  FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
  FORMAT_HWCN,
  FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
  FORMAT_BN_WEIGHT,
  FORMAT_FILTER_HWCK,  // filter input tensor format
  FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
  FORMAT_HASHTABLE_LOOKUP_KEYS,
  FORMAT_HASHTABLE_LOOKUP_VALUE,
  FORMAT_HASHTABLE_LOOKUP_OUTPUT,
  FORMAT_HASHTABLE_LOOKUP_HITS = 24,
  FORMAT_C1HWNCoC0,
  FORMAT_MD,
  FORMAT_NDHWC,
  FORMAT_FRACTAL_ZZ,
  FORMAT_FRACTAL_NZ,
  FORMAT_NCDHW,
  FORMAT_DHWCN,  // 3D filter input tensor format
  FORMAT_NDC1HWC0,
  FORMAT_FRACTAL_Z_3D,
  FORMAT_CN,
  FORMAT_NC,
  FORMAT_DHWNC,
  FORMAT_FRACTAL_Z_3D_TRANSPOSE,  // 3D filter(transpose) input tensor format
  FORMAT_FRACTAL_ZN_LSTM,
  FORMAT_FRACTAL_Z_G,
  FORMAT_RESERVED,
  FORMAT_ALL,
  FORMAT_NULL
};

enum DeviceType { HOST, DEVICE };
}  // namespace aicpu
#endif  // CPU_KERNEL_TYPES_H
