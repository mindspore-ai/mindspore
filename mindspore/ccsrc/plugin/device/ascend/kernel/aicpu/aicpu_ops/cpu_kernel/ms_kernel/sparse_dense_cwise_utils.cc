#include "sparse_dense_cwise_utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>
#include <vector>

#include "broadcast_iterator.h"
#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "kernel_log.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace aicpu {
namespace {
const uint32_t kInputNum_SparseDenseCwiseOp = 4;
const uint32_t kOutputNum_SparseDenseCwiseOp = 1;
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
}  // namespace

template <typename Op>
uint32_t SparseDenseCwiseOpKernel<Op>::CheckParams(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum_SparseDenseCwiseOp, kOutputNum_SparseDenseCwiseOp),
                      "SparseDenseCwise%s normal check failed.", Op::Name().c_str());

  Tensor *x1_indices = ctx.Input(0);
  Tensor *x1_values = ctx.Input(1);
  Tensor *x1_shape = ctx.Input(2);
  Tensor *x2 = ctx.Input(3);
  Tensor *y = ctx.Output(0);

  DataType x1_indices_type = x1_indices->GetDataType();
  DataType x1_values_type = x1_values->GetDataType();
  DataType x1_shape_type = x1_shape->GetDataType();
  DataType x2_type = x2->GetDataType();
  DataType y_type = y->GetDataType();
  KERNEL_CHECK_FALSE((x1_indices_type == x1_shape_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of x1_indices_type [%s] need be same with "
                     "x1_shape [%s].",
                     DTypeStr(x1_indices_type).c_str(), DTypeStr(x1_shape_type).c_str())
  KERNEL_CHECK_FALSE(((x1_values_type == x2_type) && (x1_values_type == y_type)), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of x1_values_type [%s] need be same with "
                     "x2_type[%s] and y_type [%s].",
                     DTypeStr(x1_values_type).c_str(), DTypeStr(x2_type).c_str(), DTypeStr(y_type).c_str())
  KERNEL_CHECK_FALSE((x1_indices_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of x1_indices_type [%s] need be int_64.", DTypeStr(x1_indices_type).c_str())
  int32_t input0_dims = x1_indices->GetTensorShape()->GetDims();
  int32_t input1_dims = x1_values->GetTensorShape()->GetDims();
  int32_t input2_dims = x1_shape->GetTensorShape()->GetDims();
  int32_t input3_dims = x2->GetTensorShape()->GetDims();
  int32_t output_dims = y->GetTensorShape()->GetDims();
  int64_t shape_elements_nums = x1_shape->GetTensorShape()->NumElements();
  int64_t indices_0 = x1_indices->GetTensorShape()->GetDimSize(0);
  int64_t value_0 = x1_values->GetTensorShape()->GetDimSize(0);
  KERNEL_CHECK_FALSE((int(input0_dims) == 2), KERNEL_STATUS_PARAM_INVALID, "The dims of input0 need be 2.")
  KERNEL_CHECK_FALSE((input1_dims == 1), KERNEL_STATUS_PARAM_INVALID, "The dims of input1 need be 1 .")
  KERNEL_CHECK_FALSE((input2_dims == 1), KERNEL_STATUS_PARAM_INVALID, "The dims of input2 need be 1.")
  KERNEL_CHECK_FALSE((output_dims == 1), KERNEL_STATUS_PARAM_INVALID, "The dims of output need be 1.")
  KERNEL_CHECK_FALSE((input3_dims <= shape_elements_nums), KERNEL_STATUS_PARAM_INVALID,
                     "The dims of DenseTensor  is large than sparseTensor.")
  KERNEL_CHECK_FALSE((indices_0 == value_0), KERNEL_STATUS_PARAM_INVALID, "The num of indices  is not equal to value.")

  int64_t indices_num = x1_indices->GetTensorShape()->GetDimSize(0);
  int64_t dims = x1_indices->GetTensorShape()->GetDimSize(1);
  auto x1_indices_data = reinterpret_cast<int64_t *>(x1_indices->GetData());
  auto x1_shape_data = reinterpret_cast<int64_t *>(x1_shape->GetData());
  for (int64_t i = 0; i < indices_num; ++i) {
    for (int64_t j = 0; j < dims; ++j) {
      KERNEL_CHECK_FALSE((x1_indices_data[i * dims + j] >= 0 && x1_indices_data[i * dims + j] < x1_shape_data[j]),
                         KERNEL_STATUS_PARAM_INVALID, "For SparseDenseCwise%s, indices go out of bounds.",
                         Op::Name().c_str());
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpSpecialCompute(BcastShapeType type, CpuKernelContext &ctx) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  auto sparse_values_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  auto dense_data = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t value_nums = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dimension = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  int64_t data_num = ctx.Input(1)->NumElements();

  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
  }
  int64_t dims = ctx.Input(2)->NumElements();
  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }
  if (Sparse_numelements >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
    }
    if (max_core_num > value_nums) {
      max_core_num = value_nums;
    }

    auto sharder_Op = [&](int64_t start, int64_t end) {
      switch (type) {
        case BcastShapeType::SAME_SHAPE:
          for (int64_t i = start; i < end; i++) {
            int index = 0;
            for (int64_t j = 0; j < dimension - 1; j++) {
              int c = 1;
              for (int k = j + 1; k < dimension; k++) {
                c = c * sparse_shape_data[k];
              }
              index += c * sparse_indices_data[j + i * dimension];
            }
            index += sparse_indices_data[(i + 1) * dimension - 1];
            std::string name = Op::Name().c_str();
            if (name == "Add") {
              output_data[i] = sparse_values_vec[i] + dense_data[index];
            } else if (name == "Div") {
              if (fabs(double(dense_data[index])) < 1e-6) {
                KERNEL_LOG_ERROR("Cannot be divided by 0");
                return KERNEL_STATUS_PARAM_INVALID;
              } else {
                output_data[i] = sparse_values_vec[i] / dense_data[index];
              }
            } else {
              output_data[i] = sparse_values_vec[i] * dense_data[index];
            }
          }
          break;

        case BcastShapeType::Y_ONE_ELEMENT:
          for (int64_t i = start; i < end; i++) {
            std::string name = Op::Name().c_str();
            if (name == "Add") {
              output_data[i] = sparse_values_data[i] + *(dense_data);
            } else if (name == "Div") {
              if (fabs(double(*(dense_data))) < 1e-6) {
                KERNEL_LOG_ERROR("Cannot be divided by 0");
                return KERNEL_STATUS_PARAM_INVALID;
              } else {
                output_data[i] = sparse_values_data[i] / *(dense_data);
              }
            } else {
              output_data[i] = *(dense_data)*sparse_values_data[i];
            }
          }
          break;
        default:
          KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
          break;
      }
      return KERNEL_STATUS_OK;
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, value_nums, value_nums / max_core_num, sharder_Op),
                        "Op Compute failed.");
  } else {
    switch (type) {
      case BcastShapeType::SAME_SHAPE:
        for (int64_t i = 0; i < value_nums; i++) {
          int index = 0;
          for (int64_t j = 0; j < dimension - 1; j++) {
            int c = 1;
            for (int k = j + 1; k < dimension; k++) {
              c = c * sparse_shape_data[k];
            }
            index += c * sparse_indices_data[j + i * dimension];
          }
          index += sparse_indices_data[(i + 1) * dimension - 1];
          std::string name = Op::Name().c_str();
          if (name == "Add") {
            output_data[i] = sparse_values_vec[i] + dense_data[index];
          } else if (name == "Div") {
            if (fabs(double(dense_data[index])) < 1e-6) {
              KERNEL_LOG_ERROR("Cannot be divided by 0");
              return KERNEL_STATUS_PARAM_INVALID;
            } else {
              output_data[i] = sparse_values_vec[i] / dense_data[index];
            }
          } else {
            output_data[i] = sparse_values_vec[i] * dense_data[index];
          }
        }
        break;

      case BcastShapeType::Y_ONE_ELEMENT:
        for (int64_t i = 0; i < value_nums; i++) {
          std::string name = Op::Name().c_str();
          if (name == "Add") {
            output_data[i] = sparse_values_data[i] + *(dense_data);
          } else if (name == "Div") {
            if (fabs(double(*(dense_data))) < 1e-6) {
              KERNEL_LOG_ERROR("Cannot be divided by 0");
              return KERNEL_STATUS_PARAM_INVALID;
            } else {
              output_data[i] = sparse_values_data[i] / *(dense_data);
            }
          } else {
            output_data[i] = *(dense_data)*sparse_values_data[i];
          }
        }
        break;
      default:
        KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
        break;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpNoBcastCompute(CpuKernelContext &ctx) {
  auto *input2_tensor = ctx.Input(2);
  auto *input3_tensor = ctx.Input(3);
  int64_t dimension = input2_tensor->NumElements();
  int32_t dense_dims = input3_tensor->GetTensorShape()->GetDims();
  BcastShapeType type = dimension == dense_dims ? BcastShapeType::SAME_SHAPE : BcastShapeType::Y_ONE_ELEMENT;
  SparseDenseCwiseOpSpecialCompute<T>(type, ctx);
  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpBcastCompute(CpuKernelContext &ctx) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  auto sparse_values_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  auto dense_data = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t value_nums = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dimension = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  auto dense_shape = ctx.Input(3)->GetTensorShape()->GetDimSizes();
  int64_t dims = ctx.Input(2)->NumElements();
  int64_t data_num = ctx.Input(1)->NumElements();

  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }

  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }
  std::vector<int64_t> sparse_shape1(dimension);
  for (int64_t j = 0; j < dimension; j++) {
    sparse_shape1[j] = sparse_shape[j];
  }

  BroadcastIterator broad_base_iter_1(sparse_shape, dense_shape, sparse_shape1);
  std::vector<T> Dense(Sparse_numelements);
  broad_base_iter_1.SetPos(0);
  for (int64_t i = 0; i < Sparse_numelements; i++) {
    Dense[i] = dense_data[broad_base_iter_1.GetInputPosB()];
    broad_base_iter_1.GenNextPos();
  }
  if (Sparse_numelements >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
    }
    if (max_core_num > value_nums) {
      max_core_num = value_nums;
    }
    auto sharder_Op = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        int index = 0;
        for (int64_t j = 0; j < dimension - 1; j++) {
          int c = 1;
          for (int k = j + 1; k < dimension; k++) {
            c = c * sparse_shape_data[k];
          }
          index += sparse_indices_data[j + i * dimension] * c;
        }
        index += sparse_indices_data[(i + 1) * dimension - 1];
        std::string name = Op::Name().c_str();
        if (name == "Add") {
          output_data[i] = sparse_values_vec[i] + Dense[index];
        } else if (name == "Div") {
          if (fabs(double(Dense[index])) < 1e-6) {
            KERNEL_LOG_ERROR("Cannot be divided by 0");
            return KERNEL_STATUS_PARAM_INVALID;
          } else {
            output_data[i] = sparse_values_vec[i] / Dense[index];
          }
        } else {
          output_data[i] = sparse_values_vec[i] * Dense[index];
        }
      }
      return KERNEL_STATUS_OK;
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, value_nums, value_nums / max_core_num, sharder_Op),
                        "Op Compute failed.");
  } else {
    for (int64_t i = 0; i < value_nums; i++) {
      int index = 0;
      for (int64_t j = 0; j < dimension - 1; j++) {
        int c = 1;
        for (int k = j + 1; k < dimension; k++) {
          c = c * sparse_shape_data[k];
        }
        index += sparse_indices_data[j + i * dimension] * c;
      }
      index += sparse_indices_data[(i + 1) * dimension - 1];
      std::string name = Op::Name().c_str();
      if (name == "Add") {
        output_data[i] = sparse_values_vec[i] + Dense[index];
      } else if (name == "Div") {
        if (fabs(double(Dense[index])) < 1e-6) {
          KERNEL_LOG_ERROR("Cannot be divided by 0");
          return KERNEL_STATUS_PARAM_INVALID;
        } else {
          output_data[i] = sparse_values_vec[i] / Dense[index];
        }
      } else {
        output_data[i] = sparse_values_vec[i] * Dense[index];
      }
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpCompute(CpuKernelContext &ctx) {
  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    case DT_INT8:
      return ComputeOp<int8_t>(ctx);
    case DT_INT16:
      return ComputeOp<int16_t>(ctx);
    case DT_INT32:
      return ComputeOp<int32_t>(ctx);
    case DT_INT64:
      return ComputeOp<int64_t>(ctx);
    case DT_UINT8:
      return ComputeOp<uint8_t>(ctx);
    case DT_UINT16:
      return ComputeOp<uint16_t>(ctx);
    case DT_UINT32:
      return ComputeOp<uint32_t>(ctx);
    case DT_UINT64:
      return ComputeOp<uint64_t>(ctx);
    case DT_FLOAT16:
      return ComputeOp<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeOp<float>(ctx);
    case DT_DOUBLE:
      return ComputeOp<double>(ctx);
    case DT_COMPLEX64:
      return ComputeOpComplex<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return ComputeOpComplex<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("sparse_dense_cwise kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::ComputeOp(CpuKernelContext &ctx) {
  auto *input3_tensor = ctx.Input(3);
  auto dimension = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  int32_t dense_dims = input3_tensor->GetTensorShape()->GetDims();
  auto dense_shape = input3_tensor->GetTensorShape()->GetDimSizes();
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  int64_t dense_num = ctx.Input(3)->GetTensorShape()->NumElements();

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }

  bool isNeedBcast = (dense_shape == sparse_shape || dense_num == 1);
  if (isNeedBcast) {
    return SparseDenseCwiseOpNoBcastCompute<T>(ctx);
  } else {
    if (dense_dims <= dimension) {
      for (int i = dense_dims - 1; i >= 0; --i) {
        if ((dense_shape[i] != 1) && (dense_shape[i] != sparse_shape[i + dimension - dense_dims])) {
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      return SparseDenseCwiseOpBcastCompute<T>(ctx);
    } else {
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::ComputeOpComplex(CpuKernelContext &ctx) {
  auto *input2_tensor = ctx.Input(2);
  auto *input3_tensor = ctx.Input(3);
  int64_t dense_num = ctx.Input(3)->GetTensorShape()->NumElements();
  int64_t dimension = input2_tensor->NumElements();
  int32_t dense_dims = input3_tensor->GetTensorShape()->GetDims();
  auto dense_shape = input3_tensor->GetTensorShape()->GetDimSizes();
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }

  bool isNeedBcast = (dense_shape == sparse_shape || dense_num == 1);
  if (isNeedBcast) {
    return SparseDenseCwiseOpNoBcastComputeComplex<T>(ctx);
  } else {
    if (dense_dims <= dimension) {
      for (int i = dense_dims - 1; i >= 0; --i) {
        if ((dense_shape[i] != 1) && (dense_shape[i] != sparse_shape[i + dimension - dense_dims])) {
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      return SparseDenseCwiseOpBcastComputeComplex<T>(ctx);
    } else {
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpSpecialComputeComplex(BcastShapeType type,
                                                                               CpuKernelContext &ctx) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  auto sparse_values_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  auto dense_data = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t value_nums = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dimension = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  int64_t data_num = ctx.Input(1)->NumElements();
  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
  }
  int64_t dims = ctx.Input(2)->NumElements();
  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }
  if (Sparse_numelements >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
    }
    if (max_core_num > value_nums) {
      max_core_num = value_nums;
    }

    auto sharder_Op = [&](int64_t start, int64_t end) {
      switch (type) {
        case BcastShapeType::SAME_SHAPE:
          for (int64_t i = start; i < end; i++) {
            int index = 0;
            for (int64_t j = 0; j < dimension - 1; j++) {
              int c = 1;
              for (int k = j + 1; k < dimension; k++) {
                c = c * sparse_shape_data[k];
              }
              index += c * sparse_indices_data[j + i * dimension];
            }
            index += sparse_indices_data[(i + 1) * dimension - 1];
            std::string name = Op::Name().c_str();
            if (name == "Add") {
              output_data[i] = sparse_values_vec[i] + dense_data[index];
            } else if (name == "Div") {
              if (fabs(dense_data[index]) < 1e-6) {
                KERNEL_LOG_ERROR("Cannot be divided by 0");
                return KERNEL_STATUS_PARAM_INVALID;
              } else {
                output_data[i] = sparse_values_vec[i] / dense_data[index];
              }
            } else {
              output_data[i] = sparse_values_vec[i] * dense_data[index];
            }
          }
          break;

        case BcastShapeType::Y_ONE_ELEMENT:
          for (int64_t i = start; i < end; i++) {
            std::string name = Op::Name().c_str();
            if (name == "Add") {
              output_data[i] = sparse_values_data[i] + *(dense_data);
            } else if (name == "Div") {
              if (fabs(*(dense_data)) < 1e-6) {
                KERNEL_LOG_ERROR("Cannot be divided by 0");
                return KERNEL_STATUS_PARAM_INVALID;
              } else {
                output_data[i] = sparse_values_data[i] / *(dense_data);
              }
            } else {
              output_data[i] = *(dense_data)*sparse_values_data[i];
            }
          }
          break;
        default:
          KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
          break;
      }
      return KERNEL_STATUS_OK;
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, value_nums, value_nums / max_core_num, sharder_Op),
                        "Op Compute failed.");
  } else {
    switch (type) {
      case BcastShapeType::SAME_SHAPE:
        for (int64_t i = 0; i < value_nums; i++) {
          int index = 0;
          for (int64_t j = 0; j < dimension - 1; j++) {
            int c = 1;
            for (int k = j + 1; k < dimension; k++) {
              c = c * sparse_shape_data[k];
            }
            index += c * sparse_indices_data[j + i * dimension];
          }
          index += sparse_indices_data[(i + 1) * dimension - 1];
          std::string name = Op::Name().c_str();
          if (name == "Add") {
            output_data[i] = sparse_values_vec[i] + dense_data[index];
          } else if (name == "Div") {
            if (fabs(dense_data[index]) < 1e-6) {
              KERNEL_LOG_ERROR("Cannot be divided by 0");
              return KERNEL_STATUS_PARAM_INVALID;
            } else {
              output_data[i] = sparse_values_vec[i] / dense_data[index];
            }
          } else {
            output_data[i] = sparse_values_vec[i] * dense_data[index];
          }
        }
        break;

      case BcastShapeType::Y_ONE_ELEMENT:
        for (int64_t i = 0; i < value_nums; i++) {
          std::string name = Op::Name().c_str();
          if (name == "Add") {
            output_data[i] = sparse_values_data[i] + *(dense_data);
          } else if (name == "Div") {
            if (fabs(*(dense_data)) < 1e-6) {
              KERNEL_LOG_ERROR("Cannot be divided by 0");
              return KERNEL_STATUS_PARAM_INVALID;
            } else {
              output_data[i] = sparse_values_data[i] / *(dense_data);
            }
          } else {
            output_data[i] = *(dense_data)*sparse_values_data[i];
          }
        }
        break;
      default:
        KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
        break;
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpNoBcastComputeComplex(CpuKernelContext &ctx) {
  auto *input2_tensor = ctx.Input(2);
  auto *input3_tensor = ctx.Input(3);
  int64_t dimension = input2_tensor->NumElements();
  int32_t dense_dims = input3_tensor->GetTensorShape()->GetDims();

  BcastShapeType type = dimension == dense_dims ? BcastShapeType::SAME_SHAPE : BcastShapeType::Y_ONE_ELEMENT;
  SparseDenseCwiseOpSpecialComputeComplex<T>(type, ctx);
  return KERNEL_STATUS_OK;
}

template <typename Op>
template <typename T>
uint32_t SparseDenseCwiseOpKernel<Op>::SparseDenseCwiseOpBcastComputeComplex(CpuKernelContext &ctx) {
  auto sparse_indices_data = reinterpret_cast<int64_t *>(ctx.Input(0)->GetData());
  auto sparse_values_data = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto sparse_shape_data = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  auto dense_data = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t value_nums = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  int64_t dimension = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  auto dense_shape = ctx.Input(3)->GetTensorShape()->GetDimSizes();
  int64_t dims = ctx.Input(2)->NumElements();
  int64_t data_num = ctx.Input(1)->NumElements();

  int64_t Sparse_numelements = 1;
  for (int64_t i = 0; i < dims; i++) {
    Sparse_numelements *= sparse_shape_data[i];
  }

  std::vector<T> sparse_values_vec(data_num);
  for (int64_t i = 0; i < data_num; i++) {
    sparse_values_vec[i] = (sparse_values_data[i]);
  }

  std::vector<int64_t> sparse_shape(dimension);
  for (int64_t i = 0; i < dimension; i++) {
    sparse_shape[i] = sparse_shape_data[i];
  }
  std::vector<int64_t> sparse_shape1(dimension);
  for (int64_t j = 0; j < dimension; j++) {
    sparse_shape1[j] = sparse_shape[j];
  }

  BroadcastIterator broad_base_iter_1(sparse_shape, dense_shape, sparse_shape1);
  std::vector<T> Dense(Sparse_numelements);
  broad_base_iter_1.SetPos(0);
  for (int64_t i = 0; i < Sparse_numelements; i++) {
    Dense[i] = dense_data[broad_base_iter_1.GetInputPosB()];
    broad_base_iter_1.GenNextPos();
  }

  if (Sparse_numelements >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0");
    }
    if (max_core_num > value_nums) {
      max_core_num = value_nums;
    }

    auto sharder_Op = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        int index = 0;
        for (int64_t j = 0; j < dimension - 1; j++) {
          int c = 1;
          for (int k = j + 1; k < dimension; k++) {
            c = c * sparse_shape_data[k];
          }
          index += sparse_indices_data[j + i * dimension] * c;
        }
        index += sparse_indices_data[(i + 1) * dimension - 1];
        std::string name = Op::Name().c_str();
        if (name == "Add") {
          output_data[i] = sparse_values_vec[i] + Dense[index];
        } else if (name == "Div") {
          if (fabs(Dense[index]) < 1e-6) {
            KERNEL_LOG_ERROR("Cannot be divided by 0");
            return KERNEL_STATUS_PARAM_INVALID;
          } else {
            output_data[i] = sparse_values_vec[i] / Dense[index];
          }
        } else {
          output_data[i] = sparse_values_vec[i] * Dense[index];
        }
      }
      return KERNEL_STATUS_OK;
    };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, value_nums, value_nums / max_core_num, sharder_Op),
                        "Op Compute failed.");
  } else {
    for (int64_t i = 0; i < value_nums; i++) {
      int index = 0;
      for (int64_t j = 0; j < dimension - 1; j++) {
        int c = 1;
        for (int k = j + 1; k < dimension; k++) {
          c = c * sparse_shape_data[k];
        }
        index += sparse_indices_data[j + i * dimension] * c;
      }
      index += sparse_indices_data[(i + 1) * dimension - 1];
      std::string name = Op::Name().c_str();
      if (name == "Add") {
        output_data[i] = sparse_values_vec[i] + Dense[index];
      } else if (name == "Div") {
        if (fabs(Dense[index]) < 1e-6) {
          KERNEL_LOG_ERROR("Cannot be divided by 0");
          return KERNEL_STATUS_PARAM_INVALID;
        } else {
          output_data[i] = sparse_values_vec[i] / Dense[index];
        }
      } else {
        output_data[i] = sparse_values_vec[i] * Dense[index];
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template class SparseDenseCwiseOpKernel<AddOp>;
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<int8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<int16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<int32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<int64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<uint8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<uint16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<uint32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<uint64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<Eigen::half>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<float>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<double>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<std::complex<float>>(
  CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<AddOp>::SparseDenseCwiseOpCompute<std::complex<double>>(
  CpuKernelContext &ctx);

template class SparseDenseCwiseOpKernel<DivOp>;
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<int8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<int16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<int32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<int64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<uint8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<uint16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<uint32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<uint64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<Eigen::half>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<float>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<double>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<std::complex<float>>(
  CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<DivOp>::SparseDenseCwiseOpCompute<std::complex<double>>(
  CpuKernelContext &ctx);

template class SparseDenseCwiseOpKernel<MulOp>;
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<int8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<int16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<int32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<int64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<uint8_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<uint16_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<uint32_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<uint64_t>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<Eigen::half>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<float>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<double>(CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<std::complex<float>>(
  CpuKernelContext &ctx);
template uint32_t SparseDenseCwiseOpKernel<MulOp>::SparseDenseCwiseOpCompute<std::complex<double>>(
  CpuKernelContext &ctx);
}  // namespace aicpu
