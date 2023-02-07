#include "sparse_slice_grad.h"
#include <complex>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const char *kSparseSliceGrad = "SparseSliceGrad";
}  // namespace
namespace aicpu {
uint32_t SparseSliceGradCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *backprop_val_grad = ctx.Input(0);
  Tensor *indices = ctx.Input(1);
  Tensor *start = ctx.Input(2);
  Tensor *new_indices = ctx.Input(3);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "sparseslicegrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(SparseSliceGradParamCheck(backprop_val_grad, indices, start, new_indices),
                      "sparseslicegrad check params failed.");
  DataType input0_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[a] data type is [%s].", kSparseSliceGrad, DTypeStr(input0_type).c_str());
  switch (input0_type) {
    case DT_INT8:
      GradCompute<int8_t>(ctx);
      break;
    case DT_UINT8:
      GradCompute<uint8_t>(ctx);
      break;
    case DT_INT16:
      GradCompute<int16_t>(ctx);
      break;
    case DT_UINT16:
      GradCompute<uint16_t>(ctx);
      break;
    case DT_INT32:
      GradCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      GradCompute<int64_t>(ctx);
      break;
    case DT_FLOAT:
      GradCompute<float>(ctx);
      break;
    case DT_FLOAT16:
      GradCompute<Eigen::half>(ctx);
      break;
    case DT_DOUBLE:
      GradCompute<double>(ctx);
      break;
    case DT_COMPLEX64:
      GradCompute<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      GradCompute<std::complex<double>>(ctx);
      break;

    default:
      KERNEL_LOG_ERROR("SparseSliceGrad kernel data type [%s] not support.", DTypeStr(input0_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SparseSliceGradCpuKernel::GradCompute(CpuKernelContext &ctx) {
  Tensor *backprop_val_grad = ctx.Input(0);
  Tensor *indices = ctx.Input(1);
  Tensor *start = ctx.Input(2);
  Tensor *new_indices = ctx.Input(3);
  auto indices_shape = indices->GetTensorShape();
  const int64_t input_nnz = indices_shape->GetDimSize(0);
  Tensor *y_grad = ctx.Output(0);
  auto *y_grad_vec = static_cast<T *>(y_grad->GetData());
  memset(y_grad_vec, 0, sizeof(T) * input_nnz);

  std::vector<T> backprop_val_grad_flat;
  auto *backprop_val_grad_vec = static_cast<T *>(backprop_val_grad->GetData());
  const auto indices_mat = (EigenTensor(indices, indices->GetData())).matrix<int64_t>();
  const auto new_indices_mat = (EigenTensor(new_indices, new_indices->GetData())).matrix<int64_t>();
  EigenTensor start_ET(start, start->GetData());
  const auto start_flat = start_ET.flat<int64_t>();

  int64_t j = 0;
  const int num_dims = indices_shape->GetDimSize(1);
  for (int64_t i = 0; i < input_nnz && j < backprop_val_grad->NumElements(); ++i) {
    bool is_same = true;
    for (int d = 0; d < num_dims; ++d) {
      const int64_t indices_value = indices_mat(i, d);
      const int64_t new_indices_value = new_indices_mat(j, d);
      const int64_t offset = start_flat(d);
      if (indices_value != new_indices_value + offset) {
        is_same = false;
        break;
      }
    }
    if (is_same) {
      y_grad_vec[i] = *(backprop_val_grad_vec + j);
      ++j;
    }
  }
  KERNEL_CHECK_FALSE((backprop_val_grad->NumElements() == j), KERNEL_STATUS_PARAM_INVALID,
                     "Elements of backprop_val_grad aren't all propagated."
                     "Num elements:",
                     backprop_val_grad->NumElements(), ", used: ", j);
  return KERNEL_STATUS_OK;
}

uint32_t SparseSliceGradCpuKernel::SparseSliceGradParamCheck(Tensor *backprop_val_grad, Tensor *indices, Tensor *start,
                                                             Tensor *new_indices) {
  KERNEL_CHECK_FALSE((IsVector(backprop_val_grad->GetTensorShape()->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input backprop_val_grad should be a vector but received shape: [%d].",
                     backprop_val_grad->GetTensorShape()->GetDimSizes());
  KERNEL_CHECK_FALSE(
    (IsMatrix(indices->GetTensorShape()->GetDimSizes()) && IsMatrix(new_indices->GetTensorShape()->GetDimSizes())),
    KERNEL_STATUS_PARAM_INVALID,
    "Input and output indices should be matrices [%lld], but "
    "received shapes: [%lld].",
    indices->GetTensorShape()->GetDimSizes(), new_indices->GetTensorShape()->GetDimSizes());
  auto indices_shape = indices->GetTensorShape();
  auto new_indices_shape = new_indices->GetTensorShape();
  KERNEL_CHECK_FALSE((indices_shape->GetDimSize(1) == new_indices_shape->GetDimSize(1)), KERNEL_STATUS_PARAM_INVALID,
                     "The input and output should have the same, ndims: got: [%d] and [%d].",
                     indices_shape->GetDimSize(1), new_indices_shape->GetDimSize(1));
  KERNEL_CHECK_FALSE((new_indices_shape->GetDimSize(0) <= indices_shape->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "# rows of output_indices should be not greater than of input_indices, "
                     "got: [%d] and [%d].",
                     new_indices_shape->GetDimSize(0), indices_shape->GetDimSize(0));
  KERNEL_CHECK_FALSE((backprop_val_grad->NumElements() == new_indices_shape->GetDimSize(0)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "# elements of backprop_val_grad and rows of new_indices should match "
                     "(#nnz of sum): got [%d] and [%d].",
                     backprop_val_grad->NumElements(), new_indices_shape->GetDimSize(0));
  KERNEL_CHECK_FALSE((IsVector(start->GetTensorShape()->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "The start should be a vector but received shape [%s].",
                     VectorToString(start->GetTensorShape()->GetDimSizes()).c_str());
  const int num_dims = indices_shape->GetDimSize(1);
  KERNEL_CHECK_FALSE((num_dims == start->NumElements()), KERNEL_STATUS_PARAM_INVALID,
                     "Expected start must be a vector of length [%d] but got length [%d].", num_dims,
                     start->NumElements());
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSparseSliceGrad, SparseSliceGradCpuKernel);
}  // namespace aicpu
