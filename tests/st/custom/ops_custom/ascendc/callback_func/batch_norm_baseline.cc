#include <vector>
#include "acl/acl_base.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

extern "C" int aclnnBatchNormGetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                                              uint64_t *workspace_size, aclOpExecutor **executor) {
  using FuncType = int (*)(aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, bool, float, float,
                           aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
  aclTensor *input0 = static_cast<aclTensor *>(inputs[0]);
  aclTensor *input1 = static_cast<aclTensor *>(inputs[1]);
  aclTensor *input2 = static_cast<aclTensor *>(inputs[2]);
  aclTensor *input3 = static_cast<aclTensor *>(inputs[3]);
  aclTensor *input4 = static_cast<aclTensor *>(inputs[4]);
  bool *input5 = static_cast<bool *>(inputs[5]);
  float *input6 = static_cast<float *>(inputs[6]);
  float *input7 = static_cast<float *>(inputs[7]);
  aclTensor *output0 = static_cast<aclTensor *>(outputs[0]);
  aclTensor *output1 = static_cast<aclTensor *>(outputs[1]);
  aclTensor *output2 = static_cast<aclTensor *>(outputs[2]);
  return func(input0, input1, input2, input3, input4, *input5, *input6, *input7, output0, output1, output2,
              workspace_size, executor);
}
