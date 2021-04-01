#include "include/errorcode.h"
#include "include/lite_session.h"
#include "include/ms_tensor.h"
#include "mnist_input_data.h"
using namespace mindspore;
int main(void) {
  while (1) {
    /* USER CODE END WHILE */
    SEGGER_RTT_printf(0, "***********mnist test start***********\n");
    float a = 3.1415926;
    SEGGER_RTT_printf(0, "output: [%d] \n", (int)(a * 10000));
    const char *model_buffer = nullptr;
    int model_size = 0;
    session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, nullptr);
    Vector<tensor::MSTensor *> inputs = session->GetInputs();
    size_t inputs_num = inputs.size();
    void *inputs_binbuf[inputs_num];
    int inputs_size[inputs_num];
    for (size_t i = 0; i < inputs_num; ++i) {
      inputs_size[i] = inputs[i]->Size();
    }
    // here mnist only have one input data,just hard code to it's array;
    inputs_binbuf[0] = mnist_inputs_data;
    for (size_t i = 0; i < inputs_num; ++i) {
      void *input_data = inputs[i]->MutableData();
      memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
    }
    int ret = session->RunGraph();
    if (ret != lite::RET_OK) {
      return lite::RET_ERROR;
    }
    Vector<String> outputs_name = session->GetOutputTensorNames();
    for (int i = 0; i < outputs_name.size(); ++i) {
      tensor::MSTensor *output_tensor = session->GetOutputByTensorName(outputs_name[i]);
      if (output_tensor == nullptr) {
        return -1;
      }
      float *casted_data = static_cast<float *>(output_tensor->MutableData());
      if (casted_data == nullptr) {
        return -1;
      }
      for (size_t j = 0; j < 10 && j < output_tensor->ElementsNum(); j++) {
        SEGGER_RTT_printf(0, "output[%d]: [%d]\n", j, (int)(casted_data[j] * 10000));
      }
    }
    delete session;
    SEGGER_RTT_printf(0, "***********mnist test end***********\n");
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}
