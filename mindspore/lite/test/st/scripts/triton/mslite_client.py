import sys
import argparse
import numpy as np
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        default='',
                        help='Inference model name.')
    parser.add_argument('--input_name',
                        type=str,
                        required=True,
                        default='',
                        help='The input of model.')
    parser.add_argument('--input_shape',
                        type=str,
                        required=True,
                        default='',
                        help='The input shape of model.')
    parser.add_argument('--input_data',
                        type=str,
                        required=True,
                        default='',
                        help='The input data of model.')
    parser.add_argument('--output_name',
                        type=str,
                        required=True,
                        default='',
                        help='The output of model.')
    parser.add_argument('--output_data',
                        type=str,
                        required=True,
                        default='',
                        help='The output data of model.')

    flags = parser.parse_args()
    shape = list(map(int, flags.input_shape.split(',')))
    with httpclient.InferenceServerClient(url=flags.url) as client:
        try:
            input_data = np.fromfile(flags.input_data, np.float32).reshape(shape)
            inputs = [httpclient.InferInput(flags.input_name, shape, np_to_triton_dtype(np.float32))]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [httpclient.InferRequestedOutput(flags.output_name)]

            response = client.infer(flags.model, inputs, request_id=str(0), outputs=outputs)
            result = response.get_response()
            output_data = response.as_numpy(flags.output_name)

            # check the output.
            clib_data = np.fromfile(flags.output_data, np.float32).reshape(output_data.shape)
            if not np.allclose(clib_data, output_data):
                print(f"The output data is not correct.")
                sys.exit(1)
        except Exception as err:  # pylint: disable=broad-except
            print(f"Check output data failed with error: {err}")
            sys.exit(1)

    sys.exit(0)
