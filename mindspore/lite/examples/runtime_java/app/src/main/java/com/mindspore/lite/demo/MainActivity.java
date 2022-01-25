/*
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

package com.mindspore.lite.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.mindspore.MSTensor;
import com.mindspore.Model;
import com.mindspore.config.CpuBindMode;
import com.mindspore.config.DataType;
import com.mindspore.config.MSContext;
import com.mindspore.config.ModelType;
import com.mindspore.config.Version;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Random;


public class MainActivity extends AppCompatActivity {
    private String TAG = "MS_LITE";
    private Model model1;
    private Model model2;
    private boolean model1Finish = true;
    private boolean model2Finish = true;
    private boolean model1Compile = false;
    private boolean model2Compile = false;

    public float[] generateArray(int len) {
        Random rand = new Random();
        float[] arr = new float[len];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = rand.nextFloat();
        }
        return arr;
    }

    private byte[] floatArrayToByteArray(float[] floats) {
        if (floats == null) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(floats.length * Float.BYTES);
        buffer.order(ByteOrder.nativeOrder());
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.put(floats);
        return buffer.array();
    }

    private MSContext createCPUConfig() {
        MSContext context = new MSContext();
        context.init(2, CpuBindMode.HIGHER_CPU, false);
        boolean ret = context.addDeviceInfo(com.mindspore.config.DeviceType.DT_CPU, false, 0);
        if (!ret) {
            Log.e(TAG, "Create CPU Config failed.");
            return null;
        }
        return context;
    }

    private MSContext createGPUConfig() {
        MSContext context = new MSContext();
        context.init(2, CpuBindMode.MID_CPU, false);
        boolean ret = context.addDeviceInfo(com.mindspore.config.DeviceType.DT_GPU, true, 0);
        if (!ret) {
            Log.e(TAG, "Create GPU Config failed.");
            return null;
        }
        return context;
    }

    private Model createLiteModel(String filePath, boolean isResize) {
        MSContext msContext = createCPUConfig();
        if (msContext == null) {
            Log.e(TAG, "Init context failed");
            return null;
        }

        // Create the MindSpore lite model.
        Model model = new Model();

        // Compile graph.
        boolean ret = model.build(filePath, ModelType.MT_MINDIR, msContext);
        if (!ret) {
            model.free();
            Log.e(TAG, "Compile graph failed");
            return null;
        }

        if (isResize) {
            List<MSTensor> inputs = model.getInputs();
            int[][] dims = {{1, 300, 300, 3}};
            ret = model.resize(inputs, dims);
            if (!ret) {
                Log.e(TAG, "Resize failed");
                model.free();
                return null;
            }
            StringBuilder msgSb = new StringBuilder();
            msgSb.append("in tensor shape: [");
            int[] shape = model.getInputs().get(0).getShape();
            for (int dim : shape) {
                msgSb.append(dim).append(",");
            }
            msgSb.append("]");
            Log.i(TAG, msgSb.toString());
        }

        return model;
    }

    private boolean printTensorData(MSTensor outTensor) {
        int[] shape = outTensor.getShape();
        StringBuilder msgSb = new StringBuilder();
        msgSb.append("out tensor shape: [");
        for (int dim : shape) {
            msgSb.append(dim).append(",");
        }
        msgSb.append("]");
        if (outTensor.getDataType() != DataType.kNumberTypeFloat32) {
            Log.e(TAG, "output tensor data type is not float, the data type is " + outTensor.getDataType());
            return false;
        }
        float[] result = outTensor.getFloatData();
        if (result == null) {
            Log.e(TAG, "decodeBytes return null");
            return false;
        }
        msgSb.append(" and out data:");
        for (int i = 0; i < 10 && i < outTensor.elementsNum(); i++) {
            msgSb.append(" ").append(result[i]);
        }
        Log.i(TAG, msgSb.toString());
        return true;
    }

    private boolean runInference(Model model) {
        Log.i(TAG, "runInference: ");
        MSTensor inputTensor = model.getInputByTensorName("graph_input-173");
        if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
            Log.e(TAG, "Input tensor data type is not float, the data type is " + inputTensor.getDataType());
            return false;
        }
        // Generator Random Data.
        int elementNums = inputTensor.elementsNum();
        float[] randomData = generateArray(elementNums);
        byte[] inputData = floatArrayToByteArray(randomData);

        // Set Input Data.
        inputTensor.setData(inputData);
        // Run Inference.
        boolean ret = model.predict();
        if (!ret) {
            Log.e(TAG, "MindSpore Lite run failed.");
            return false;
        }

        // Get Output Tensor Data.
        MSTensor outTensor = model.getOutputByTensorName("Softmax-65");
        // Print out Tensor Data.
        ret = printTensorData(outTensor);
        if (!ret) {
            return false;
        }

        outTensor = model.getOutputsByNodeName("Softmax-65").get(0);
        ret = printTensorData(outTensor);
        if (!ret) {
            return false;
        }
        List<MSTensor> outTensors = model.getOutputs();
        for (MSTensor output : outTensors) {
            Log.i(TAG, "Tensor name is:" + output.tensorName());
            ret = printTensorData(output);
            if (!ret) {
                return false;
            }
        }

        return true;
    }

    private void freeBuffer() {
        model1.free();
        model2.free();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String version = Version.version();
        Log.i(TAG, version);
        String modelPath = "mobilenetv2.ms";
        model1 = createLiteModel(modelPath, false);
        if (model1 != null) {
            model1Compile = true;
        } else {
            Toast.makeText(getApplicationContext(), "model1 Compile Failed.",
                    Toast.LENGTH_SHORT).show();
        }
        model2 = createLiteModel(modelPath, true);
        if (model2 != null) {
            model2Compile = true;
        } else {
            Toast.makeText(getApplicationContext(), "model2 Compile Failed.",
                    Toast.LENGTH_SHORT).show();
        }

        TextView btn_run = findViewById(R.id.btn_run);
        btn_run.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (model1Finish && model1Compile) {
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            model1Finish = false;
                            runInference(model1);
                            model1Finish = true;
                        }
                    }).start();
                } else {
                    Toast.makeText(getApplicationContext(), "MindSpore Lite is running...",
                            Toast.LENGTH_SHORT).show();
                }
            }
        });
        TextView btn_run_multi_thread = findViewById(R.id.btn_run_multi_thread);
        btn_run_multi_thread.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if (model1Finish && model1Compile) {
                            new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    model1Finish = false;
                                    runInference(model2);
                                    model1Finish = true;
                                }
                            }).start();
                        }
                        if (model2Finish && model2Compile) {
                            new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    model2Finish = false;
                                    runInference(model2);
                                    model2Finish = true;
                                }
                            }).start();
                        }
                        if (!model2Finish && !model2Finish) {
                            Toast.makeText(getApplicationContext(), "MindSpore Lite is running...",
                                    Toast.LENGTH_SHORT).show();
                        }
                    }
                }
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        freeBuffer();
    }
}
