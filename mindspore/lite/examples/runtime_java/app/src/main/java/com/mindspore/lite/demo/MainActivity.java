package com.mindspore.lite.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.mindspore.lite.DataType;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.CpuBindMode;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;
import com.mindspore.lite.Version;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class MainActivity extends AppCompatActivity {
    private String TAG = "MS_LITE";
    private Model model;
    private LiteSession session1;
    private LiteSession session2;
    private boolean session1Finish = true;
    private boolean session2Finish = true;
    private boolean session1Compile = false;
    private boolean session2Compile = false;

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

    private MSConfig createCPUConfig() {
        MSConfig msConfig = new MSConfig();
        boolean ret = msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.HIGHER_CPU, true);
        if (!ret) {
            Log.e(TAG, "Create CPU Config failed.");
            return null;
        }
        return msConfig;
    }

    private MSConfig createGPUConfig() {
        MSConfig msConfig = new MSConfig();
        boolean ret = msConfig.init(DeviceType.DT_GPU, 2, CpuBindMode.MID_CPU, true);
        if (!ret) {
            return null;
        }
        return msConfig;
    }

    private LiteSession createLiteSession(boolean isResize) {
        MSConfig msConfig = createCPUConfig();
        if (msConfig == null) {
            Log.e(TAG, "Init context failed");
            return null;
        }

        // Create the MindSpore lite session.
        LiteSession session = new LiteSession();
        boolean ret = session.init(msConfig);
        msConfig.free();
        if (!ret) {
            Log.e(TAG, "Create session failed");
            return null;
        }

        // Compile graph.
        ret = session.compileGraph(model);
        if (!ret) {
            session.free();
            Log.e(TAG, "Compile graph failed");
            return null;
        }

        if (isResize) {
            List<MSTensor> inputs = session.getInputs();
            int[][] dims = {{1, 300, 300, 3}};
            ret = session.resize(inputs, dims);
            if (!ret) {
                Log.e(TAG, "Resize failed");
                session.free();
                return null;
            }
            StringBuilder msgSb = new StringBuilder();
            msgSb.append("in tensor shape: [");
            int[] shape = session.getInputs().get(0).getShape();
            for (int dim : shape) {
                msgSb.append(dim).append(",");
            }
            msgSb.append("]");
            Log.i(TAG, msgSb.toString());
        }

        return session;
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
            Log.e(TAG, "output tensor shape do not float, the data type is " + outTensor.getDataType());
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

    private boolean runInference(LiteSession session) {
        Log.i(TAG, "runInference: ");
        MSTensor inputTensor = session.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
        if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
            Log.e(TAG, "Input tensor shape do not float, the data type is " + inputTensor.getDataType());
            return false;
        }
        // Generator Random Data.
        int elementNums = inputTensor.elementsNum();
        float[] randomData = generateArray(elementNums);
        byte[] inputData = floatArrayToByteArray(randomData);

        // Set Input Data.
        inputTensor.setData(inputData);

        session.bindThread(true);
        // Run Inference.
        boolean ret = session.runGraph();
        session.bindThread(false);
        if (!ret) {
            Log.e(TAG, "MindSpore Lite run failed.");
            return false;
        }


        // Get Output Tensor Data.
        MSTensor outTensor = session.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
        // Print out Tensor Data.
        ret = printTensorData(outTensor);
        if (!ret) {
            return false;
        }

        outTensor = session.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204").get(0);
        ret = printTensorData(outTensor);
        if (!ret) {
            return false;
        }

        Map<String, MSTensor> outTensors = session.getOutputMapByTensor();

        Iterator<Map.Entry<String, MSTensor>> entries = outTensors.entrySet().iterator();
        while (entries.hasNext()) {
            Map.Entry<String, MSTensor> entry = entries.next();

            Log.i(TAG, "Tensor name is:" + entry.getKey());
            ret = printTensorData(entry.getValue());
            if (!ret) {
                return false;
            }
        }

        return true;
    }

    private void freeBuffer() {
        session1.free();
        session2.free();
        model.free();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        String version = Version.version();
        Log.i(TAG, version);
        model = new Model();
        String modelPath = "mobilenetv2.ms";
        boolean ret = model.loadModel(this.getApplicationContext(), modelPath);
        if (!ret) {
            Log.e(TAG, "Load model failed, model is " + modelPath);
        } else {
            session1 = createLiteSession(false);
            if (session1 != null) {
                session1Compile = true;
            } else {
                Toast.makeText(getApplicationContext(), "session1 Compile Failed.",
                        Toast.LENGTH_SHORT).show();
            }
            session2 = createLiteSession(true);
            if (session2 != null) {
                session2Compile = true;
            } else {
                Toast.makeText(getApplicationContext(), "session2 Compile Failed.",
                        Toast.LENGTH_SHORT).show();
            }
        }

        if (model != null) {
            // Note: when use model.freeBuffer(), the model can not be compiled again.
            model.freeBuffer();
        }

        TextView btn_run = findViewById(R.id.btn_run);
        btn_run.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (session1Finish && session1Compile) {
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            session1Finish = false;
                            runInference(session1);
                            session1Finish = true;
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
                        if (session1Finish && session1Compile) {
                            new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    session1Finish = false;
                                    runInference(session1);
                                    session1Finish = true;
                                }
                            }).start();
                        }
                        if (session2Finish && session2Compile) {
                            new Thread(new Runnable() {
                                @Override
                                public void run() {
                                    session2Finish = false;
                                    runInference(session2);
                                    session2Finish = true;
                                }
                            }).start();
                        }
                        if (!session2Finish && !session2Finish) {
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