import com.mindspore.lite.DataType;
import com.mindspore.lite.LiteSession;
import com.mindspore.lite.MSTensor;
import com.mindspore.lite.Model;
import com.mindspore.lite.config.DeviceType;
import com.mindspore.lite.config.MSConfig;

import java.io.*;

public class Benchmark {
    private static Model model;
    private static LiteSession session;

    public static byte[] readBinFile(String fileName, int size) {
        try {
            DataInputStream is = new DataInputStream(
                    new BufferedInputStream(new FileInputStream(
                            fileName)));
            byte[] buf = new byte[size];
            is.read(buf);
            is.close();
            return buf;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static boolean compareData(String filePath, float accuracy) {
        double meanError = 0;
        File file = new File(filePath);
        if (file.exists()) {
            try {
                FileReader fileReader = new FileReader(file);
                BufferedReader br = new BufferedReader(fileReader);
                String lineContent = null;

                int line = 0;
                MSTensor outTensor = null;
                String name = null;
                while ((lineContent = br.readLine()) != null) {
                    String[] strings = lineContent.split(" ");
                    if (line++ % 2 == 0) {
                        name = strings[0];
                        outTensor = session.getOutputByTensorName(name);
                        continue;
                    }
                    float[] benchmarkData = new float[strings.length];
                    for (int i = 0; i < strings.length; i++) {
                        benchmarkData[i] = Float.parseFloat(strings[i]);
                    }
                    float[] outData = outTensor.getFloatData();
                    int errorCount = 0;
                    for (int i = 0; i < benchmarkData.length; i++) {
                        double relativeTolerance = 1e-5;
                        double absoluteTolerance = 1e-8;
                        double tolerance = absoluteTolerance + relativeTolerance * Math.abs(benchmarkData[i]);
                        double absoluteError = Math.abs(outData[i] - benchmarkData[i]);
                        if (absoluteError > tolerance) {
                            if (Math.abs(benchmarkData[i] - 0.0f) < Float.MIN_VALUE)
                                if (absoluteError > 1e-5) {
                                    meanError += absoluteError;
                                    errorCount++;
                                } else {
                                    continue;
                                }
                        } else {
                            meanError += absoluteError / (Math.abs(benchmarkData[i]) + Float.MIN_VALUE);
                            errorCount++;
                        }
                    }

                    if (meanError > 0.0f) {
                        meanError /= errorCount;
                    }
                    if (meanError <= 0.0000001) {
                        System.out.println("Mean bias of node/tensor " + name + " : 0%");
                    } else {
                        System.out.println("Mean bias of node/tensor " + name + " : " + meanError * 100 + "%");
                    }
                }
                br.close();
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return meanError < accuracy;
    }

    private static boolean compile() {
        MSConfig msConfig = new MSConfig();
        boolean ret = msConfig.init(DeviceType.DT_CPU, 2);
        if (!ret) {
            System.err.println("Init context failed");
            return false;
        }

        // Create the MindSpore lite session.
        session = new LiteSession();
        ret = session.init(msConfig);
        msConfig.free();
        if (!ret) {
            System.err.println("Create session failed");
            model.free();
            return false;
        }

        // Compile graph.
        ret = session.compileGraph(model);
        if (!ret) {
            System.err.println("Compile graph failed");
            model.free();
            return false;
        }
        return true;
    }


    private static void freeBuffer() {
        session.free();
        model.free();
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("We must pass parameters such as modelPath, inDataFile, benchmarkDataFile and accuracy.");
            return;
        }

        String modelPath = args[0];
        String[] inDataFile = args[1].split(",");
        String benchmarkDataFile = args[2];
        float accuracy = Float.parseFloat(args[3]);

        model = new Model();


        boolean ret = model.loadModel(modelPath);
        if (!ret) {
            System.err.println("Load model failed, model path is " + modelPath);
            return;
        }
        ret = compile();
        if (!ret) {
            System.err.println("MindSpore Lite compile failed.");
            return;
        }
        for (int i = 0; i < session.getInputs().size(); i++) {
            MSTensor inputTensor = session.getInputs().get(i);
            if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
                System.err.println("Input tensor shape do not float, the data type is " + inputTensor.getDataType());
                freeBuffer();
                return;
            }
            // Set Input Data.
            byte[] data = readBinFile(inDataFile[i], (int) inputTensor.size());
            inputTensor.setData(data);
        }

        // Run Inference.
        if (!session.runGraph()) {
            System.err.println("MindSpore Lite run failed.");
            freeBuffer();
            return;
        }

        boolean benchmarkResult = compareData(benchmarkDataFile, accuracy);
        freeBuffer();
        if (!benchmarkResult) {
            System.err.println(modelPath + " accuracy error is too large.");
            System.exit(1);
        }
    }
}
