
# Objective

The goal of this tool is to allow the user to reduce the size of MindData lite package they ship with their code.

# How to run

This tool has two parts: the first part only needs to be run once, when the source code for mindspore is changed
while the second part should be run every time the user code changes.

Note that you need to run this tool on the server side if you are planning to use your code on an edge device.

## Step 1: Configure the cropper tool

You need to have mindspore installed on your system to run this python script.
Additionally, you need to have the mindspore source code present in your system
as this script processes mindspore's source code.

To execute the first part simply run:

```console
python cropper_configure.py
```

## Step 2: Crop the MindData lite package

The second part needs to be run every time the user adds or removes one of MD operators in their code.

For the second part, you need to run:

```console
./crop.sh -p <path to mindspore package> <source files>
```

Note that you need to provide the name of all files that are using any of the MindData functionalities.

`ANDROID_NDK` environment variable needs to be set as well if the target device is android.

Example: `./crop.sh -p ~/mindspore/ foo.cc foo.h bar.cc bar.h`

This code will create the __libminddata-lite_min.so__ library specific to your code and will also print for you a list of
shared objects that your code depends on (including __libminddata-lite\_min.so__).
Note that you need to copy these files to your target device and set the linker flag accordingly.

# How it works

The first step (configuration) creates a few of files that are needed in the second step.
These files include _dependencies.txt_, _associations.txt_, and _debug.txt_.
While the third file (_debug.txt_) is only for debugging purposes (debugging cropper tool),
the other two files are used in the second part.
_associations.txt_ contains the entry points (IR level source files) for ops that the user may use in their code.
The other file, _dependencies.txt_, contains all dependencies for all those entry points.

When the user runs the crop script, _parser.py_ will be run on their code to find the ops they have used.
Afterwards, the text files will be used to keep the needed object files
(by removing unnecessary object files from the static library containing all of them).
Finally, the remaining object files will be used to create a new shared object file (_libminddata-lite\_min.so_).

# Requirements

Step 1:

* Python3
* mindspore
* mindspore source code

Step 2:

* Python3
* cmake
* Android NDK (if target device is android)