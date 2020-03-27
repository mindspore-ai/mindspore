# Security Risk Description

1. When MindSpore is used for AI model training, if the user-defined computational graph structure (for example, Python code for generating the MindSpore computational graph) is provided by an untrusted third party, malicious code may exist and will be loaded and executed to attack the system.
2. Model files are stored in binary mode. When MindSpore is used to optimize or infer AI models and the model files are loaded in deserialization mode, once malicious code is written into the model files, the code are loaded and executed, causing attacks on the system.
3. MindSpore performs only model training and inference based on the data provided by users. Users need to protect data security to avoid privacy leakage.
4. MindSpore is a distributed training platform. When MindSpore is used for distributed training, if an Ascend chip is used for training, a device provides a secure transmission protocol for gradient fusion. If GPUs or other clusters are used for training, identity authentication and secure transmission are not provided.

# Security Usage Suggestions

1. Run MindSpore in the sandbox.
2. Run MindSpore as a non-root user.
3. Ensure that the source of a computational graph structure is trustworthy. Do not write code irrelevant to model training in the network structure definition.
4. Ensure that the source of a network model is trustworthy or enter secure network model parameters to prevent model parameters from being tampered with.
5. Ensure that GPU distributed training is performed on an isolated cluster network.
