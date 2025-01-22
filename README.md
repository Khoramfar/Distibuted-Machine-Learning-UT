# Distributed Machine Learning Systems: Assignments

This repository includes all **Computer Assignments (CAs)** and **Homework Assignments (HWs)** from the **Distributed Machine Learning Systems (DMLS)** course. The course is designed to bridge the gap between distributed systems and machine learning by teaching students how to train large-scale models across distributed environments. Key areas covered include high-performance computing, privacy-preserving algorithms, big data processing, and neural network optimization.

---

## **Assignment Summaries**

### **CA1: Distributed Algorithms and Optimization**
This assignment focuses on distributed computing fundamentals, algorithm optimization, and benchmarking:
- Implemented **Taylor series expansion** using MPI and analyzed computational overheads.
- Designed a privacy-preserving **distributed logistic regression** model and implemented it using **mpi4py**.
- Benchmarked matrix multiplication and inversion using different **BLAS libraries** with NumPy, exploring the impact of hardware acceleration.

---

### **CA2: CUDA Programming and PyTorch DDP**
This assignment introduces GPU acceleration and distributed deep learning:
- Converted **color images to grayscale** using both Python and CUDA to compare computation times.
- Trained convolutional neural networks on **STL-10 dataset** using **PyTorch Distributed Data Parallel (DDP)**.
- Explored the impact of **batch size**, **backends** (e.g., NCCL, GLOO), and multi-GPU setups on training speed and memory consumption.

---

### **CA3: Big Data Processing and Language Models**
This assignment leverages **PySpark** and big data frameworks for large-scale machine learning:
- Processed **Shahnameh** text data using **n-grams** and implemented a **language model**.
- Developed **Word2Vec embeddings** using skip-gram and CBOW methods for semantic analysis of key terms.
- Applied **K-Means clustering** on customer data stored in **HDFS**, visualizing results to segment customers by income and spending habits.

---

### **CA4: Distributed Neural Network Training and Profiling**
This assignment delves into distributed training pipelines and profiling tools:
- Trained a **Feedforward Neural Network** with **BatchNorm** and **ReLU** layers using **SLURM** and **torchrun** across multiple machines.
- Used **HuggingFace Accelerate** to simplify multi-GPU training setups, implementing mixed precision and distributed model evaluation.
- Profiled the memory and computation costs of activation functions (e.g., ReLU, Sigmoid, GeLU) using the **PyTorch Profiler**, analyzing their efficiency.

---

## **Usage**
This repository is structured to highlight both technical implementations and experimental results. Each assignment is self-contained with:
- Source code for all algorithms and models.
- Detailed reports explaining methodologies, results, and analyses.

For further inquiries, feel free to open an issue or contact the repository maintainer.
