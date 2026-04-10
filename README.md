# DGCD: Understanding Group Cognition through Explainable Diffusion-Based Diagnosis Model

This paper presents a graph diffusion-based collaborative diagnosis model for assessing students' knowledge states in the educational domain. The proposed approach further incorporates a streaming graph diffusion-based group cognitive diagnosis framework (GCDE), which models the natural propagation of knowledge across different hierarchical levels through a streaming knowledge diffusion mechanism.

## GCDE key characteristics:
It formulates knowledge propagation as a continuous flow from individual students to intra-group interactions and further to inter-group dynamics, where a graph diffusion mechanism captures the natural transfer of knowledge across hierarchical levels. It further introduces a flow-aware attention mechanism that serves as a control module for knowledge propagation, which adaptively regulates both the intensity and direction of knowledge flow and thereby improves interpretability. To maintain coherence during propagation, the model imposes a flow consistency constraint that enforces alignment of knowledge representations across different levels and enhances overall performance. In addition, the graph diffusion process draws on the principles of diffusion models, where the iterative addition and removal of noise improves the model's capacity to capture complex distributions of knowledge states.


## Quick Start


```bash
pip install torch numpy scikit-learn
```

### Data format

The model supports two data formats：
1. In the CSV format, each row contains the student ID, exercise ID, class ID, and label (0/1).
2. In the JSON format, each element in the list is a dictionary that includes the fields stu_id, exer_id, class_id, and label.

### Train the original DGCD model

```bash
python train.py --data_path data/SLPbio_ --num_train_epochs 100 --batch_size 32 --verbose
```

### Train DGCD model

```bash
python train_flowdiff.py --data_path ./data/SLPbio_/ --train_file train.json --val_file val.csv --test_file test.json --skill_n 21 --batch_size 32 --epochs 100
```

### Parameter Description

Parameters of the DGCD model:
- `--data_path`: data path
- `--num_train_epochs`: number of training epochs
- `--batch_size`: batch size
- `--lr`: learning rate
- `--prune_threshold`: pruning threshold
- `--weight_bits`: number of bits for weight quantization
- `--verbose`: enable verbose output

### Test

```bash
python test.py --data_path data/SLPbio_ --model_path data/SLPbio_model.pt
```

## Model Performance

Compared with traditional group cognitive diagnosis methods, the newly proposed GCDE model has the following advantages:：

1. **Capturing the dynamics of knowledge flow**：Through a flow-based graph diffusion mechanism, it can capture the continuous changes of knowledge states over time.
2. **Modeling cross-group knowledge flow**：It effectively models knowledge flow between groups and accurately evaluates cross-group knowledge influence.
3. **Enhanced interpretability**：By incorporating flow attention and gating mechanisms, it provides interpretability for the knowledge flow process.

