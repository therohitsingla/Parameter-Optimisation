# SVM Parameter Optimization Experiment

## Overview
This repository contains a comprehensive implementation of Support Vector Machine (SVM) parameter optimization using the Breast Cancer Wisconsin dataset. The experiment systematically evaluates SVM performance across multiple training/testing splits and kernel configurations, tracking convergence and identifying optimal parameters for breast cancer classification.

## Features
- **Comprehensive Hyperparameter Optimization**: Tests multiple kernels, C values, and gamma parameters using random search
- **Multi-Sample Evaluation**: Runs experiments across 10 different train/test splits to ensure robust results
- **Detailed Visualization**: Includes convergence tracking, kernel comparison, and parameter effect analysis
- **Performance Metrics**: Provides detailed accuracy measurements and classification reports
- **Real-World Application**: Applies machine learning to the important task of breast cancer classification

## Dataset
The experiment uses the `load_breast_cancer` dataset from `sklearn.datasets`, with the following properties:
- 569 samples
- 30 numeric features
- Binary classification (malignant vs benign tumors)
- Standardized preprocessing using `StandardScaler`

## Methodology
1. **Data Preparation**: Load and standardize the Breast Cancer dataset
2. **Parameter Space Definition**: Define search space for SVM hyperparameters
   - Kernels: linear, rbf, poly
   - C values: logarithmic range from 0.001 to 1000
   - Gamma values: logarithmic range from 0.001 to 1000
3. **Multi-Sample Testing**: Perform 10 iterations of different train/test splits (70%/30%)
4. **Random Search**: For each split:
   - Test 100 random SVM configurations
   - Track best accuracy and parameters per sample
   - Monitor convergence for best-performing configuration
5. **Final Model**: Train a final model using the best parameters identified
6. **Evaluation**: Assess performance using confusion matrix and classification report

## Results

### Sample-by-Sample Performance

| Sample | Best Accuracy | Parameters (kernel, C, gamma) |
|--------|--------------|------------------------------|
| S1     | 0.971        | rbf, 0.464, 0.215            |
| S2     | 0.977        | poly, 21.544, 0.046          |
| S3     | 0.965        | linear, 100.0, —             |
| S4     | 0.982        | rbf, 2.154, 0.100            |
| S5     | 0.971        | poly, 10.0, 0.001            |
| S6     | 0.977        | rbf, 0.215, 0.464            |
| S7     | 0.971        | linear, 10.0, —              |
| S8     | 0.977        | poly, 2.154, 0.215           |
| S9     | 0.977        | rbf, 1.0, 0.100              |
| S10    | 0.971        | linear, 2.154, —             |

### Overall Best Configuration

- **Best Accuracy**: 0.982
- **Best Kernel**: rbf
- **Best C Value**: 2.154
- **Best Gamma**: 0.100
- **Found in Sample**: S4

### Final Model Performance

- **Training Accuracy**: 0.995
- **Testing Accuracy**: 0.965

### Confusion Matrix

|              | Predicted Benign | Predicted Malignant |
|--------------|------------------|---------------------|
| True Benign  | 105              | 2                   |
| True Malignant | 4                | 60                  |

### Classification Report

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Benign       | 0.96      | 0.98   | 0.97     | 107     |
| Malignant    | 0.97      | 0.94   | 0.95     | 64      |
| **Accuracy** |           |        | **0.97** | **171** |

## Key Findings

1. **Kernel Performance**:
   - RBF kernel achieved the highest accuracy overall
   - Polynomial kernel showed strong performance in some samples
   - Linear kernel was competitive but slightly less accurate

2. **Parameter Sensitivity**:
   - Moderate C values (1-10) generally performed better than extreme values
   - For RBF kernel, gamma values around 0.1-0.5 yielded best results
   - For polynomial kernel, lower gamma values often worked better

3. **Model Stability**:
   - High consistency across different samples (accuracy range: 0.965-0.982)
   - Final model showed excellent balance of precision and recall

## Visualizations

### Convergence Plot
![Convergence Plot](figures/convergence_example.png)

The convergence plot shows how accuracy improved over iterations for the best sample, ultimately reaching 0.982.

### Kernel Comparison
![Kernel Comparison](figures/kernel_comparison_example.png)

Box plots comparing the distribution of accuracy scores across different kernel types.

### C Parameter Effect
![C Parameter Effect](figures/c_parameter_effect_example.png)

Scatter plots showing the relationship between C parameter values and accuracy for each kernel type.

## Usage

```bash
# Clone the repository
git clone https://github.com/therohitsingla/Parameter-Optimisation.git
cd svm-parameter-optimization

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook SVM_Parameter_Optimization.ipynb
```

## Dependencies
- Python 3.x
- NumPy
- pandas
- scikit-learn
- Matplotlib
- seaborn
- Jupyter
- tqdm

## Conclusion
This experiment demonstrates the effectiveness of SVM for breast cancer classification, achieving over 97% accuracy with optimized parameters. The results highlight the importance of proper hyperparameter tuning for maximizing model performance in medical diagnostic applications.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
