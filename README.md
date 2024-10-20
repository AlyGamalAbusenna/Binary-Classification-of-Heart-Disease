# Binary-Classification-of-Heart-Disease
Binary Classification of Heart Disease using a Multi-Layer Perceptron (MLP) with Keras

## Objective
The goal of this project is to train a deep learning model using Keras to solve a specific problem using the provided dataset. The project also demonstrates how to visualize the model's training process using TensorBoard, evaluate its performance using ROC-AUC curves, and analyze training and validation losses.

## Dataset Description
The dataset used in this project is focused on [describe the type of data here, e.g., classification of images, sentiment analysis, etc.]. The dataset is divided into training and validation subsets to facilitate model learning and evaluate its performance during training. Details on the dataset, such as its source and format, should be included to help users understand its structure and purpose.

## Steps to Run the Code in Jupyter
1. **Clone the Repository**: First, clone the GitHub repository to your local machine.
   ```bash
   git clone <repository_link>
   cd <repository_name>
   ```

2. **Open Jupyter Notebook**: Launch Jupyter Notebook to open the project notebook (`Deep2_keras.ipynb`).
   ```bash
   jupyter notebook
   ```

3. **Install Dependencies**: Make sure all necessary dependencies are installed by running the following command in a terminal:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**: Open the Jupyter Notebook file (`Deep2_keras.ipynb`) and execute the cells in sequence.
   - The notebook includes code for loading the dataset, preprocessing, model training, and performance evaluation.
   - Ensure you execute each cell one by one, following the logical flow from data loading to model training.

## Dependencies and Installation Instructions
- **TensorFlow**: Used for deep learning model training and evaluation. Install TensorFlow using the command:
  ```bash
  pip install tensorflow
  ```
- **Keras**: The model is built using the Keras API, which is included with TensorFlow.
- **Matplotlib**: Required for visualizing ROC-AUC curves and other plots.
  ```bash
  pip install matplotlib
  ```
- **Scikit-Learn**: Used for calculating the ROC-AUC score.
  ```bash
  pip install scikit-learn
  ```
- **Jupyter Notebook**: Required to run the code interactively.
  ```bash
  pip install notebook
  ```
- **TensorBoard**: For visualizing training logs and monitoring the model's learning progress.
  ```bash
  pip install tensorboard
  ```

## Visualizations and Performance Evaluation

### TensorBoard Visualizations
- **Training and Validation Loss**: During the model training process, the loss for both training and validation datasets is recorded and can be visualized using TensorBoard. To start TensorBoard and see these graphs, run the following command in the terminal:
  ```bash
  tensorboard --logdir=logs/fit
  ```
- **Histograms and Metrics**: TensorBoard also provides histograms and metric visualizations that help in understanding model performance during training.

### ROC-AUC Curves
- After training, the model's performance is evaluated using the ROC-AUC metric, which is a popular metric for evaluating classification models. The ROC curve provides insight into the model's ability to distinguish between classes, and the AUC score is calculated to summarize the overall performance.
- The ROC-AUC curve is plotted using Matplotlib, and the plot can be found in the notebook under the performance evaluation section.

## Notes
- Make sure to update the paths to the dataset or other relevant files as per your local setup.
- The TensorBoard logs are saved in the `logs/fit` directory, which can be modified as per user requirements.

Feel free to explore the notebook, modify the hyperparameters, and see how the model's performance changes!

