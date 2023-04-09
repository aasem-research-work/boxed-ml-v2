# Boxed-ML V2

This project provides a collection of scripts for training, predicting/inferencing, and explaining machine learning models using various explainable AI techniques such as Class Activation Mapping (CAM), Layer-wise Relevance Propagation (LRP), Local Interpretable Model-agnostic Explanations (LIME), and more. The primary goal is to make it easy for users to understand and interpret the predictions made by their models, thereby increasing transparency and trust in the AI systems.

# Environment
To set up the environment locally to run the scripts, we recommend using conda for easy package management. First, install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already. Then, follow the steps below:

1. Create a new conda environment:  
`conda create --name boxedml python=3.7'   
2. activate the conda environment:  
`conda activate boxedml'   

3.  Install the required packages:  
`conda install tensorflow keras numpy pandas matplotlib scikit-learn scikit-image
conda install -c conda-forge lime
`

Note: You may need to install additional packages depending on the specific explainable AI techniques used in the project.  

Now you are ready to run the scripts in this project.  

# Usage
To train a model, run the following command:  
`python train.py
`
This script will train the model based on the configurations specified in the PARAM dictionary and save the trained model to the specified workspace directory.  

To perform inference on a new dataset using the trained model, run the following command:  
`python inference.py
`

This script will load the trained model from the workspace directory and predict the class of each image in the test dataset. The predictions, along with the top probabilities and class indices, will be saved to a JSON file in the workspace directory.  

For more information on using and customizing the scripts, refer to the comments and documentation within each script.




