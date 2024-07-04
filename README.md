# Movie Recommendation System
### Table of Contents

1. Introduction
2. Data Preparation
3. Model Architecture
4. Training and Evaluation
5. Recommendation Generation
6. Technical Details
7. Future Improvements

## Introduction
This project implements a movie recommendation system using collaborative filtering and deep learning techniques. The system is designed to provide personalized movie recommendations by analyzing user interaction history, movie features, and user demographics.
The core of the system is a neural network model that learns to predict user ratings for movies based on historical data. It leverages techniques such as sequence modeling, attention mechanisms, and multi-task learning to capture complex patterns in user behavior and movie characteristics.

Key features of the system include:

- Utilization of the MovieLens 1M dataset and IMDB datasets
- Incorporation of both user and movie features
- Sequence-based modeling of user viewing history
- Attention mechanism for weighing the importance of different movies in a user's history
- Paired user approach for collaborative filtering

Before arriving to this result, i tried several models and techniques, such as:
- Matrix Factorization
- Deep Learning with Embeddings
- Collaborative Filtering

For the matrix factorization and collaborative filtering i had good results, but i wanted to try a more complex model, so i decided to use a deep learning model with attention mechanism.

If you want to check the user-item interaction matrix, you can run the matrix.ipynb file.

For the embedding model, i had the problem that i need to create 3 different embeddings for the user, movie and genre, and i wanted to create a model that could use all the information in a single model. The solution was to split the model in 2, one for the user and another for the movie, and then combine the information in a single model but it was a costly solution in terms of time.

## Data Preparation
### Dataset Acquisition and Processing

1. **Data Download**: The system downloads several datasets:

   - MovieLens 1M dataset (movies, ratings, users)
   - IMDB datasets (name basics, title basics, title ratings)


2. **Data Extraction**: Compressed files are unzipped and extracted to a local directory.
3. **Data Loading and Cleaning**:

   - MovieLens data is loaded into pandas DataFrames.
   - IMDB data is processed to extract relevant movie information.
   - Data is cleaned by removing duplicates, handling missing values, and normalizing formats.


4. **Feature Engineering**:

   - User IDs and movie IDs are converted to a consistent format.
   - Movie titles are cleaned and standardized.
   - Genre information is merged from MovieLens and IMDB sources.
   - Runtime information is added from IMDB data.
   - Actor/director information is encoded using LabelEncoder.



## Sequence Creation

1. **Viewing History Sequences**:

   - User viewing histories are converted into sequences of fixed length (4 in this case).
   - A sliding window approach with a step size of 2 is used to create overlapping sequences.


2. **Rating Sequences**:

   - Corresponding rating sequences are created to match the movie sequences.


3. **Sequence Padding**:

   - Sequences shorter than the defined length are padded with None values.



## User Pairing
A novel approach is implemented to pair users who have rated the same movies:

1. The last movie in each sequence is treated as the target movie.
2. Users who have rated the same target movie are paired.
3. The average rating of the paired users for the target movie is used as the target rating.

This pairing approach allows the model to learn from collaborative information, enhancing its ability to make recommendations based on similar users' preferences.

## Model Architecture
The recommendation system employs a deep learning model with the following key components:
### Input Layer
The model takes inputs for two users simultaneously, including:

- User IDs
- Sequence of movie IDs watched by each user
- Sequence of ratings given by each user
- Target movie ID
- User features (sex, age group, occupation)

### Feature Encoding

1. **Categorical Feature Encoding**:
   - StringLookup layers are used to encode categorical features like user IDs, movie IDs, and user attributes.


2. **Movie Feature Embedding**:
   - Movie features (genres, runtime, actor/director encoding) are embedded into a dense vector representation.


3. **Sequence Processing**:

   - A custom SequenceProcessor layer handles the movie ID and rating sequences, replacing empty strings with 'unknown_movie' and 0.0 ratings respectively.



### Attention Mechanism

1. **Multi-Head Attention**:

   - Applied to the sequence of watched movies to weigh the importance of different movies in a user's history.
   - Helps the model focus on the most relevant past interactions when making predictions.


2. **Layer Normalization and Residual Connections**:

   - Used to stabilize the learning process and allow for deeper networks.



### Feature Combination

1. **User Features**:

   - Transformed user features are concatenated with the output of the attention layer.


2. **Movie Features**:

   - Target movie features are combined with user features.



### Fully Connected Layers

1. **Dense Layers**:

   - Multiple dense layers with LeakyReLU activation process the combined features.


2. **Batch Normalization**:

   - Applied after each dense layer to normalize the activations and stabilize training.


3. **Dropout**:

   - Used for regularization to prevent overfitting.



### Output Layer
A single dense unit produces the final rating prediction.

## Training and Evaluation
### Data Pipeline

1. **CSV Reading**:

   - Training and test data are read from CSV files using TensorFlow's data pipeline.


2. **Batching and Shuffling**:

   - Data is batched and shuffled to improve training efficiency and generalization.



## Model Compilation

1. **Optimizer**:

   - Adagrad optimizer is used with a learning rate of 0.01.


2. **Loss Function**:

   - Mean Squared Error (MSE) is used as the loss function for rating prediction.


3. **Metrics**:

   - Mean Absolute Error (MAE) is tracked during training and evaluation.



## Training Process

1. **Epoch-based Training**:

The model is trained for 10 epochs on the training dataset.


2. **Verbose Output**:

Training progress is displayed, showing loss and MAE for each epoch.



## Evaluation

1. **Test Dataset**:

A separate test dataset is used to evaluate the model's performance.


2. **Metrics Calculation**:

Root Mean Square Error (RMSE) is calculated on the test set to measure prediction accuracy.



## Recommendation Generation
The trained model can be used to generate movie recommendations for pairs of users:

1. **User Input**:

   - Two user IDs are provided as input.


2. **Feature Extraction**:

   - User features and viewing histories are extracted for the given users.


3. **Candidate Generation**:

   - A set of candidate movies is generated, possibly including popular movies or movies similar to those the users have watched.


4. **Rating Prediction**:

   - The model predicts ratings for each candidate movie for the given user pair.


5. **Ranking**:

   - Candidate movies are ranked based on the predicted ratings.


6. **Top-N Recommendations**:

   - The top N movies with the highest predicted ratings are returned as recommendations.



## Technical Details
### Libraries and Frameworks

- TensorFlow and Keras for model implementation and training
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for preprocessing tasks

### Custom Components

1. **SequenceProcessor Layer**:

   - A custom Keras layer for processing movie ID and rating sequences.


2. **Attention Mechanism**:

   - Implemented using Keras' MultiHeadAttention layer.


3. Feature Encoding:

   - Custom functions for encoding user and movie features.

### Hyperparameters

- Sequence length: 4
- Step size for sequence creation: 2
- Hidden units in dense layers: [256, 128]
- Dropout rate: 0.1
- Number of attention heads: 3
- Batch size: 32
- Learning rate: 0.01
- Number of epochs: 10

## Model Size and Complexity
The model's architecture results in a significant number of trainable parameters, primarily due to the embedding layers for user and movie IDs, and the dense layers in the network.

## Future Improvements

1. **Hyperparameter Tuning**:

   - Implement a systematic approach to find optimal hyperparameters, such as using grid search or Bayesian optimization.


2. **Cold Start Problem**:

   - Develop strategies for recommending movies to new users or handling new movies with limited rating data.


3. **Temporal Dynamics**:

   - Incorporate time-based features to capture changing user preferences over time.


4. **Diversity and Serendipity**:

   - Implement techniques to ensure diverse recommendations and introduce an element of surprise in recommendations.


5. **Explainability**:

   - Add features to explain why certain movies are recommended, improving user trust and system transparency.


6. **Scalability**:

   - Optimize the model and data pipeline for larger datasets and real-time recommendations.


7. **Multi-modal Data**:

   - Incorporate additional data sources such as movie posters, trailers, or user reviews to enhance recommendation quality.


8. **Online Learning**:

   - Implement mechanisms for continuous model updates as new user interactions become available.



By implementing these improvements, the recommendation system can be further enhanced to provide more accurate, diverse, and personalized movie recommendations.