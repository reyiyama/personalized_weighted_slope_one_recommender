# Personalized Weighted Slope One Recommender

## Overview
This repository contains the implementation of a **Personalized Weighted Slope One** recommendation system, developed as part of an academic project for **PDS 2310 Assignment 3**. The project aims to combine collaborative filtering methodologies with weighted deviation schemes to enhance recommendation quality through personalization. Specifically, we build upon the **Slope One scheme** by introducing a weighted component based on user similarity, thereby creating a **Personalized Weighted Slope One method** that provides better prediction accuracy for user preferences.

The dataset used here is part of the **Movielens** dataset, with split configurations in multiple training and testing files (`u1.base`, `u1.test`, etc.) provided in the `ml-100k` directory.

## Data Science Techniques Demonstrated
- **Collaborative Filtering**: Leveraging **Slope One** and its enhancements for a recommendation system.
- **User Similarity Calculations**: Introducing personalized weightings based on **Centered Cosine Similarity** to improve the prediction accuracy.
- **Weighted Aggregation of Ratings**: Using customized weighted deviations to account for user similarity.
- **Algorithm Evaluation**: Predictive accuracy evaluation to compare personalized versus non-personalized approaches.

## Project Motivation
The **Slope One Predictors** are a simple yet efficient collaborative filtering algorithm with advantages such as being easy to implement, efficient at query time, and updateable. The goal of this project is to combine the benefits of Slope One predictors with personalization for enhanced user-specific recommendations.

By considering the similarity between the active user and other users, the proposed **Personalized Weighted Slope One** scheme aims to overcome limitations of the basic and weighted versions of Slope One by providing more relevant recommendations. This can be especially impactful in recommender systems for streaming platforms, e-commerce sites, and any context where personalization is critical to user experience.

## Slope One Equations and Extensions

### Weighted Slope One Deviation
The formula for the deviation between items \( j \) and \( i \) is defined as follows:
<img width="689" alt="Screenshot 2024-10-09 at 2 45 11 PM" src="https://github.com/user-attachments/assets/075ffba3-4743-497f-8d6d-25c3d48dc64e">


Where:
- \( \lambda \) is a pre-defined parameter between 0 and 1 that balances the influence of general and personalized ratings.
- \( S_{j,i}(\chi) \) represents the set of users who rated both items \( j \) and \( i \).
- \( \text{sim}(u, u') \) is the **Centered Cosine Similarity** between users \( u \) and \( u' \).

### Prediction for Personalized Weighted Slope One
The predicted rating for item \( j \) by user \( u' \) is given by:
<img width="366" alt="Screenshot 2024-10-09 at 2 45 45 PM" src="https://github.com/user-attachments/assets/08e11c5b-3892-477f-affa-0ce25afd1f49">

Where:
- \( S(u') \) is the set of items rated by user \( u' \), excluding item \( j \).
- \( c_{j,i} \) is the number of users who rated both items \( j \) and \( i \).

## Implementation Details
The main implementation is provided in the **Jupyter Notebook** `assignment3framework.ipynb`, which includes:
- Loading and preprocessing the data from the **Movielens 100k dataset**.
- Implementing the **Weighted Slope One** and the **Personalized Weighted Slope One** methods.
- Evaluating prediction quality using **Mean Absolute Error (MAE)**, similar to evaluation techniques from related literature.

The implementation leverages standard Python libraries such as `NumPy` for numerical calculations, without relying on third-party recommender system libraries, ensuring that the core algorithmic understanding is demonstrated.

## Folder Structure
- `ml-100k/` - Contains the Movielens dataset files, such as `u.data`, `u.item`, `u.genre`, as well as training and test splits (`u1.base`, `u1.test`, etc.).
- `assignment3framework.ipynb` - Jupyter Notebook with the implementation and results.
- `Slides.pdf` - Presentation explaining Slope One, Weighted Slope One, and Personalized Weighted Slope One schemes.
- `Slope One Predictors for Online Rating-Based Collaborative Filtering.pdf` - Research paper that provided foundational concepts for the project.
- `Presentation_video.mp4` - Recorded presentation of the project.

## Key Findings
- **Parameter Tuning**: The parameter \( \lambda \) balances between general and personalized contributions, allowing fine control over how much the model relies on global versus user-specific trends.
- **MAE Evaluation**: The **Personalized Weighted Slope One** scheme demonstrated a reduction in **Mean Absolute Error** compared to the non-personalized versions, indicating improved recommendation accuracy.

## Applications
- **Recommender Systems**: The personalized slope one approach can be directly used in streaming services like Netflix or e-commerce websites such as Amazon to provide tailored recommendations that consider both general user behavior and specific user preferences.
- **Data Mining and Knowledge Discovery**: This project exemplifies how collaborative filtering can be leveraged to extract valuable knowledge from user rating patterns, which is essential in applications like **targeted advertising** and **content personalization**.

## How to Run
1. Clone this repository: 
    ```sh
    git clone https://github.com/reyiyama/personalized_weighted_slope_one_recommender.git
    ```
2. Ensure you have Python installed with the necessary packages (`NumPy`, `pandas`, `Jupyter`).
3. Open the Jupyter Notebook:
    ```sh
    jupyter notebook assignment3framework.ipynb
    ```
4. Execute all cells to reproduce the results.

## References
- **Daniel Lemire and Anna Maclachlan**, "Slope One Predictors for Online Rating-Based Collaborative Filtering," SIAM Data Mining (SDM’05).
