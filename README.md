**Status:** Archive (code is provided as-is, no updates expected)

# **Identifying**-Complementary-Patents-Based-on-Deep-Learning

Code and models from the paper “An Approach for Identifying Complementary Patents Based on Deep Learning”

This article proposes three types of patent semantic representation embedding methods, namely network embedding, text embedding, and fusion embedding, to capture the structural and textual content features of patents. A deep learning framework enhanced by Convolutional Block Attention Module (CBAM) is proposed to handle complex interactions between different dimensions of patent representation. This article verifies the effectiveness of the proposed method in complementary patent recognition tasks.

## Requirements

To run this code you need the following:

| Libraries     | version |
| ------------- | ------- |
| keras         | 2.6.0   |
| keras-metrics | 1.1.0   |
| numpy         | 1.20.3  |
| pandas        | 2.0.0   |
| protobuf      | 3.20.0  |
| requests      | 2.28.2  |
| scikit-learn  | 1.2.2   |
| tensorflow    | 2.6.0   |
| torch         | 2.0.0   |

The environment configuration of the code is not the same. Please modify it according to the specific situation. The following is for reference only.

## Usage

The uploaded file name may contain extra symbols, please modify it accordingly. The comments in the code have been changed from Chinese to English.

### Process

- Construct patent heterogeneous data using the file in `"2_Network_Representation_Learning"`, and then use HeGAN, Metapath2Vec, and CompGCN models to obtain the patent representation under model training.
- Obtain patent chapter representations using the doc2vec, LSTM, Glove, SBERT, ESimCSE (+self attention) models from the file in `"3_Text_representation_learning"`
- By using the file in `"4_Training_of_Complementary_Patent_Recommendation_Model"`, complementary relationship annotation is performed on the original patent data, and a complementary patent dataset is constructed to obtain a patent complementary label matrix. Complementary patent recommendation models are trained using single and multiple features of the patent, respectively. Finally, the trained model is used to complete the recommendation.

## Evaluation Metric

In our research, we treat the evaluation as a **link prediction task**. Specifically, we train the complementary patent identification model and use the trained model to predict complementary relationships in the testing dataset. 

These predictions, together with the actual complementary relationships in the testing dataset, form the basis of complementary patent identification evaluation.

To quantitatively assess the effectiveness of the complementary patent identification method, we use several established metrics, including **accuracy, precision, recall, and F1 score**. 
