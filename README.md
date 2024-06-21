# **Identifying**-Complementary-Patents-Based-on-Deep-Learning

Note: The uploaded file name may contain extra symbols, please modify it accordingly

The comments in the code have been changed from Chinese to English



### 一.Data preprocessing

***1.Install PostgreSQL Database***

You can download PostgreSQL according to the tutorials on the following website: ( https://blog.csdn.net/feriman/article/details/119519772 )

Parameter settings:  The username is usually 'postgres', the password is set to 'root', the port is the default '5432', and the host is set to '127.0.0.1', the server is 'USPTO', and the default maintenance role is 'patient_manager'

Create a new database, role, and required table according to the statements in the file `"Database_creation_statement.txt"`

***2.Data Preparation***

Firstly, on the webpage ( https://bulkdata.uspto.gov/ ) download patent files in XML format from;

Secondly, the selected data is 'Patent Application Full Text Data (No Images)', which means the full text of the application data without images.

Finally, after downloading, extract it into XML and place it in a unified folder

***3.Upload data to the database***

Firstly, open line 96-109 in `"1_Data_processing\utils\db_interface.py"` to change the database settings

Secondly, after making the changes, run `"uspto_XMLparse.py"` on the terminal

### 二.`Network` representation learning

Partial use of Python libraries：

`dgl、pyyaml、pydantic、ordered_set、torch_scatter`

### 三.Text representation learning

Before running the code for Glove, it is necessary to pre download the "glove.840B.300d.txt" file from the web page ([GitHub - stanfordnlp/GloVe: Software in C and data files for the popular GloVe model for distributed word representations, a.k.a. word vectors or embeddings](https://github.com/stanfordnlp/GloVe/tree/master)) and place it in the specified location

Please place the `"patient_2022.csv"` file in the corresponding location of each file for the code under this chapter

### 四.Training of Complementary Patent Recommendation Model

***1.Construction of complementary patent datasets***

Annotate complementary relationships based on the file `"patient_2022. csv"`. Run the "Complementary Relationship Labeling. py" file to obtain the complementary relationship matrix, which is the `"Patent Complementary Label Matrix. csv"`.

***2.Model training based on single/multiple features***

Firstly, determine the patent representation required for model training, and modify line 222 under the premise of a single feature; In the case of multiple features, it is necessary to modify line 219 and line 220;
Finally, run the file "Nonlinear merging of CBAM+CNN with single/multiple features".

The training of this model is greatly affected by the environment configuration and the training speed is slow. It is recommended to train it in the moment pool cloud and install it according to the following version:

| Third party libraries | version |
| --------------------- | ------- |
| keras                 | 2.6.0   |
| keras-metrics         | 1.1.0   |
| numpy                 | 1.20.3  |
| pandas                | 2.0.0   |
| protobuf              | 3.20.0  |
| requests              | 2.28.2  |
| scikit-learn          | 1.2.2   |
| tensorflow            | 2.6.0   |
| torch                 | 2.0.0   |
| urllib3               | 1.26.15 |
