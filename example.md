UNIVERSITY OF HERTFORDSHIRE
School of Physics, Engineering, and Computer Science
7COM1039-0509-2023 - Advanced Computer Science Masters Project
A Study of Using Clustering Algorithms to Optimize
Marketing Strategies in African Banks Through Customer
Segmentation
Name:
Student ID:
Supervisor: Paul Moggridge
1
MSc Final Project Declaration
This report is submitted in partial fulfilment of the requirement for the degree of Master of Science
in Data Science and Analytics at the University of Hertfordshire.
It is solely my work except where indicated in the report. No human participation was used in the
project and permission is given to make this report available on the university website
2
ACKNOWLEDGMENT
3
ABSTRACT
In the competitive landscape of African banking, the ability to effectively target and engage customers
is essential for enhancing profitability and fostering customer loyalty. This study explores the
application of clustering algorithms to optimize marketing strategies within African banks through
customer segmentation. By analyzing large datasets using advanced data analytics techniques, the
research aims to identify distinct customer segments characterized by shared attributes and behaviors.
The study evaluates various clustering algorithms to determine the most effective methods for
customer segmentation, with the goals of improving marketing efficiency, enhancing customer
targeting, and promoting long-term loyalty.
K-means, K-prototype, Spectral, and DBSCAN clustering techniques were applied to a dataset
comprising 45,211 observations, 17 features. The clustering algorithms were assessed using
Silhouette, Calinski-Harabasz, and Davies-Bouldin evaluation metrics. Among the algorithms,
DBSCAN demonstrated superior performance, achieving a Silhouette score of 0.99, a Davies-
Bouldin score of 0.01, and a Calinski-Harabasz score of 1,163,469,880.96. Furthermore, an
investigation into the robustness of these clustering evaluation metrics revealed that the Silhouette
metric generalizes well across different scenarios, closely followed by the Davies-Bouldin metric
whereas, the Calinski-Harabasz index exhibits variability depending on the size of the clustered data.
These findings are expected to provide valuable insights for African banks on leveraging data-driven
approaches, offering practical recommendations for implementing clustering-based customer
segmentation. Such strategies are anticipated to refine marketing efforts, secure a competitive
advantage, and better address the diverse needs of their customer base.
4
TABLE OF CONTENT
MSc Final Project Declaration 1
Acknowledgements 2
Abstract 3
1.0 Introduction 8
1.1 Motivation 8
1.2 Problem Statement 9
1.3 Research Aim 9
1.4 Objectives 9
1.5 Research Question 10
1.6 Ethical, Legal, Professional, and Social Issues Consideration 10
1.6.1 Ethical Issues 10
1.6.2 Legal Issues 10
1.6.3 Professional Issues 10
1.6.4 Social Issues 10
1.6.5 Solution 10
1.7 University’s Ethics Approval 11
1.8 Report Structure 11
2.0 Literature Review 12
2.1 Clustering Techniques 12
2.2 Evaluation Metrics 14
2.3 Linkage to Aim 14
3.0 Methodology 15
3.1 Technical Tool 15
3.2 Data Collection 15
3.3 Data Description 16
3.4 Feature Selection 16
3.5 Exploratory Data Analysis 17
3.6 Data Preprocessing 19
3.6.1 Data Transformation 19
3.6.2 Handling Outliers 20
3.6.3 Data Standardization 21
3.6.4 Data Split 21
3.6.5 Cluster Size Determination 21
3.6.5.1 Elbow Method 21
3.6.5.2 Silhouette Method 23
3.6.5.3 Calinski-Harabasz Method 24
3.6.5.4 Davies-Bouldin Method 25
3.7 Hyperparameter Tuning 26
3.8 Clustering Implementation 26
3.8.1 K-Means Clustering 27
3.8.2 Kprototype Clustering 28
3.8.3 Spectral Clustering 29
5
3.8.4 DBSCAN Clustering 30
4.0 Results and Analysis 32
4.1 Clustering Algorithm Results 36
4.2 Result Comparison with Previous Research 39
4.3 Evaluation Metrics Robustness 40
4.4 Cluster Analysis 41
4.4.1 Cluster A 42
4.4.2 Cluster B 43
4.4.3 Cluster C 44
5.0 Discussion and Evaluation 46
5.1 Summary 46
5.2 Research Recommendation 46
5.3Critical Evaluation 47
5.4 Future Work 48
5.5 Commercial and Economic Considerations 48
5.5.1 Commercial Context 48
5.5.2 Economic Context 49
5.5 Project Management 49
5.6 Conclusion 50
6.0 Reference 52
7.0 Appendix 54
6
TABLE OF FIGURES
1.0 Fig 1.0 Flow chart of the research methodology 15
2.0 Fig 2.0: Histograms showing the Numerical feature distribution. 18
3.0 Fig 3.0: Bar charts showing the distribution of the categorical features. 18
4.0 Fig 4.0: Scatter Plot of the Age and Balance Features showing Outliers. 19
5.0 Fig 5.0 Code snippet for Elbow method Implementation. 22
6.0 Fig 6.0 Elbow method plot for Optimal Number of Clusters 22
7.0 Fig 7.0 Code snippet for Silhouette method Implementation. 23
8.0 Fig 8.0 Silhouette method plot for Optimal Number of Clusters 24
9.0 Fig 9.0 Code snippet for Calinski-harabasz method Implementation 24
10.0 Fig 10.0 Calinski-Harabasz plot for Optimal Number of Clusters 25
11.0 Fig 11.0 Code Snippet of Davies-Bouldin method Implementation 25
12.0 Fig 12.0 Davies-Bouldin plot for Optimal Number of Clusters 26
13.0 Fig 13.0 Code snippet for the implementation of K-means 28
14.0 Fig 14.0 3D graph of K-means clusters. 28
15.0 Fig 15.0 Code snippet for the implementation of K-prototype 29
16.0 Fig 16.0 Graph for K-prototype clusters 29
17.0 Fig 17.0 Code snippet for the implementation of Spectral clustering 30
18.0 Fig 18.0 Code snippet for DBSCAN implementation 31
19.0 Fig 19.0 DBSCAN Cluster 3D visualization 31
20.0 Fig 20.0 Figure showing the details of the final dataset. 32
21.0 Fig 21.0 Figure showing no missing value in the final dataset. 32
21.1 Fig 21.1 Partition A (60%) 33
21.2 Fig 21.2 Partition A (40%) 33
22.0 Fig 22.0 Scatter plot after preprocessing. 33
22.1 Fig 22.1 Scatter plot before preprocessing 34
22.2 Fig 22.2 Scatter plot after preprocessing 34
23.0 Fig 23.0 Histogram after normalization and outlier removal 34
23.1 Fig 23.1 Histogram Before Normalization 35
23.2 Fig 23.2 Histogram After Normalization 35
24.0 Fig 24.0 Bar chart of categorical features preprocessing 35
24.1 Fig 24.1 Bar Chart Before preprocessing 36
24.2 Fig 24.2 Bar Chart After preprocessing 36
25.0 Fig 25.0 Bar chart of categorical features after preprocessing 37
26.0 Fig 26.0 Silhouette Score Chart 37
27.0 Fig 27.0 Davies-Bouldin Score Chart 38
28.0 Fig 28.0 Calinski-Harabasz Score Chart 39
29.0 Fig 29.0 Silhouette score comparison between previous and current research 40
30.0 Fig 30.0 DBI score comparison between previous and current research 42
31.0 Fig 31.0 Heatmap of Cluster Zero 43
32.0 Fig 32.0 Heatmap of Cluster One 44
33.0 Fig 33.0 Heatmap of Cluster Two 45
7
TABLE OF TABLES
1.0 Table 1.0: Details of the full dataset 16
2.0 Table 2.0: Details of the selected features 17
3.0 Table 3.0: Numerical-to-Categorical Conversion 20
4.0 Table 4.0: Summary of the clustering algorithms to be implemented. 27
5.0 Table 5.0: Table of the score of the different clustering algorithms 36
6.0 Table 6.0: Comparison of previous and current research outcomes 39
7.0 Table 7.0: Table of the performance of the evaluation metrics. 41
8.0 Table 8.0: Table of cluster sizes 41
8
1.0 Introduction
In the highly competitive banking sector, particularly in the African context, understanding customer
behavior and preferences is crucial for developing effective marketing strategies. The banks strive to
enhance customer satisfaction, retain loyalty, and drive revenue growth however, marketing
approaches stemming from traditional customer segmentation always fall short of addressing the
diverse needs and characteristics of their customers. This often leads to a negative impact on
profitability, as increased budget spending is directed toward ineffective marketing campaigns
As described by Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020), customer segmentation
involves dividing a bank's customer base into distinct groups based on various attributes, such as
demographic information, transaction behavior, and product usage patterns. so that banks can develop
targeted marketing strategies that resonate with each group, ultimately leading to improved customer
engagement and business outcomes.
In agreement with Ganar and Hosein (2022), effectively segmenting customers into appropriate
groups has also become increasingly difficult but possible, especially in Africa with the increasing
financial inclusion and the proliferation of digital banking services which generate huge amounts of
data daily. This challenge necessitates the adoption of more sophisticated techniques for customer
segmentation hence, the clustering machine learning technique.
Clustering, according to Gupta, Kumar, Jain, Shrotriya, and Sinha (2022) is a concept in data science
and machine learning that involves grouping similar objects or data points to form a cluster. Objects
in the same group are more like each other than those in different groups. It is an unsupervised
machine-learning technique because it does not require predefined categories or labels. It rather finds
patterns from the input data based on some similarity measures. The main objective of clustering is
to find natural groupings in the data without knowing the group definitions.
Clustering algorithms, a key component of unsupervised machine learning, offer a powerful tool for
uncovering hidden patterns and natural groupings within large datasets. These algorithms can analyze
vast amounts of customer data and automatically identify segments with similar characteristics. When
applied effectively, clustering can reveal actionable insights that enable banks to optimize their
marketing strategies, enhance product offerings, and allocate resources more efficiently.
1.1 Motivation
The rapid growth and digital transformation of the banking sector in Africa present opportunities and
challenges for financial institutions seeking to stay competitive and relevant. As the continent’s
economies continue to expand, so does the complexity of the consumer landscape, characterized by
diverse demographics, varying financial needs, and increasingly sophisticated customer expectations.
In this context, traditional one-size-fits-all marketing approaches are becoming less effective,
highlighting the need for more personalized and data-driven strategies.
Customer segmentation is a powerful tool that enables banks to understand and cater to the unique
needs of different customer groups Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020). However,
the effectiveness of segmentation depends heavily on the methodologies employed to identify these
groups. Clustering algorithms, a core component of data science, offer a robust solution to this
9
challenge by analyzing large datasets to uncover hidden patterns and groupings that might not be
apparent through conventional analysis.
The motivation for this study stems from the pressing need to optimize marketing strategies in African
banks through more precise customer segmentation. By leveraging clustering algorithms, banks
cannot only enhance their understanding of diverse customer bases but also tailor their products,
services, and marketing efforts more effectively. This, in turn, can lead to increased customer
satisfaction, loyalty, and ultimately, profitability.
Moreover, as African banks face increasing competition from both traditional financial institutions
and emerging fintech companies, the ability to innovate and adapt through advanced analytics
becomes crucial. This study seeks to contribute to the banking sector's evolution by demonstrating
how clustering algorithms can be utilized to drive more effective and efficient marketing strategies,
thus supporting the broader goals of financial inclusion and economic development across the
continent.
In essence, this research is motivated by the potential to harness the power of data science to address
the unique challenges of the African banking sector, transforming how banks engage with their
customers and paving the way for more personalized, impactful, and sustainable marketing practices.
1.2 Problem Statement
African banks need more sophisticated methods to accurately segment their customers for
personalized marketing campaigns. Traditional segmentation methods often fail to capture the
complexities and dynamics of customer behaviours and preferences leading to inefficiencies in
marketing efforts and suboptimal customer engagement. Therefore, there is a need for advanced
analytical techniques to provide more precise and actionable customer insights.
1.3 Research Aim
This study aims to investigate and recommend the most effective clustering methods for customer
segmentation in African Bank to enhance customer targeting, boost marketing efficiency, and
strengthen customer loyalty. The research also investigates the stability and robustness of different
evaluation metrics used for measuring the performance of the clustering methods.
1.4 Objectives
The critical tasks that would be completed to achieve these aims are listed in the bullet points below.
• To get suitable datasets that contain customer demographic, transactional, and behavioral data
from an African bank.
• To review relevant pieces of literature and identify the gaps.
• To evaluate different clustering algorithms and select the appropriate ones for the dataset.
• To preprocess the data in preparation for clustering.
10
• To design and implement the selected clustering techniques on the Bank’s customer database.
• To optimize the clustering algorithm parameters to achieve the best results.
• To evaluate the performance of the different clustering algorithms.
• To recommend the most suitable clustering algorithm in the circumstances.
• To investigate the robustness of the clustering evaluation metrics.
1.5 Research Questions
This piece of work intends to address the following questions.
• Which clustering algorithm (e.g. K-Means, K-Prototype, Spectra, Hierarchical
Clustering, DBSCAN) is most effective for segmenting customers in the African banking
context? This question explores the performance of different clustering algorithms in
accurately segmenting banking customers and recommends the most effective one.
• Which intrinsic metric is robust enough to measure the performance of clustering
algorithms? This question investigates the intrinsic methods for measuring the performance
of clusters, such as the Davies Bouldin index, Silhouette Score, and Calinski Harabasz, to
comment on their robustness or otherwise.
• What are the best practices and recommendations for African banks to successfully
implement and leverage clustering algorithms for optimizing marketing strategies? This
question aims to provide actionable insights and guidelines for African banks on effectively
using clustering algorithms to enhance their marketing strategies and achieve better business
outcomes.
1.6 Ethical, Legal, Professional, and Social Issues Consideration
This research involves using advanced data analytics to enhance marketing strategies in the banking
sector. While this project has the potential to improve business outcomes and customer satisfaction
significantly, it also raises several ethical, legal, professional, and social concerns that must be
carefully considered and addressed.
1.6.1 Ethical Issues: The ethical issues that may arise from this project include privacy and
confidentiality issues where the use of customer data requires access to potentially sensitive
information. Informed consent where customers may not be aware that their data is being used for
such a project is another concern.
1.6.2 Legal Issues: Compliance with data protection laws such as GDPR (General Data Protection
Regulation) and local African data protection regulations is mandatory. Additionally, the use of
proprietary algorithms, programming language, development environment, and data may raise
intellectual property issues.
1.6.3 Professional Issues: The accuracy and integrity of the data fed into the clustering algorithms
are crucial to guarantee effective customer segmentation and actionable outcomes.
1.6.4 Social Issues: Misuse of customer data or perceived invasions of privacy could erode
customers’ trust in the bank and affect its reputation.
11
1.6.5 Solutions: Personal Identifiable Information has been removed from the dataset used in this
research. That is, the observations cannot be traced to any customer. Equally, the dataset is available
for public use on the UCI machine learning repository and has been used by Pandey, Amrutha, Dash,
and Patra (2023) and other researchers.
The programming language utilized for this research, Python, is both free and open source. Similarly,
PyCharm Community Edition, the integrated development environment employed for Python
development, is also available at no cost. Before inputting the dataset into the clustering algorithms,
industry-standard data preprocessing techniques were carried out to ensure the integrity of the
clustering outcome.
1.7 University’s Ethics Approval
It is required to seek the University’s approval if a study involves human participants. The objective
is to ensure that studies follow established ethical standards and protect participants from physical
and emotional harm.
However, I assert that the intrinsic ethical, legal, professional, and social considerations inherent in
this research have been comprehensively considered and addressed through the measures highlighted
above. As stated earlier, the dataset used for this research is available to the public and from a
reputable source. Consequently, obtaining the university’s ethics approval is deemed unnecessary.
1.8 Report Structure
This report is structured as follows: Chapter Two provides a review of related literature and prior
research. Chapter Three details the methodology and tasks undertaken in this study. Chapter Four
presents an analysis of the results obtained. Finally, Chapter Five offers conclusions and
recommendations based on the findings followed by References and Appendix.
12
2.0 LITERATURE REVIEW
Researchers employ various clustering techniques like graph-based, partition-based, density-based,
model-based, etc. for customer segmentation, with the choice of algorithm largely influenced by the
characteristics of the datasets and the specific objectives of the analysis Umuhoza, Ntirushwamaboko,
Awuah, and Birir (2020). Among these techniques, partition-based algorithms such as K-means,
along with its variants like K-modes, K-medoids, and K-prototypes, are very popular and the most
utilized.
2.1 Clustering Techniques
Pandey, Amrutha, Dash, and Patra (2023) applied K-modes—a variant of K-means designed for
categorical data—alongside the Gaussian Mixture Model and Hierarchical Agglomerative Clustering
to segment a bank’s customer database. Their study, which involved the same dataset as used in this
research, concluded that K-modes was the most effective clustering algorithm, achieving a Davies-
Bouldin score of 4.82 and a silhouette score of 0.03.
Conversely, Gupta, Kumar, Jain, Shrotriya, and Sinha (2022) advocated for K-prototype, another Kmeans
variant that can handle both numerical and categorical data types, as a more suitable algorithm
for segmenting a bank’s customer database. In their study titled “Demographic Customer
Segmentation of Banking Users Based on K-prototype Methodology”, they used the elbow method
to determine the optimal number of clusters and concluded that Kprototype performed very well even
though they did not provide any specific score to justify this position.
Sharma and Desai (2023) compared K-means and Louvain algorithms to support data-driven
decisions for business optimization and concluded that the Louvain algorithm outperformed K-means
in segmenting customers with minimal outliers. They opined that specifying the number of clusters
for any clustering algorithm beforehand may not yield an optimal performance and as such represents
a drawback of the K-means technique. According to them, the Louvain algorithm was developed by
Blondel, Guillaume, Lambiotte, and Lefebvre (2008) to address this concern. The algorithm works
by finding high modularity partition in a network and accurately shows the relationships between
nodes. They used the silhouette method to find the optimal number of cluster for K-means.
To enhance business service offerings and increase sales, Gankidi, Gundu, viqar Ahmed, Tanzeela,
rasad, and Yalabaka (2022) developed a real-time customer segmentation web application using the
K-means clustering algorithm. The study utilized a Kaggle dataset comprising demographic and
transactional data of approximately 8,068 customers. After data preprocessing, the K-means model
was built, with the optimal number of clusters generate by Elbow method, and used to segment both
new and unseen customers. They indicated that K-means was effective and computationally efficient.
In their research on customer segmentation for credit card users in an African financial institution,
Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020) also utilized the K-means clustering
algorithm. The researchers suggested that while the Gaussian Mixture Model and feed-forward neural
networks are known for handling complex and arbitrary data, they may inadvertently capture noise
rather than true patterns. They hinged their choice of K-means on its widespread deployment,
simplicity and efficiency, particularly with large datasets. In addition to Elbow and Silhouette
methods, they used Davies-Bouldin to determine the optimal number of clusters.
13
Aryuni, Madyatmadja, and Miranda (2018) applied K-means and K-medoids clustering algorithms to
segment customers of XYZ Bank based on their use of internet banking. K-medoids, is a variant of
K-means which uses an actual data point as the centroid rather than the mean of the data points in a
cluster in the case of K-means. Using intra cluster distance and Davies-bouldin index to determine
the optimal number of clusters, their results indicated that K-means performed better than K-medoids
on the two performance metrics used.
To identify customer requirements and preferences, Bhimarapu, Bhelkar, Chavhan, Dhule, and
Agrawal (2023) utilized K-means clustering and a decision tree. Their analysis concluded that Kmeans’
performance is superior to decision tree algorithm. In a related study on churn prediction
based on customer segmentation in the banking industry, Sai, Kumari, and Gupta (2024) combined
K-means for customer segmentation with prediction models such as decision tree, logistic regression,
and random forest classification model.
Maddumala, Chaikam, Velanati, Ponnaganti, and Enuguri (2022) echoed the consensus among many
researchers that K-means is the most efficient and straightforward algorithm for customer
segmentation. They used the Elbow method to determine the optimal number of clusters and applied
K-means to a customer database, which resulted in five distinct groups. This position was
corroborated by Akbar, Liu, and Latif (2020) in their study to support organizations in target
marketing.
Conversely, Nagaraj, Kumar, Kumar, Venu, Sekar, and Rajkumar (2023) investigated the
performance of K-means and Naïve Bayes for customer segmentation, concluding that Naïve Bayes
outperformed K-means. However, they agreed that the effectiveness of clustering algorithms depends
heavily on data structure and proper data preprocessing.
Deng and Gao (2020) argued that the limitations of K-means necessitated its combination with
Affinity Propagation for e-commerce customer segmentation. Despite being more computationally
expensive, they found that this combination outperformed K-means and Affinity Propagation when
applied to the Iris and Ionosphere datasets.
Similarly, Li and Lee (2024) explored the integration of Support Vector Machines (SVM) with Kmeans
for customer segmentation, facilitating enterprises in providing precise and timely solutions to
customer needs amid the proliferation of the internet and big data. They argued that using SVM for
dimensionality reduction before applying K-means led to credible and robust segmentation outcomes.
Finally, Dinavahi, Thatavarti, Rangala, Vallamsetti, and Nannuri, (2023) investigated seven
clustering techniques spanning partition-based, density-based, graph-based, and model-based
algorithms for retail customer segmentation using the transaction records of the online retail store
dataset within two years including the RFM (Recency Frequency Monetary) values. They concluded
that K-means and spectral clustering techniques were the most suitable for this purpose.
In conclusion, the partition-based unsupervised machine learning technique is extensively employed
by researchers. However, there is a consensus that the underlying data structure and the specific
objectives of the segmentation significantly influence the selection of an appropriate clustering
technique.
14
2.2 Evaluation metrics
The reviewed literature underscores the importance of selecting an appropriate clustering model for
segmenting a customer dataset, as this choice is crucial to achieving successful outcomes. Equally
important is the evaluation of model performance, for which various methods have been employed
by researchers. Among the most widely used intrinsic performance evaluation metrics—particularly
when the ground truth is unknown—are the Davies-Bouldin Index, Calinski-Harabasz Index, and
Silhouette score. These metrics assess the coherence and separation of data points within the
generated clusters.
As noted by Dinavahi, Thatavarti, Rangala, Vallamsetti, and Nannuri (2023), the Davies-Bouldin
Index measures the average similarity of clusters, with lower scores indicating better segmentation.
In contrast, the Calinski-Harabasz Index evaluates the ratio of intra-cluster to inter-cluster dispersion,
where higher scores denote more effective partitioning while the Silhouette Score, which is the
similarity of a data point to the cluster it has been assigned compared to other clusters, ranges from -
1 to 1. The closer to 1 the score is the better the partition.
2.3 Linkage to Aim
In summary, the exploration of various clustering techniques for customer segmentation underscores
the complexity and diversity of approaches available for optimizing marketing strategies. The studies
reviewed highlight that while K-means and its variants, such as K-modes and K-prototypes, are
commonly favoured due to their simplicity and efficiency, there is no one-size-fits-all solution as
correctly put by Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020).
Different algorithms, including Gaussian Mixture Models, Hierarchical Agglomerative Clustering,
DBSCAN, and Spectral clustering, offer varying advantages depending on the specific dataset
characteristics and segmentation goals. Furthermore, advancements in combining clustering
techniques with other methods, such as Affinity Propagation or Support Vector Machines,
demonstrate the potential for improved outcomes in customer segmentation. However, no research
has investigated the most effective clustering algorithm for customer segmentation in African banks.
In addition, the review suggests that the choice of algorithm should be tailored to the data structure
and business objectives, emphasizing the importance of thorough evaluation and comparison using
multiple metrics but there is lack of a research exploring the stability and robustness of these metrics.
This study aims to close these gaps by investigating the most efficient clustering technique for
customer segmentation in African Bank landscape while also determining the robustness or otherwise
of the clustering performance evaluation metrics.

16
3.3 Data Description
After the data was downloaded, it was saved as a CSV file and read by the Pandas library in Python.
The dataset has 45,211 observations,16 features, and 1 target. Details of these are shown in the table
below.
S/
N
Feature
Name
Type Category Description
1 age numeric demographic age of the customer (integer)
2 job categorical demographic
occupation (non-binary: 'admin.', 'bluecollar',
'entrepreneur', 'housemaid',
'management', 'retired',' selfemployed','
services','student','technician','un
employed','unknown')
3 marital categorical demographic
marital status (non-binary: 'divorced',
'married', 'single'; note: 'divorced' means
divorced or widowed)
4 education categorical demographic
educational level (non-binary: 'tertiary',
'secondary', 'unknown', 'primary')
5 default categorical bank has credit in default?(binary: 'no', 'yes')
6 balance numeric bank average yearly balance(integer)
7 housing categorical bank has a housing loan?(binary: 'no', 'yes')
8 loan categorical bank has a personal loan?(binary: 'no', 'yes')
9 contact categorical other
communication
type(binary:’cellular’,’telephone’)
10
day_of_w
eek
date other
last contact day of the week(non-binary:
‘Mon’,’Tue’…’Sun’)
11 month date other
last contact month of the year(nonbinary:’
jan’,’feb’,…’dec’)
12 duration numeric other last contact duration in seconds
13 campaign numeric bank
number of contact performed during the
campaign
14 pdays numeric other
number of days that passed by after client
was contacted
15 previous numeric other
number of contacts performed before the
campaign
16 poutcome categorical bank
outcome of the previous marketing
campaign(nonbinary:’
failure’,’nonexistent’,’success’)
17 target categorical bank
has the client subscribed to term
deposit(binary:’no’,’yes’)
Table 1.0 Details of the full dataset
3.4. Feature Selection
This was the first preprocessing step carried out. 8 features were found useful and relevant to the
research out of the 17 features because of my domain knowledge of Banking. 4 out of these 8 are
demographic data while the other 4 are bank transaction data. The other features were dropped and a
17
new dataset containing the selected features was created. consequently, we have a dataset with 45,211
observations and 8 features. Details of the selected features are provided in the table below.
S/
N
Feature
Name
Type Category Description
1 age numeric demographic age of the customer (integer)
2 job categorical demographic
occupation (non-binary: 'admin.', 'bluecollar',
'entrepreneur', 'housemaid',
'management', 'retired',' selfemployed','
services','student','technician','un
employed','unknown')
3 marital categorical demographic
marital status (non-binary: 'divorced',
'married', 'single'; note: 'divorced' means
divorced or widowed)
4 education categorical demographic
educational level (non-binary: 'tertiary',
'secondary', 'unknown', 'primary')
5 default categorical bank has credit in default?(binary: 'no', 'yes')
6 balance numeric bank average yearly balance(integer)
7 housing categorical bank has a housing loan?(binary: 'no', 'yes')
8 loan categorical bank has a personal loan?(binary: 'no', 'yes')
Table 2.0: Details of the selected features
A Random Forest Classifier was implemented to prune down the number of features further. However,
the result indicated that no significant difference would be made if any of the selected features is
removed. Please find the snapshot of the snippet and result in the Appendix. However, Principal
Component Analysis (PCA), a dimensionality reduction technique that transforms the original
features into a set of linearly uncorrelated components, was carried out on the pre-processed dataset
before implementing the clustering techniques.
Implementing PCA before clustering can be beneficial in reducing dimensionality, improving
computational efficiency, and enhancing visualization. However, it can also introduce challenges such
as loss of interpretability, potential distortion of clusters, and loss of important information. The
decision to apply PCA was based on the high dimensionality of the dataset which has a huge impact
on the computational efficiency of the algorithms. Having obtained the optimum number of clusters
for the dataset, any loss of information due to PCA will not be critical for this task hence, its
application before clustering.
3.5 Exploratory Data Analysis
This is a key process in data analysis used to understand the characteristics, structure, and
relationships within a dataset. It involves summarizing data through statistics, visualizing it with
various plots, and identifying patterns, outliers, or anomalies. EDA also includes data cleaning,
checking data distributions, and exploring relationships between variables. This iterative process
provides insights that guide further analysis, model selection, and feature engineering, making it an
essential step in any data-driven project.
The new dataset came out clean after checking for missing values and datatype mismatches. Data
visualization was done to understand the distribution and connection between the features while also
18
checking for outliers. Pandey, SV, Amrutha, Dash, and Patra (2023) said this also helps to identify
any obvious variance, correlations, skewness, or dispersion.
The figures below are some of the plots generated during this phase.
Figure 2.0: Histograms showing the “age” and “balance” feature distribution.
Figure 3.0: Bar charts showing the distribution of the categorical features.
19
Figure 4.0: Scatter Plot of the Age and Balance Features showing Outliers.
3.6 Data Preprocessing
The integrity of any machine learning model is as good as the accuracy, cleanliness, correctness, and
completeness of the training data used to build the model as rightly opined by Ahmed, Ahmed,
Ahmed, Latha, and Kumar (2024). Therefore, adequate care and attention must be placed on preparing
the data for the modeling task.
3.6.1 Data Transformation
Data transformation is a key data preprocessing step in machine learning, especially in unsupervised
learning techniques like clustering, that involves modifying the data to improve clustering
performance. It ensures that the data is appropriately scaled, normalized, and transformed to make
the clustering process more effective and the results more meaningful.
Except for Kptotoype which has an internal mechanism to handle categorical data before performing
clustering, the other clustering algorithms investigated in this study rely on numerical distance metrics
(like Euclidean, Manhattan, etc.) to measure similarity or dissimilarity between data points.
Therefore, encoding categorical features is a crucial preprocessing step. Ordinal encoding was carried
out on the non-binary categorical features (‘education’, ‘marital’, and ‘job’) while one-hot encoding
was used for the binary categorical features (‘housing’, ‘loan’ ‘default’).
The following data transformation was further carried out to reduce the complexity of the data for
clustering efficiency and better performance, particularly for Kprototype because executing the
algorithm with this complex dataset was too computationally expensive. The following attributes of
some features were simplified to reduce the data complexity.
20
1. The 11 unique attributes of the ‘job’ feature are reduced to 2 (‘employed’ and ‘unemployed’).
The observations with ‘retired’, ‘student’, ’unknown’, and ‘unemployed’ are categorized as
‘unemployed’ while the remaining are categorized as ‘employed’.
2. The observations with the ‘divorced’ attribute for the ‘marital’ feature are converted to
‘single’ reducing the unique attribute of the marital feature to 2 (‘married’ and ‘single’)
3. The ‘primary’ attribute was assigned to the observations with ‘unkown’ in the ‘education’
feature leaving it with 3 unique attributes (‘primary’, ‘secondary’, and ‘tertiary’)
As a result of this exercise, only ‘education’ remained as a non-binary categorical feature while
‘marital’ and ‘job’ are now binary categorical features.
For visualization purposes, the numerical variables ‘age’ and ‘balance’ were converted to categorical
variables. While ‘age’ was cut into three, balance was cut into quartiles. This new dataset was also
used to implement Kmode used in the previous research. The table below shows the value used for
the conversion.
Table 3.0. Numerical -to- Categorical Conversion
3.6.2 Handling Outliers
Outliers are data points that deviate significantly from other observations in the dataset. They can
occur due to various reasons, including variability in the data, measurement errors, data entry errors,
or they might represent genuinely rare events or phenomena. Therefore, careful analysis is required
to decide how to handle them, as they can both provide valuable insights and distort the overall
analysis Sudirman, Utama, Bahri, and Susanto (2023), especially in a clustering task such as this.
Some outliers were conspicuous during the data visualization, particularly in the ‘balance’ and ‘age’
features. Since it is difficult to further investigate these outliers because of the data source, the Zscore,
the method that standardizes the data by converting the features to a distribution with a mean
of 0 and a standard deviation of 1 was used to handle the outliers in this dataset to be on a safer side.
As a rule of thumb, data points with a Z-score greater than 3 or -3 are considered outliers. This reduced
the dataset to 44,094 observations. The formula of the Z-score method is given below.
𝑍 = (𝑋 − 𝜇)/𝜎
Z is the Z-score, X is the data point, μ is the mean, and σ is the data set's standard deviation. Ezenkwu,
Ozuomba, and Kalu (2015). A ‘Z score’ of 0 indicates the data point is exactly the mean of the dataset,
a positive Z score means the value of the data point is more than the mean of the dataset while a
negative Z score indicates the data point is below the mean.
Feature Numerical Value Categorical Value
17-35 Young Adult
35-55 Adult
55-72 Senior
1st quartile Low
2nd quartile Lowe-Middle
3rd quartile Upper-Middle
4th quartile High
Age
Balance
21
3.6.3 Data Standardization
The numerical features (‘age’ and ‘balance’) were normalized with Min-Max standardization so that
they are on the same scale. This is so that these features (which are on different scales) contribute
equally to the model and improve the performance of the clustering algorithms. This standardization
method is informed by the non-normal distribution of the data observed after visualization.
A copy of this dataset was made to apply the next preprocessing method which is to encode the
categorical variables. This is because some clustering algorithms like K-means require the categorical
features to be encoded while others like Kprototype have internal capabilities to handle them as they
are.
3.6.4 Data Split
It is common in Machine Learning to split datasets into two sets so that the model can be trained on
one set while it is tested on the other. This would not have been necessary for this study as it is an
unsupervised machine learning where ground truth is not known. However, it was done to test the
robustness of the clustering evaluation metrics, the pre-processed data was split into two before
implementing the clustering algorithms on each of them. The Davies-Bouldin, Silhouette, and
Calinski-Harabasz were then implemented on the cluster generated by the algorithms. The metric that
outputs a consistent score is adjudged robust while the one with inconsistent output is deemed
unstable.
3.6.5 Cluster Size Determination
One of the key parameters that must be indicated particularly in the partition-based clustering
algorithms is the number of clusters to partition the given dataset into. The Elbow and Silhouette are
the most used methods of determining the optimal number of clusters in the literature reviewed. In
addition to these two, Calinski-Harabasz and Davies-Bouldin methods were used for validation.
3.6.5.1 Elbow Method
In line with the position of Bhimarapu, Bhelkar, Chavhan, Dhule, and Agrawal (2023), the Elbow
method is suitable for determining the optimal number of clusters for a clustering model in that it
uses the Within-Cluster Sum of Squares (WCSS) which is the sum of squared distance between a data
point and the centroid of its assigned cluster to determine the compactness of the cluster. A lower
WCSS value indicates that the data points within the cluster are tightly clustered around the centroid,
and a higher value indicates that the data points are spread out from the centroid suggesting that the
cluster is not well formed. The formula and explanation for this are given below.
WCSS = Σ Σ |
𝑥𝑖∈𝐶𝑘
𝑥𝑖
𝐾
𝑘=1
− 𝜇𝑘 |2
𝐾 is the number of clusters, 𝑥𝑖 is a data point in the cluster 𝐶𝑘 , 𝑎𝑛𝑑 𝜇𝑘 is the centroid (mean) of the
cluster 𝐶𝑘 .
22
1. For each cluster 𝐶𝑘 , the centroid 𝜇𝑘 is computed as the mean of all the data points in that
cluster.
2. For each data point 𝑥𝑖 in a cluster, the squared distance from the point to the cluster centroid
|𝑥𝑖 − 𝜇𝑘 |2 is computed.
3. All these squared distances are summed up for each cluster, and then these sums are added
across all clusters to obtain the total WCSS.
As the number of clusters increases, the WCSS usually decreases. The cluster where an increase in
the number of clusters does not translate to a significant decrease in the WCSS is then selected as the
optimal number of clusters for the dataset. This point looks like the elbow shape when plotted hence,
the name, Elbow method. Maddumala, Chaikam, Velanati, Ponnaganti, and Enuguri (2022).
As shown in the elbow graph below, the optimal number of clusters for the dataset used in this
research is 3.
Figure 5.0 Code snippet for Elbow method implementation for Optimal Number of Cluster
Figure 6.0 Elbow method plot for Optimal Number of Clusters
23
3.6.5.2 Silhouette Method
In contrast, Dinavahi, Thatavarti, Rangala, Vallamsetti, Nannuri (2023) believed that the elbow
method is not as stable as the silhouette method for determining the optimal number of clusters for a
dataset. As described by Hossain, Sebestyen, Mayank, Ardakanian, and Khazaei (2020), the
Silhouette score measures the similarity of a data point to the cluster it has been assigned compared
to other clusters. The score ranges from -1 to 1. A score of 1 suggests correct clustering while a score
of -1 indicates incorrect clustering and 0 signifies an overlapping clustering according to Dinavahi,
Thatavarti, Rangala, Vallamsetti, Nannuri (2023). The closer the silhouette score is to 1 the better.
The code and graph of the silhouette method are shown in the figures below.
For each data point ‘i’ in the dataset, the average distance between ‘i’ and all other points in the same
cluster is calculated as say d(i), this measures how close ‘I’ is to other points within the cluster
(cohesion). The average distance between ‘i’ and all other points in the nearest neighboring clusters
is also calculated as say e(i). This measures how far ‘i’ is from the points in the nearest clusters
(separation). The silhouette score of each point is then calculated as
𝑠(𝑖) = (𝑒(𝑖) − 𝑑(𝑖))/𝑚𝑎𝑥(𝑎(𝑖), 𝑏(𝑖))
Where ‘n’ is the number of data points, the silhouette scores for all points are averaged to give an
overall silhouette score for the clustering configuration with the formula as.
𝑆 = 1/𝑛 Σ𝑠(𝑖)
𝑛
𝑖=1
Figure 7.0 Code snippet for Silhouette method implementation for Optimal Number of Clusters.
24
Figure 8.0 Silhouette method plot for Optimal Number of Clusters
3.6.5.3 Calinski-Harabasz
This measures the ratio of the sum of between-clusters dispersion and inter-cluster dispersion for all
clusters. Higher values indicate better-defined clusters. The values are plotted against the cluster sizes
and the cluster with the highest value represents the optimal cluster. Like Umuhoza,
Ntirushwamaboko, Awuah, and Birir (2020), this was implemented to validate the elbow and
silhouette methods. Below is the snapshot of the artifact and the plot for calinski-harahasz.
Figure 9.0 Code snippet for Calinski-harabasz method for Optimal Number of Clusters
25
Figure 10.0 Calinski-harabasz plot for Optimal Number of Clusters
3.6.5.4 Davies-Bouldin
This method of estimating the optimal number of clusters evaluates the average similarity ratio of
each cluster with the cluster that is most similar to it. The number whose value is close to zero is
selected as the optimum number of clusters for the dataset. Dinavahi, Thatavarti, Rangala,
Vallamsetti, Nannuri (2023)
Figure 11.0 Code Snippet of Davies-Bouldin method for Optimal Number of Clusters
26
Figure 12.0 Davies-Bouldin plot for Optimal Number of Clusters
3.7 Hyperparameter Tuning
Gupta, Kumar, Jain, Shrotriya, and Sinha (2022) described hyperparameters as values that define the
model architecture, and searching for the values that provide a model’s best result is referred to as
hyperparameter tuning. For the partition-based algorithms like K-means where the key parameter to
be tuned is “K” that is, the number of clusters, the methods described above were used to determine
the optimal value. However, grid search is used to tune the hyperparameters for DBSCAN (‘eps’ and
‘min-samples), and Spectral (‘affinity’, ‘gamma’, ‘eigen solver’, ‘n-neighbors’, etc.).
3.8 Clustering Implementation
The plan is to implement K-means, K-prototype, Spectra clustering, DBSCAN, Hierarchical
Clustering techniques, and conduct a comparative analysis of their results vis-à-vis the results of
previous research with the same dataset. According to Gupta, Kumar, Jain, Shrotriya, and Sinha
(2022), K-means is the most deployed clustering algorithm known for its simplicity and effectiveness
in customer segmentation and handling large datasets while K-prototype is a partition-based
clustering algorithm that combines the capabilities of K-means and K-modes with its ability to handle
data both numerical and categorical data types.
Spectra clustering, DBSCAN, and Hierarchical clustering are considered for implementation because
of the multivariate and dense nature of the dataset. These algorithms are robust to noise, and they are
adjudged to handle complex relationships and multiple features better. Ahmed, Ahmed, Ahmed,
Latha, and Kumar (2024) correctly state that DBSCAN is particularly effective in detecting clusters
of arbitrary shape and handling noise.
The table below summarises these clustering algorithms.
27
Algorithm Key Strengths Key Weaknesses Best Use Cases
K-Means Simple, efficient for
large datasets
Assumes spherical clusters,
sensitive to outliers
Large datasets with wellseparated,
spherical
clusters
Hierarchical No need for
predefined K, visual
hierarchy
Computationally expensive Smaller datasets,
hierarchical relationships
DBSCAN Arbitrary shapes,
robust to noise
Sensitive to parameters,
varying density issues
Irregularly shaped clusters,
outlier detection
Spectral Captures complex
relationships
Computationally intensive,
less efficient for large data
Non-linear relationships,
complex cluster shapes
K-Prototypes Handles mixed data
types
Requires specifying K,
sensitive to initial centers
Mixed numerical and
categorical data
Table 4.0: Summary of the clustering algorithms to be implemented.
3.8.1 K-Means Clustering
K-Means clustering is a widely acknowledged and extensively utilized unsupervised machine
learning algorithm for customer segmentation. According to Umuhoza, Ntirushwamaboko, Awuah,
and Birir (2020), its popularity is attributed to its simplicity, speed, and capacity to handle large
datasets efficiently. The fundamental objective of K-Means is to partition a dataset into K distinct,
non-overlapping clusters, where each data point is assigned to the cluster with the closest mean. This
algorithm is particularly effective in identifying underlying patterns or groupings within a dataset,
even in the absence of prior knowledge of the labels. The following steps represent the K-means
algorithm.
STEP 1 - Initialisation: Select K number of clusters (3 in this case as suggested by the elbow
and silhouette methods above) and initialize K centroids (the center of a cluster that is
calculated as the mean of all the data points in the cluster) with K-means++
STEP 2 - Assignment: Each data point is assigned to the nearest centroid based on Euclidean
distance metric. This forms K clusters, where each data point belongs to the cluster with the
closest centroid.
STEP 3 - Update: The K centroids are recalculated by computing the mean of all data points
assigned to each cluster. The centroid is now the average position of all points in the cluster.
STEP 4 - Iteration: Steps 2 and 3 are repeated until no significant changes in the centroids
or the assignment of data points to clusters no longer occur.
STEP 5 – Convergence: The algorithm stops with the final output of K clusters
28
Figure 13.0 Code snippet for the implementation of K-means
Figure 14.0 3D graph of K-means clusters.
3.8.2 K-Prototypes Clustering
K-Prototypes is an extension of clustering algorithms like K-Means and K-Modes, designed to handle
datasets containing numerical and categorical variables. Traditional clustering algorithms like KMeans
work well with numerical data, while K-Modes is more suitable for categorical data. However,
many real-world datasets like the one used in this study contain a mix of numerical and categorical
features, and this is where K-Prototypes becomes essential Gupta, Kumar, Jain, Shrotriya, and Sinha
(2022). The algorithm is explained in the steps below.
STEP 1 - Initialisation: Select K number of clusters (3 in this case as initially suggested by the elbow
and silhouette methods) and initialize K prototypes (centroids combining the mean of numerical
features and mode of the categorical features) with “K-prototypes++” or “cao”
STEP 2 - Assignment: Assign each data point to the nearest prototype based on the combined
distance metric. The metric considers both the numerical and categorical components.
STEP 3 - Update: The prototypes are updated by recalculating the mean of the numerical features
and the mode of the categorical features for each cluster.
STEP 4 - Iteration: Steps 2 and 3 are repeated until the prototypes no longer change significantly,
or the assignments of data points to clusters become stable.
29
STEP 5 – Convergence: The algorithm converges with the final output of K clusters that account for
numerical and categorical data.
Figure 15.0 Code snippet for the implementation of K-prototype
Figure 16.0 Graph for K-prototype clusters
3.8.3 Spectral Clustering
Spectral clustering methods use eigenvalues of a similarity matrix to reduce dimensions before
clustering in fewer dimensions. It converts the data into a graph and uses the spectrum (eigenvalues)
of the graph Laplacian for dimensionality reduction before applying clustering Dinavahi, Thatavarti,
Rangala, Vallamsetti, and Nannuri (2023). Below are the Spectral clustering steps and the Python
code snippet to achieve it.
STEP 1: Construct a similarity graph/matrix (this can either be done using K-Nearest neighbor or
Gaussian RBF
STEP 2: Form a Laplacian Matrix (which is the Diagonal Matrix minus the Similarity Matrix)
STEP 3: Compute the first k (the number of clusters you want to form) eigenvectors of the Laplacian
matrix
30
STEP 4: Cluster the data on step 3
STEP 5: Assign each data point to a cluster.
Figure 17.0 Code snippet for the implementation of Spectral clustering
3.8.4 DBSCAN Clustering
DBSCAN an acronym for Density-Based Spatial Clustering of Applications with Noise defines
clusters based on the density of data, identifying regions of high density separated by regions of low
density. The Points in high-density regions are grouped, while points in low-density regions are
marked as noise (outliers).
DBSCAN is effective for discovering clusters of arbitrary shape and handling noise but requires
careful tuning of parameters (‘eps’ and ‘min-samples’) to get the best outcomes. ‘eps’ is the minimum
distance between two data points while ‘min-samples’ is the minimum number of data points within
a certain distance (‘eps’) Dinavahi, Thatavarti, Rangala, Vallamsetti and Nannuri (2023).
After implementing a grid search with ‘eps’ from 0.1 to 1.0, and a step of 0.1, ‘min-samples’ from 2
to 10, 0.6 was returned as the best value for ‘eps’ and 2 for ‘min-samples’. Hence, these values were
used with the preprocessed dataset. Please find below the algorithm and the Python code snippet
STEP 1: Initialize Parameters (‘eps’ and ‘min-samples’).
STEP 2: Classify Points as core, border, or noise based on step 1.
STEP 3: Form Clusters by expanding core points’ neighborhoods.
STEP 4: Label Noise points that are not part of any cluster.
STEP 5: End when all points have been processed.
31
Figure 18.0 Code snippet for the implementation of DBSCAN
Fig 19.0 DBSCAN Cluster 3D visualization
32
4.0 RESULTS AND ANALYSIS
This chapter presents the outcome of this research for analysis starting with the outcomes of the data
preprocessing and the performance of the various clustering techniques. The outcome of the data
preprocessing is shown the figures below.
Fig 20.0 Figure showing the details of the final dataset.
Fig 21.0 Figure showing no missing value in the final dataset.
The outcome of the Data preprocessing is a dataset of 44094 observations and 8 features showing the
data is devoid of any missing value, null value, or datatype mismatch.
The following figures show the size of each partition after it was split into two (60% and 40%) to test
the robustness of the evaluation metrics.
33
Fig 21.1 Partition A (60%) Fig 21.2 Partition B (40%)
Figure 22.0 Scatter plot after preprocessing.
34
Fig 22.1. Scatter Plot before outlier removal Fig 22.2 Scatter Plot after outlier removal
Figure 23.0 Histogram of the numerical features after Preprocessing.
35
Fig 23.1 Histogram before Preprocessing Fig 23.2 Histogram after Preprocessing
Fig 24.0 Bar chart of categorical features after preprocessing

37
Fig 25.0 Silhouette score chart
Followed by K-Means, DBSCAN achieved a Silhouette score closest of 0.99, indicating it is the bestperforming
algorithm according to this metric. In contrast, Spectral demonstrated the lowest
performance with 0.06.
Fig 26.0 Davies-Bouldin score chart
Followed by K-Means, DBSCAN achieved a Davies-Bouldin score of 0.01, indicating it is the bestperforming
algorithm according to this metric. In contrast, K-Prototype demonstrated the lowest
performance with a score of 1.37.
38
Fig 27.0 Calinski-Harabasz Score chart
Followed by K-Means, DBSCAN achieved the highest Calinski-Harabasz score of 1163469880.96,
indicating it is the best-performing algorithm according to this metric. Conversely, Spectral
demonstrated the lowest performance with a score of 20893.48.
The partition-based clustering algorithms like K-means are the most utilized for segmenting customer
databases in the literature reviewed for this research. However, going by this outcome, the Densitybased
clustering algorithm (DBSCAN) has proven to be the most effective for the dataset used in this
study with the highest silhouette scores of 0.99, Davies-Boulding score of 0.01, and Calinski-
Harabasz score of 1163469880.96.55.
It is strongly believed that the reason for this outstanding performance of the density-based technique
(DBSCAN) is, that it is not restricted to 3 clusters like the other techniques and can identify clusters
based on density and arbitrariness in shape. The density-based technique’s robustness to noise may
also be an advantage over the partition-based and graph-based techniques.
With all the merits of K-means, it is ineffective for clusters with irregular shapes or varying sizes and
sensitive to outliers validating the position of Nagaraj, Kumar, Kumar, Venu, Sekar, and Rajkumar
(2023) that data structure and thorough data preprocessing impact the effectiveness or performance
of clustering techniques.
Out of all the clustering techniques investigated, K-means was the easiest and the most efficient to
implement. It is executed seamlessly and Spectral is understandably computationally more expensive
because of eigendecomposition, especially with a dataset of more than 45,000 observations. A subset
of the data had to be utilized for spectral clustering to execute successfully. Even at that, there were
warnings that the graph was not fully connected and that results may not be optimal.
0.06 1.3260893.48 0.58 0.51747216.17 0.35 1.3272191.24 0.99 0.01
1163469880.96
0.00
200000000.00
400000000.00
600000000.00
800000000.00
1000000000.00
1200000000.00
1400000000.00
Spectral K-Means Kprototype DBSCAN
Score
Clustering Technique
Calinski-Harabasz Metric
39
4.2 Comparison with Previous Research
In a previous study by Pandey, SV, Amrutha, Dash, and Patra (2023) on the same dataset as used in
this research, they concluded that K-mode has superior performance. However, this study proves
otherwise. Please find below the table and figures showing the performance.
Metric
Previous research This Research
K-mode Kmeans Kprototype
Silhouette Score 0.03 0.58 0.35
Davies-Bouldin 4.82 0.57 1.37
Table 6.0 Comparison of previous and current research outcomes
Fig 28.0 Silhouette score comparison between previous and current research results.
In the current research, the K-means and K-prototypes techniques achieved Silhouette scores of 0.58
and 0.35, respectively, demonstrating superior performance compared to 0.03 scored by the K-modes
technique utilized in the previous study.
It is worth noting that Kmode was implemented in this current study, and the outcome validated the
result of the previous research.
40
Fig 29.0 Davies-Boulding score comparison between current and previous research results.
In the previous research, the K-modes algorithm demonstrated suboptimal performance, as indicated
by a Davies-Bouldin index of 4.82. In contrast, the current study achieved significantly better results,
with the K-means algorithm yielding a score of 0.57 and the K-prototypes algorithm obtaining a score
of 1.37.
The superior performance of this study could possibly be attributed to the thorough data preprocessing
especially the remover of outliers (data whose ‘age’ or ‘balance’ Z-score is greater than absolute
value of 3) and the implementation of PCA.
Gupta, Kumar, Jain, Shrotriya, and Sinha (2022) in their research studied the K-prototype and
suggested that it worked well without giving a specific score. However, this research reveals that
although the K-prototype performed well, K-means performed better than Kprototype. Apart from
the superior performance, K-means is also computationally less expensive than the K-prototype.
4.3 Evaluation Metric Robustness Outcome
To check the robustness and how well the evaluation metrics can generalize, the dataset is split into
two then, each metric is used to measure the performance of the clusters generated on the two datasets
by the same algorithm.
To test the stability of the evaluation metrics, the dataset was split into A and B, 60% and 40%
respectively. K-means. K-prototype and DBSCAN were implemented on subsets and the generated
clusters were evaluated with the metrics. The outputs were compared to the already generated score
for the respective metrics against the whole dataset.
The table below shows the performance of the evaluation metrics on the split dataset. (Split A is 60%,
B is 40% and C is 100%)
41
Table 7.0: Table of the performance of the evaluation metrics.
From the table above, the Silhouette metric maintained a consistent performance irrespective of the
clustered dataset sizes. 0.58 for KMeans, 0.35 for Kprototype, and 0.99 for DBSCAN. This
performance is closely followed by the Davie-Bouldin index with a marginal variation in its
performance across different data sizes on Kprototype and DBSCAN techniques. Conversely,
Calinsky-Harabasz showed highly unstable outcomes across different data sizes on the three
techniques tested. Its score seems to increase with increase in the size of the clustered dataset.
4.4 Cluster Analysis
Except for DBSCAN which was not initialised with the number of clusters to generate, the other
clustering techniques had ‘K’ (number of clusters) set to 3 as dictated by the methods used to identify
the optimal number of clusters for the dataset. After preprocessing, the resultant 44,094 observations
and 8 features customer data were segmented into the sizes shown in the table below.
Cluster Cluster Size
Cluster Zero 16212 X 8
Cluster One 14091 X 8
Cluser Two 13791 X 8
Table 8.0 Table of cluster sizes
Metric Split K-Means K-prototype DBSCAN
A 0.58 0.35 0.99
B 0.58 0.35 0.99
C 0.58 0.34 0.99
A 0.57 1.36 0.57
B 0.57 1.38 0.80
C 0.57 1.37 0.01
A 88442.61 13377.97 17147.46
B 58772.76 8814.68 3727.98
C 147216.01 22191.24 1163469880.96
Calinski-Harabasz
Silhouette Score
Davies-Bouldin

43
4.4.2 Cluster Two
The map below indicates the features and the frequencies of their attribute in cluster one.
Fig 31.0 Heatmap of Cluster One
This cluster represents a distinct blend of the characteristics found in Cluster Zero and Cluster Two,
yet it exhibits unique attributes of its own. Notably, this cluster contains the highest number of young
adults, totalling 6,341, and the largest proportion of unmarried customers, numbering 6,764. Given
their demographic profile, these customers are likely to be enthusiastic, trendy, fashionable, and
highly tech-savvy. As such, the approach to communication and engagement with this group must
differ significantly from other clusters. Leveraging new media platforms and emerging technologies
will be crucial in reaching this audience effectively.
To further appeal to this segment, the bank could explore partnerships with the entertainment and
sports industries, offering tailored packages that resonate with their interests. Additionally,
introducing digital banking solutions, gamified financial literacy programs, and exclusive offers
through mobile apps or social media channels could enhance their banking experience. Special
incentives such as rewards for using mobile banking, cashback on lifestyle purchases, or discounts
on tech gadgets could also be effective in attracting and retaining this demographic. This cluster could
be aptly named 'The GenZ,' reflecting their vibrant, modern, and tech-oriented nature.
44
4.4.3 Cluster Three
I call this cluster ‘The Elite’. The map below indicates the features and the frequencies of their
attribute in cluster two.
Fig 32. Heatmap of Cluster Two
This cluster is distinguished by a high level of education, with 64% (8,826 individuals) holding
tertiary qualifications. Additionally, 30% (4,167 individuals) within this group possess a 'High'
account balance, the highest proportion across all clusters. Given their educational attainment and
financial status, this group is likely to be discerning and sophisticated in their financial preferences.
To effectively engage this segment, the bank should consider offering tailored investment products
that promise competitive returns and align with their financial goals. These could include wealth
management services, personalized portfolio management, and access to exclusive investment
opportunities such as real estate, stocks, or bonds. Additionally, products that cater to their long-term
financial planning, such as retirement funds, estate planning services, and tax optimization strategies,
would likely resonate with this group.
The level of engagement with this cluster should reflect their status and expectations. Personalized
banking services, such as dedicated relationship managers, priority customer service, and invitations
to exclusive events or seminars, could enhance their experience. Furthermore, digital solutions that
offer convenience and advanced features, such as sophisticated mobile banking apps with detailed
analytics and investment tracking, would appeal to their tech-savviness and need for efficient
financial management.
45
To further attract this educated and financially stable group, the bank could also consider partnerships
with prestigious educational institutions or professional organizations, offering scholarships, grants,
or continuing education programs that align with their interests. This cluster could be aptly named
'The Elite,' underscoring their distinguished educational background and financial standing.
4.4.4 Cluster Four
While a fourth cluster was not identified in the current analysis, it is crucial to consider a segment of
potential customers who are not yet part of the bank’s clientele. This hypothetical group, which could
be referred to as 'The Unbanked,' represents individuals who remain outside the formal banking
system. Developing strategies and tailored products to capture this segment aligns with the insights
of Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020), who emphasized the importance of
recognizing and addressing underserved groups.
To effectively reach 'The Unbanked,' the bank could explore several initiatives. First, the development
of accessible and low-cost banking solutions, such as mobile banking services with minimal fees and
simplified account opening processes, could lower the barriers to entry for this group. Financial
literacy programs, delivered through community outreach, mobile platforms, or partnerships with
local organizations, would also be vital in educating potential customers about the benefits of banking
and how to manage their finances effectively.
Additionally, introducing microfinance products, such as small loans, savings accounts with low
minimum balance requirements, and insurance packages designed for lower-income individuals,
could attract those who may feel excluded from traditional banking services. Collaborating with
government agencies or non-governmental organizations (NGOs) to facilitate these offerings might
further enhance the bank’s ability to reach and serve this group.
Furthermore, leveraging technology, such as mobile banking applications tailored for users with
limited digital literacy or access, could provide a gateway for the unbanked population to engage with
the banking system. By creating products and services that meet the unique needs of 'The Unbanked,'
the bank can expand its customer base and contribute to financial inclusion, thereby fostering broader
economic development.
46
5.0 DISCUSSION AND EVALUATION
5.1 Summary
Partition-based algorithms are the most widely used algorithms for customer segmentation and are
adjudged to be the best by many researchers. The outcome of this study which investigated the
partition-based (K-means and Kprototype) graph-based (Spectral) and density-based (DBSCAN)
algorithms showed that the DBSCAN clustering technique outperforms the others on the dataset used.
Equally, investigation on the robustness of the clustering evaluation metrics revealed that Silhouette
followed by Davies-Bouldin evaluation metric generalize well while Calinski_harabasz remains
unstable as its result depends on the size of the dataset clustered.
Contrary to Gupta, Kumar, Jain, Shrotriya, and Sinha's (2022) position that the K-prototype is better
at handling mixed datasets, and Pandey, SV, Amrutha, Dash, and Patra’s (2023) opinion that the
performance of Kmode is better in their research conducted on the same dataset as used in this study,
the outcome of this research shows that K-means is the best-performing partition-based clustering
algorithm for customer segmentation despite the introduction of other variants of K-means, even with
mixed data type.
5.2 Research Recommendation
To optimize customer segmentation strategies, African banks should prioritize investing in data
collection and quality assurance to obtain clean, accurate, and relevant data. This effort should include
integrating all customer touchpoints to create comprehensive customer profiles and addressing the
specific data challenges prevalent in the African market, such as data sparsity, inconsistent data
quality, and missing data. A similar sentiment was expressed by Sai, Kumari, and Gupta (2024).
Leveraging cloud technology is also recommended to establish a robust infrastructure capable of
handling large volumes of data.
Based on the findings of this study, a density-based clustering technique (DBSCAN) is recommended
for segmenting multivariate and dense customer datasets. The Silhouette Score evaluation metric is
advised due to its 99.9% performance on robustness and ability to generalize effectively across
datasets of varying sizes.
Moreover, clustering algorithms should be integrated with other machine learning models to predict
customer behaviors and dynamically adjust clusters as new data is collected. To enhance the
effectiveness of segmentation, banks should implement automated marketing platforms to deliver
personalized marketing campaigns tailored to each customer cluster. Ongoing staff training and
collaboration with technology companies and academic institutions are crucial for ensuring sustained
progress in leveraging these advanced analytical techniques.
In conclusion, data has emerged as a pivotal asset, essential not only for informed decision-making
but also for fostering development. Consequently, it is imperative that African banks engage in
collaborative efforts to establish open-source repositories containing anonymized customer data
within the region. Such initiatives would significantly benefit scholars and academia by facilitating
research endeavours, thereby ensuring that the region remains integrated into the global landscape of
innovation and technological advancement in the banking sector.
47
5.3 Critical Evaluation
This research was initially planned to be conducted using a comprehensive customer dataset from an
African bank, including demographic, behavioural, and transactional data. However, the primary
challenge encountered was the inability to obtain the desired dataset, as the banks approached were
unwilling to share their data due to various concerns. Consequently, the reliance on the UCI Machine
Learning Repository for a relevant dataset.
While the original plan included using customer behavioural and transactional data to better capture
customer dynamics, the analysis was limited to only demographic and static banking data available
in the dataset, such as customer balances and loan status. Nevertheless, applying the methodologies
deployed in this research to a real customer dataset from an African bank, once available, will yield
the desired outcomes.
The most widely used machine learning algorithms for customer segmentation according to
Umuhoza, Ntirushwamaboko, Awuah, and Birir (2020) are partition-based algorithms like K-means
and K-prototype. This research aimed to investigate these algorithms, along with Spectral (a graphbased
algorithm), Hierarchical clustering, and DBSCAN (a density-based algorithm), to compare
their performance on the dataset. However, due to time constraints, Hierarchical clustering was not
implemented. Additionally, the Gap statistics method for selecting the optimal number of clusters was
not employed due to unresolved code complexities and was instead replaced with the Davies-Bouldin
and Calinski-Harabasz Index.
The implemented clustering algorithms, except for K-means, were computationally expensive.
Specifically, the K-prototype technique required several hours of processing time due to its use of
different distance metrics (Hamming for categorical data and Euclidean for numeric data) and
aggregating the score to get a centroid.
Running the algorithms on a system with an 11th Gen Intel(R) Core (TM) i5-1135G7 @ 2.40GHz
processor, 16GB RAM, and approximately 300GB of free disk space led to high CPU, memory, and
disk usage, ultimately resulting in computational resource errors, especially when executing grid
searches for optimal parameters for DBSCAN and Spectral clustering techniques.
To overcome these challenges, the data was further processed to sizeable attributes. For instance, the
‘job’ feature of 11 unique attributes was reduced to only 2 (‘employed’ and ‘unemployed’), ‘marital’
was reduced to ‘married’ and ‘single’, and ‘education’ was reduced to ‘primary’, ‘secondary’, and
‘tertiary’.
This experience highlights the need for quality data, thorough data preprocessing, and scalable
systems with adequate resources. These are key considerations for African banks interested in
implementing these techniques.
Additional challenges included coding errors during data preprocessing and algorithm
implementation. These issues were resolved by consulting lecture notes and reading Python library
documentation. Despite these difficulties, four of the five planned clustering techniques were
successfully implemented, with DBSCAN proving to be the most effective, despite its computational
intensity. Furthermore, the research successfully evaluated the stability of three intrinsic clustering
evaluation metrics and concluded on their suitability.
48
5.4 Future Work
An area of focus opened by this study is the investigation of the possibility of implementing a
combination of the density-based and the partition-based clustering technique on an African Bank
dataset to verify if it will result in a superior performance rather than implementing the techniques
separately and comparing the outcomes.
During this research, I encountered a distance metric known as 'Gower distance'. It is specifically
designed to handle mixed data types, including numeric, categorical, and binary variables. This metric
computes the distance between records by normalizing the contributions of each feature, thereby
preventing any single feature type from disproportionately influencing the overall distance
calculation. The resulting Gower matrix can then be utilized as an input for clustering algorithms that
accept a precomputed distance matrix, such as DBSCAN and Hierarchical Clustering. Further
investigation into this could determine its efficacy or otherwise.
It is recommended that real customer data from an African bank, encompassing transactional,
demographic, and behavioural data, be utilized for this research. The use of such data would enhance
the validity of the outcomes and yield actionable insights for banks in the region.
Due to time constraints, the investigation of other clustering techniques suitable for customer
segmentation, such as Hierarchical Clustering and the Gaussian Mixture Model was not pursued.
Additionally, exploring Gap statistics and other intrinsic metrics for evaluating clustering
performance, such as the Dunn Index, alongside those employed in this study, could further enrich
the analysis.
5.5 Commercial and Economic Considerations.
The commercial and economic context of this study provides the necessary background for which the
study is planned, executed, and evaluated. It is essential to ensure that the research is viable,
strategically aligned, and capable of delivering value in a competitive and dynamic market.
Understanding this context is critical to making informed decisions, managing risks, and achieving
successful outcomes. The comprehensive details of the key considerations in this study are given
below.
5.5.1 Commercial Context
1. Market Competition: The African banking sector is highly competitive, with Fintechs, local,
and international banks striving to attract and retain customers. Effective customer
segmentation can provide a competitive edge by enabling banks to tailor their marketing
strategies, products, and services to the specific needs and preferences of different customer
segments.
2. Personalized Marketing: By using clustering algorithms for customer segmentation, banks
can move from a one-size-fits-all marketing approach to a more personalized strategy. This
can enhance customer satisfaction and loyalty, leading to increased customer lifetime value
and a stronger market position.
49
3. Customer Retention: Understanding customer behavior and preferences through data-driven
segmentation allows banks to identify volatile customers and implement targeted retention
strategies. This can reduce churn rates and ensure a stable customer base, which is crucial in
a competitive market.
4. Product Development: Clustering algorithms can help identify gaps in the market or unmet
customer needs, enabling banks to develop new products or services tailored to specific
segments. This can lead to innovation and differentiation in a crowded market.
5.5.2 Economic Context
1. Financial Inclusion: In many African countries, a significant portion of the population
remains unbanked or underbanked. By effectively segmenting customers, banks can develop
targeted strategies to reach these populations, contributing to broader financial inclusion
efforts and economic development.
2. Cost Efficiency: Traditional marketing methods can be expensive and inefficient. Clustering
algorithms enable banks to optimize their marketing spend by focusing resources on the most
promising customer segments, leading to better returns on investment and reduced costs.
3. Economic Growth: By optimizing marketing strategies and enhancing customer engagement,
banks can drive higher sales volumes and profitability. This, in turn, can contribute to the
overall economic growth of the region by improving the financial health and performance of
banking institutions.
4. Risk Management: Effective customer segmentation can also help in managing economic
risks by identifying creditworthy customers and minimizing exposure to high-risk segments.
This is particularly important in regions with volatile economic conditions.
In summary, this project operates within a commercial and economic context where African banks
seek to enhance their competitive positioning, drive customer engagement, and contribute to
economic growth through effective customer segmentation. By leveraging clustering algorithms,
these banks can achieve greater market efficiency, innovation, and financial inclusion, ultimately
benefiting both the institutions and the broader economy.
5.6 Project Management
Project management among many other reasons is important for planning, risk management,
problem-solving, and time management to ensure successful outcomes of a project. The management
of this study involves a structured approach to ensure the successful completion of all the phases. The
project is divided into clearly defined stages from Project Topic Selection to Project Demonstration.
Some key phases in this project include Data Collection, Detailed Project Proposal, Literature Search
and Review, Evaluation and Selection of Research Methodology.
Others are Interim Project Report, Data Preprocessing, Algorithm Implementation, Analysis, Final
Report, and Viva. Each phase is governed by a timeline with specific milestones and deliverables,
ensuring that the objectives are met within the stipulated period and that the project remains on track.
Details of this are in the Gantt chart. A git repository was also created for this project where all
artifacts, literature and preliminary outcomes are pushed and pulled as required. A disciplined
approach was maintained, with regular self-assessments and adjustments made to stay on track.
50
Effective self-coordination, time management, and task prioritization are key to managing this project
independently. Clear daily and weekly goals are established to maintain focus and ensure consistent
progress. All research, data analysis, and reporting activities are documented meticulously, ensuring
that the project’s workflow remains organized and coherent. Regular reviews of the progress against
the project plan allow for the timely identification of any challenges or delays, which are then
addressed promptly.
The project was originally planned to be completed by 23rd of August but in practical, it ended on
30th of August due to some challenges and unforeseen circumstances as discussed earlier. Even at
that, some planned activities, for instance the implementation of hierarchical clustering, were not
executed due to time constraint. A screenshot of project Gantt chart is attached in the Appendix on
page 64.
Although working independently, periodic consultations with my project supervisor on milestones
and outcomes to gain insights and validate findings are constant, ensuring that the project outcomes
are robust and credible. This approach ensures that the project remains well-organized and capable of
achieving its objectives.
5.7 CONCLUSION
This study on the use of clustering algorithms to optimize marketing strategies in African banks
through customer segmentation underscores the transformative potential of data-driven approaches
in the financial sector. By leveraging advanced clustering techniques, particularly with the drive for
financial inclusion and the adoption of digital banking services which has resulted in huge amount of
customer data in the diverse and dynamic context of African banking, the value of unsupervised
machine learning techniques to tailor marketing strategies that align more closely with the specific
behaviours, preferences, and needs of distinct customer segments have been demonstrated.
The research highlights the density-based clustering algorithm (DBSCAN) as the most effective out
of the ones investigated, in identifying meaningful customer segments within large, complex datasets.
These segments allow banks to develop more targeted marketing campaigns, enhance customer
satisfaction, and improve overall business performance.
While selecting an appropriate clustering technique is important, a critical factor is the reliability of
the metric used in evaluating the performance of such a technique. This study has shown that the
Silhouette closely followed by Davies-Bouldin metric rather than the Calinski-Harabasz is robust
enough for such evaluations.
Finally, for African banks to successfully implement and leverage clustering algorithms to optimize
marketing strategies, they must focus on high-quality data collection, selecting appropriate algorithms
and evaluation metrics, personalizing marketing efforts, leveraging technology, ensuring ethical
practices, and continuously monitoring and optimizing their strategies. By following these best
practices, the banks can better understand their customers, improve engagement, and enhance
profitability through more effective marketing strategies.
In conclusion, this research contributes to the growing body of knowledge on the application of data
science in banking, offering practical insights for optimizing marketing strategies through customer
segmentation. It also sets the stage for further exploration into the use of other machine-learning
51
techniques in banking and finance, particularly in the African context. The successful implementation
of clustering algorithms in this study demonstrates that with the right analytical tools, African banks
can achieve greater precision in understanding their customers, leading to more effective and
impactful marketing strategies.
Reflecting on this research journey, I can confidently state that my understanding of data science has
significantly deepened. My Python programming, analytical, problem-solving, and project
management skills have been greatly enhanced and adequately prepared me for future opportunities
and challenges.
52
REFERENCE
1. Ahmed, G., Ahmed, A., Ahmed, M., Latha, J. and Kumar, P. (2024) ‘Indian Banking Precision
Marketing: A Comparative Analysis of Machine Learning Customer Segmentation
Algorithms’, 2024 2nd International Conference on Cyber Resilience (ICCR), pp. 1-6.
Available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10532917
(Accessed: 30 June 2024).
2. Akbar, Z., Liu, J. and Latif, Z. (2020) ‘Discovering knowledge by comparing silhouettes using
k-means clustering for customer segmentation’, International Journal of Knowledge
Management (IJKM), 16(3), pp.70-88. Available at:
http://dx.doi.org.ezproxy.herts.ac.uk/10.4018/IJKM.2020070105 (Accessed:19 June 2024)
3. Bhimarapu, H., Bhelkar, S., Chavhan, N., Dhule, C. and Agrawal, R. (2023) ‘Customer
Segmentation Based on E-Commerce using K-Mean Clustering’, 2023 International
Conference on Recent Advances in Science and Engineering Technology (ICRASET), pp. 1-5.
Available at: IEEE Xplore Full-Text PDF: (herts.ac.uk) (Accessed: 30 June 2024)
4. Deng, Y. and Gao, Q. (2020) ‘A study on e-commerce customer segmentation management
based on improved K-means algorithm’, Information systems and e-business management,
18(4), pp. 497–510. Available at: A study on e-commerce customer segmentation management
based on improved K-...: EBSCOhost (herts.ac.uk) (Accessed:19 June 2024)
5. Dinavahi, G.A., Thatavarti, H., Rangala, J., Vallamsetti, S. and Nannuri, J.L. (2023)
‘Customer Segmentation in Retailing using Machine Learning Techniques’ 2023 IEEE 8th
International Conference for Convergence in Technology (I2CT) pp. 1-5 Available at:
https://ieeexplore.ieee.org/document/10126155 (Accessed:19 June 2024)
6. Ezenkwu, C., Ozuomba, S., and Kalu C (2015) ‘Application of kmeans algorithm for efficient
customer segmentation: A strategy for targeted customer services’, International Journal of
Advanced Research in Artificial Intelligence (IJARAI), vol. 4, 10 2015. Available at:
Application of K-Means Algorithm for Efficient Customer Segmentation: A Strategy for
Targeted Customer Services (thesai.org) (Accessed: 12 July 2024)
7. Ganar, C. and Hosein, P. (2022), ‘Customer segmentation for improving marketing campaigns
in the banking industry’ 2022 5th Asia conference on machine learning and computing
(ACMLC) pp. 48-52. Available at: https://ieeexplore.ieee.org/document/10221808 (Accessed:
30 June 2024).
8. Gupta, R., Kumar, H., Jain, T., Shrotriya, A. and Sinha, A. (2022) ‘Demographic Customer
Segmentation of banking users Based on k-prototype methodology’, 2022 12th International
Conference on Cloud Computing, Data Science & Engineering (Confluence) pp. 578-584.
Available at: https://ieeexplore.ieee.org/document/9734169 (Accessed: 30 June 2024).
9. Gankidi, N., Gundu, S., viqar Ahmed, M., Tanzeela, T., Prasad, C.R. and Yalabaka, S. (2022)
‘Customer segmentation using machine learning’, 2022 2nd International Conference on
Intelligent Technologies (CONIT) pp. 1-5. Available at:
https://ieeexplore.ieee.org/document/9848389 (Accessed: 19 June 2024).
10. Feng, Y., Wang, X. and Li, L. (2019) ‘The Application Research of Customer Segmentation
Model in Bank Financial Marketing’ 2019 2nd International Conference on Safety Produce
Informatization (IICSPI) pp. 564-569. Available at:
https://ieeexplore.ieee.org/document/9095900 (Accessed: 27 July 2024)
11. Hossain, M.M., Sebestyen, M., Mayank, D., Ardakanian, O. and Khazaei, H. (2020) ‘Largescale
data-driven segmentation of banking customers’ 2020 IEEE International Conference
on Big Data (Big Data) pp. 4392-4401. Available at:
https://ieeexplore.ieee.org/document/9378483 (Accessed: 19 June 2024)
12. Li, X. and Lee, Y.S. (2024) ‘Customer Segmentation Marketing Strategy Based on Big Data
Analysis and Clustering Algorithm’ Journal of Cases on Information Technology
53
(JCIT), 26(1), pp.1-16. Available at: https://www.igi-global.com/article/customersegmentation-
marketing-strategy-based-on-big-data-analysis-and-clusteringalgorithm/
336916 (Accessed: 19 June 2024).
13. Maddumala, V.R., Chaikam, H., Velanati, J.S., Ponnaganti, R. and Enuguri, B. (2022)
‘Customer segmentation using machine learning in Python’, 2022 7th International
Conference on Communication and Electronics Systems (ICCES) pp. 1268-1273. Available
at: https://ieeexplore.ieee.org/document/9836018 (Accessed: 19 June 2024).
14. Moro, S., Rita, P., and Cortez, P. (2012). Bank Marketing. UCI Machine Learning Repository.
Available at:https://archive.ics.uci.edu/dataset/222/bank+marketing(Accessed: 19 June 2024)
15. Nagaraj, P., Kumar, C.B., Kumar, K.C., Venu, B., Sekar, R.R. and Rajkumar, T.D. (2023),
‘Customer Segmentation Using Supervised and Unsupervised Machine Learning
Techniques’, 2023 International Conference on Data Science, Agents & Artificial Intelligence
(ICDSAAI) pp. 1-6. Available at: https://ieeexplore.ieee.org/document/10452643 (Accessed:
19 June 2024)
16. Nandapala, E.Y.L. and Jayasena, K.P.N. (2020), ‘The practical approach in Customers
segmentation by using the K-Means Algorithm’, 2020 IEEE 15th international conference on
industrial and information systems (ICIIS) pp. 344-349. Available at:
https://ieeexplore.ieee.org/document/9342639 (Accessed: 18 June 2024).
17. Pandey, T.N., SV, N.K., Amrutha, M.S., Dash, B.B. and Patra, S.S. (2023). ‘Experimental
Analysis on Banking Customer Segmentation using Machine Learning Techniques’, 2023
Global Conference on Information Technologies and Communications (GCITC) pp. 1-6.
Available at: https://ieeexplore.ieee.org/document/10426116 (Accessed: 30 June 2024).
18. Patel, Y.S., Agrawal, D. and Josyula, L.S. (2016), ‘The RFM-based ubiquitous framework for
secure and efficient banking’, 2016 International Conference on Innovation and Challenges
in Cyber Security (ICICCS-INBUSH) pp. 283-288. Available at:
https://ieeexplore.ieee.org/document/7542333 (Accessed: 30 June 2024).
19. Sai, C.A.L.V.S., Kumari, K.S. and Gupta, B.A. (2024), ‘Churn Prediction Based on Customer
Segmentation in Banking Industry using Machine Learning Techniques’, 2024 International
Conference on Automation and Computation (AUTOCOM) pp. 388-393. Available at:
https://ieeexplore.ieee.org/document/10486164 (Accessed: 30 June 2024).
20. Sharma, S. and Desai, N. (2023), ‘Data-Driven Customer Segmentation Using Clustering
Methods for Business Success’, 2023 4th IEEE Global Conference for Advancement in
Technology (GCAT) pp. 1-7. Available at: https://ieeexplore.ieee.org/document/10353367
(Accessed: 19 June 2024).
21. Sudirman, I.D., Utama, I.D., Bahri, R.S. and Susanto, R.H. (2023), ‘Unveiling Purchasing
Patterns in Grocery Store Consumer Segmentation Insight From K-Means Clustering’, 2023
1st IEEE International Conference on Smart Technology (ICE-SMARTec) pp. 145-150.
Available at: https://ieeexplore.ieee.org/document/10461973 (Accessed: 19 June 2024).
22. Tibshirani, R., Walther, G. and Hastie, T. (2001), ‘Estimating the number of clusters in a data
set via the gap statistic’, Journal of the Royal Statistical Society: Series B (Statistical
Methodology), 63(2), pp.411-423. Available at:
https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00293 (Accessed: 19 July
2024).
23. Umuhoza, E., Ntirushwamaboko, D., Awuah, J. and Birir, B. (2020). ‘Using unsupervised
machine learning techniques for behavioral-based credit card users segmentation in
Africa’. SAIEE Africa Research Journal, 111(3), pp.95-101. Available at:
https://ieeexplore.ieee.org/document/9142602 (Accessed: 18 June 2024).
54
APPENDIX
Bank-full_2.csv, MSc_Project_Artifacts.py and MSc_Project_Gantt_chart.xlsx are the Dataset,
python codes and project plan for this project.
These and other materials including preliminary results, graphs, artefacts, and literature relating to
this study can be found in the git repository below:
Appendix 1
A snapshot of the bank’s customer database
55
Appendix 2
Figure 1.0: Histograms showing the “age” and “balance” feature distribution.
Appendix 3
Figure 2.0: Bar charts showing the distribution of the categorical features.
56
Appendix 4
A snapshot of the code and the result of removing the unsuitable features.
Appendix 5
A snapshot of the Data Statistics
57
Appendix 6
A snapshot of the check for missing Data
Appendix 7
58
Another snapshot of the check for missing Data
Appendix 8
A snapshot of checking datatype mismatch
Appendix 9
59
A snapshot of the artifact to visualize the Data distribution via Histogram and Bar Charts
Appendix 10
A snapshot of the artifact for standardizing the numerical features, visualizing them through
scatter plots and box plots, and checking for outliers.
Appendix 11
60
Appendix 12
Appendix 13
61
A snapshot of the dataset showing the new values for the numerical features, (age and balance)
after standardization.
Appendix 14
A snapshot of the Elbow method implementation and the result.
Appendix 15
62
A snapshot of the Silhouette method implementation and the result.
Appendix 16
A snapshot of the artifact implementing K-Means on the dataset
63
Appendix 17
A snapshot of the dataset showing the clusters after K-Means implementation
Appendix 18
A snapshot of the artifact for K-prototype implementation.
64
Appendix 19
A snapshot of the artifact for Random Forest Classifier implementation.
Appendix 20
A snapshot of the Project’s Gantt chart