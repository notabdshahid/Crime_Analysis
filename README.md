You can run the project for yourself [here](https://uscrimeanalysis.streamlit.app/)
# Crime Analysis
Crime is a pervasive issue that affects the safety, economic stability, and the overall quality of life in a community. Understanding factors that contribute to criminal activity and identifying patterns in crime trends are essential steps in implementing effective policies and interventions. However, given the complex interplay of socioeconomic, demographic, and geographical factors influencing crime, the task of deriving actionable insights from crime data is a non-trivial challenge. This project aims to address these challenges through the application of data analytics and machine learning techniques to explore and analyse crime trends. 
The primary objective of this project is to analyse a crime dataset to uncover key features influencing violent crime rates and evaluate patterns across different states and communities in the United States. The analysis focuses on the following critical aspects:
1.	Feature Importance Identification
2.	Correlation and Perturbation Analysis
3.	Geospatial and Hotspot Analysis
4.	State-by-State Comparison

The raw dataset contained several irrelevant or redundant features, which were excluded to improve model performance. Examples include columns representing specific police demographic data and features unrelated to crime trends. Additionally, rows with missing values in the target variable (ViolCrimesPerPop) were dropped, while other missing values were imputed using mean imputation. 
To evaluate the predictive capabilities of various machine learning models, three different regressors were employed: Decision Tree Regressor, Random Forest Regressor, and Support Vector Regression (SVR). Each regressor's performance was initially assessed using standard regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and the R² Score. Among the three models, the Random Forest Regressor demonstrated the best performance, achieving a significantly higher R² score and lower error metrics compared to the other regressors. 
## Feature Importance
![image](https://github.com/user-attachments/assets/f0351573-8753-4039-b20d-0a26cec19961)

The Random Forest model identified the most critical features influencing violent crime rates. The feature racePctWhite was the most important, with an importance score of 0.382839, far surpassing the other variables. This suggests that areas with a higher percentage of white population might exhibit distinct trends in violent crime rates. Other significant factors included TotalPctDiv (0.086631) and racepctblack (0.059428), indicating that diversity and racial demographics play crucial roles.
Interestingly, features like PctHousLess3BR and PctUsePubTrans had relatively lower importance scores but still showed measurable effects. These features may be indirect indicators of socio-economic conditions, such as housing quality and access to public transportation, which influence crime rates.
## Correlation Analysis
![image](https://github.com/user-attachments/assets/d5789a0d-bc7d-4fdb-99ed-d6efe169b0a4)

Correlation analysis further revealed the relationships between the top 10 features and the target variable, ViolentCrimesPerPop. Notable observations include:

•	racePctWhite exhibited the strongest negative correlation (-0.679), implying that as the proportion of the white population increases, violent crime rates tend to decrease significantly. This result highlights potential socio-economic and systemic differences between communities with varying racial compositions.

•	racepctblack had a strong positive correlation (0.632), suggesting that communities with a higher percentage of Black populations tend to experience higher violent crime rates. This correlation may reflect underlying systemic inequalities, such as economic disparities or historical disenfranchisement, that disproportionately affect these communities.

•	Features such as PctPopUnderPov (0.512) and PctHousNoPhone (0.479) were also strongly correlated with violent crime rates. These findings highlight the critical role of poverty and access to resources in 
shaping crime trends.

It is, however, important to mention that racePctWhite exhibiting the strongest negative correlation and racepctblack exhibiting the strongest positive correlation does not mean that reducing the population of a particular people in that particular area would lead to less crime. But rather, it tells us that the percentage of black people in areas that have more poverty and less access to resources to pull themselves out of the cycle of crime is higher. The opposite holds true for white people and that in turn means there is less crime in areas where they reside. 
## Perturbation Analysis:
![image](https://github.com/user-attachments/assets/bb027899-4516-4908-9ba9-3cd0d679eb0a)

![image](https://github.com/user-attachments/assets/440043d5-538a-4f92-8a63-b7cd9758f38d)

The feature perturbation analysis provided further insight into the sensitivity of violent crime predictions to changes in key features. Notably:

•	Decreasing racePctWhite led to an average increase of 91.26 in violent crime predictions, while increasing it reduced predictions by 32.03. This strong sensitivity aligns with the high feature importance and negative correlation observed, further reinforcing its significant role.

•	TotalPctDiv, a measure of divorce rates in the population, exhibited high sensitivity in both directions, with increases leading to a rise of 35.42 and decreases causing a drop of 33.59. This suggests that areas with more divorce experience higher variability in crime rates, potentially reflecting the complex dynamics of such communities.

•	Socio-economic indicators, such as PctPopUnderPov and NumUnderPov, also showed consistent effects, highlighting the persistent impact of poverty on crime rates.
The analysis reveals that addressing specific demographic and socio-economic factors, such as increasing opportunities for underprivileged communities, could lead to substantial improvements in public safety.
## Hotspot Analysis:
![image](https://github.com/user-attachments/assets/8f2a930d-56c2-4e4b-94ea-3975c41fc4d2)

The hotspot analysis identified states with the highest average violent crime rates. The District of Columbia (3048.38) was a clear outlier, with crime levels far exceeding those in other states. This may reflect its unique urban environment and socio-economic disparities. Louisiana (1312.71) and South Carolina (1233.46) also reported notably high rates, likely influenced by systemic issues such as poverty, education disparities, and historical inequalities.
Conversely, states like North Dakota (85.06), Vermont (116.11), and Maine (151.49) reported the lowest crime rates. These states are generally less populous, with predominantly rural landscapes and smaller income disparities. Their lower crime rates may also reflect stronger community cohesion and access to resources.
The stark regional differences highlight the need for targeted interventions. For example:

•	Urban areas like Washington, D.C., could benefit from community-based crime reduction strategies and economic investment.

•	States with high poverty rates, such as Louisiana and South Carolina, may need comprehensive socio-economic reforms to address root causes of crime.

•	The success of low-crime states could serve as models for fostering safer communities.
## Overall Insights
The combined results from feature importance, correlation analysis, perturbation studies, and hotspot analysis offer valuable insights:

1.	Demographics and socio-economic factors significantly influence crime rates, highlighting the importance of addressing inequality and systemic barriers.

2.	Urban areas and diverse communities require nuanced approaches to tackle the underlying causes of crime effectively.

3.	States with lower crime rates demonstrate that socio-economic stability, education, and access to resources play a pivotal role in maintaining public safety. The interactive dashboard can be used to generate visualisations dynamically on a state-by-state basis so policy makers can compare the most important features in areas with low crime and areas with high crime. This comparison could further lead them to more insights on what policies could be implemented to fix the issue of high crime. 
These findings provide a foundation for future research and policy-making, aimed at reducing violent crime through data-driven, targeted strategies.
##
# Usage

