# Stock-Price-Prediction-of-Apple-Using-Linear-Regression-and-Technical-Indicators
Apple Stock Price Prediction using Regression Models
This project focuses on predicting Apple Inc. (AAPL) stock prices using various linear regression models and technical indicators. It explores data acquisition, comprehensive feature engineering, rigorous assumption validation, and detailed performance evaluation to forecast future stock movements.

Table of Contents
Introduction

Project Objective

Data Acquisition

Feature Engineering

Methodology

Ordinary Least Squares (OLS) Regression

Lasso Regression

Ridge Regression

Elastic Net Regression

Model Evaluation

Assumption Validation (for OLS)

Future Predictions (2025)

Conclusion & Limitations

Technologies Used

How to Run

Contributing

License

Introduction
Stock price prediction is a challenging yet crucial task in quantitative finance, driven by a multitude of complex and often unpredictable factors. This project aims to delve into the application of various linear regression models to forecast the next-day closing price of Apple Inc. (AAPL). We leverage a combination of historical stock data, including the performance of peer financial stocks and broader market indices, along with carefully selected technical indicators. The goal is to capture relevant market dynamics and trends that can inform future price movements. The project emphasizes a robust analytical pipeline, starting from meticulous data preparation and advanced feature engineering, extending through rigorous model validation, and concluding with a comprehensive performance assessment to understand the strengths and limitations of each approach.

Project Objective
The primary objectives of this project are multifaceted, designed to provide a thorough understanding of applying regression techniques to financial time series:

Model Implementation: To design and implement various linear regression models, including Ordinary Least Squares (OLS), Lasso, Ridge, and Elastic Net, to predict stock prices. This involves understanding their underlying mathematical principles and practical application in Python.

Comprehensive Feature Engineering: To perform in-depth feature engineering by creating meaningful variables such as lagged returns and various moving averages. This step is crucial for transforming raw data into a format that enhances the predictive power of the models.

Rigorous Assumption Validation: To critically examine and validate the core assumptions underlying linear regression models. This ensures the statistical validity and reliability of the OLS model's results, using a suite of diagnostic tests.

Performance Evaluation: To thoroughly evaluate the performance of each developed model using key statistical metrics like R-squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Additionally, visual comparisons of actual versus predicted prices will provide intuitive insights into model accuracy.

Forward-Looking Analysis: To utilize the best-performing model to generate forward-looking stock price predictions for a specific future period, demonstrating the practical applicability of the developed models.

Data Acquisition
Reliable and extensive historical stock data is the foundation of any predictive financial model. For this project, data was efficiently acquired using the yfinance library, a popular and convenient tool for accessing Yahoo Finance data.

Primary Stock: Apple Inc. (AAPL), chosen for its high liquidity and significant market presence, offering a robust dataset for analysis.

Other Tickers (for cross-market variables): To account for broader market influences and inter-stock correlations, data was also collected for:

Amazon (AMZN)

Microsoft (MSFT)

Invesco QQQ Trust (QQQ) - representing the NASDAQ-100 Index.

S&P 500 Index (^GSPC) - a key benchmark for the overall market.

Date Range:

Training & Testing: Historical daily 'Close' prices were collected from January 1, 2020, to December 31, 2024. This extensive period allows the models to learn from diverse market conditions, including periods of volatility and stability.

Future Prediction: A separate dataset for the prediction period was acquired from January 1, 2025, to March 31, 2025, to evaluate the models' out-of-sample forecasting capabilities.

Data Fetched: For all specified tickers, only the daily 'Close' prices were extracted, as this is the primary variable of interest for price prediction.

Data Cleaning: Missing values, which commonly arise from shift() or rolling() operations during feature engineering (as these operations can produce NaN values at the beginning of the series), were systematically handled by dropping the affected rows (dropna()). This ensures that the dataset used for model training is complete and free from missing observations.

Feature Engineering
Why Feature Engineering?
Raw closing prices alone often lack the sophisticated patterns and relationships necessary for accurate stock price prediction. Feature engineering is the art and science of transforming raw data into features that better represent the underlying dynamics of the market. This process is crucial for enhancing the predictive power of models by providing them with more informative inputs. By creating new variables from existing ones, we aim to capture trends, momentum, and inter-market dependencies that simple price series might overlook.

Features Created:

Lagged Closing Prices: These features represent the stock's or index's price from previous trading days.

AAPL(t-1): Apple's closing price from the immediate previous day. This is a fundamental feature, as today's price is highly correlated with yesterday's.

AMZN(t-1), MSFT(t-1), QQQ(t-1), ^GSPC(t-1): Lagged closing prices of peer stocks and major market indices. These act as cross-market variables, capturing broader market sentiment, sector-specific movements, and potential correlations that might influence AAPL's price. For instance, a strong performance in the overall market (^GSPC) or within the tech sector (AMZN, MSFT, QQQ) could indicate a positive outlook for AAPL.

Moving Averages (Technical Indicators): Moving averages are widely used technical indicators that smooth out price data over a specified period, helping to identify trends.

AAPL_MA_5, AMZN_MA_5, MSFT_MA_5, QQQ_MA_5, ^GSPC_MA_5: These represent the 5-day simple moving averages for each respective stock/index. A 5-day MA is a short-term indicator, useful for understanding recent price momentum and potential short-term reversals or continuations.

Target Variable:

Target: This is the dependent variable that our models are designed to predict. It is set as the next day's Apple closing price (df['AAPL'].shift(-1)). This "shifted" value ensures that our models are predicting a future price, not simply fitting to the current day's price.

After creating these features, any rows containing NaN values (which occur at the beginning of the dataset due to the shift() or rolling() operations) were systematically removed using df.dropna(). This ensures a clean and complete dataset for subsequent model training and evaluation.

Methodology
The project employs a range of linear regression models to predict stock prices, allowing for a comparative analysis of their effectiveness. The dataset was meticulously prepared and split to ensure robust model training and evaluation.

Data Splitting: The data was divided into a training set (95% of the data) and a testing set (5% of the data). A crucial aspect of this split was using shuffle=False. This ensures that the time-series order of the data is preserved, which is absolutely critical for financial time-series prediction, as future data points should not influence the training of the model.

Ordinary Least Squares (OLS) Regression
Description: Ordinary Least Squares (OLS) is the most fundamental form of linear regression. Its objective is to find the line (or hyperplane in multiple dimensions) that best fits the data by minimizing the sum of the squared differences between the observed values and the values predicted by the model. It provides an unbiased estimate of the regression coefficients under certain assumptions.

Implementation: We utilized statsmodels.api.OLS for its comprehensive statistical summary, which provides detailed insights into the model's coefficients, p-values, and diagnostic statistics, making it excellent for understanding the underlying relationships.

Initial Features: All 10 engineered features (5 lagged prices and 5 moving averages) were initially included in the OLS model.

Outcome (Initial Model):

R-squared: 0.993

Explanation: This exceptionally high R-squared value suggests that the initial OLS model explains approximately 99.3% of the variance in the target variable (next-day AAPL price) within the training data. While seemingly impressive, such a high value in-sample can sometimes be a red flag for overfitting, especially with highly correlated time-series data.

Feature Selection: After the initial OLS analysis, features with high p-values (typically > 0.05), indicating that their coefficients are not statistically significant, were considered for removal. The notebook specifically highlights AAPL(t-1) and ^GSPC(t-1) as statistically significant features. This iterative process of feature selection helps in simplifying the model, potentially reducing multicollinearity, and improving generalization.

Outcome (After Feature Selection - for 2025 prediction):

R-squared: 0.992

MSE: 17.35

RMSE: 4.16

Explanation: Even after feature selection, the OLS model maintained a very high R-squared, indicating its continued strong explanatory power. The MSE and RMSE values provide concrete measures of prediction error, with an RMSE of 4.16 meaning that, on average, the model's predictions were off by about $4.16 from the actual stock price.

Lasso Regression
Description: Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a linear regression model that incorporates L1 regularization. This technique adds a penalty term to the OLS objective function, proportional to the absolute value of the magnitude of the regression coefficients. The unique property of Lasso is its ability to shrink some coefficients exactly to zero, effectively performing automatic feature selection. This makes Lasso particularly useful when dealing with a large number of features or when multicollinearity is present.

Implementation: We used sklearn.linear_model.Lasso(alpha=10). The alpha parameter controls the strength of the regularization; a higher alpha imposes a stronger penalty.

Outcome:

R-squared: 0.7598

MSE: 33.88

RMSE: 5.82

Drawback: The significantly lower R-squared compared to OLS (0.7598 vs. 0.992) suggests that while Lasso successfully performs feature selection and reduces overfitting, it might sacrifice some of the in-sample fit. This trade-off is often acceptable or even desirable for better generalization to unseen data, especially if the underlying relationships are truly sparse. However, for this specific dataset, the strong regularization might have removed some genuinely useful features or over-penalized coefficients.

Ridge Regression
Description: Ridge Regression is another linear regression model that utilizes L2 regularization. Unlike Lasso, Ridge adds a penalty term proportional to the square of the magnitude of the regression coefficients. This penalty shrinks the coefficients towards zero but rarely makes them exactly zero. Ridge Regression is particularly effective at handling multicollinearity by distributing the impact of correlated features across all of them, rather than selecting just one.

Implementation: We implemented Ridge Regression using sklearn.linear_model.Ridge(alpha=10). Similar to Lasso, alpha dictates the strength of the regularization.

Outcome:

R-squared: 0.7443

MSE: 36.06

RMSE: 6.00

Drawback: Similar to Lasso, Ridge introduces a slight bias into the model's estimates to achieve a reduction in variance. This can lead to a slightly lower R-squared on the training data compared to an unregularized OLS model. While effective against multicollinearity, its inability to perform feature selection (by setting coefficients to exactly zero) means it retains all features, which might not be optimal for very high-dimensional datasets.

Elastic Net Regression
Description: Elastic Net Regression is a powerful hybrid linear regression model that combines both L1 (Lasso) and L2 (Ridge) regularization. It aims to leverage the strengths of both: Lasso's ability to perform feature selection (sparsity) and Ridge's capacity to handle multicollinearity by shrinking coefficients. It is particularly useful when there are multiple correlated features, as it tends to select groups of correlated variables together.

Implementation: We used sklearn.linear_model.ElasticNet(alpha=1, l1_ratio=0.5). Here, alpha is the overall regularization strength, and l1_ratio (between 0 and 1) determines the mix between L1 and L2 penalties (0 for pure Ridge, 1 for pure Lasso). A value of 0.5 means an equal mix.

Outcome:

R-squared: 0.6638

MSE: 35.00

RMSE: 5.91

Drawback: Elastic Net's performance is highly sensitive to the tuning of its two hyperparameters (alpha and l1_ratio). The lower R-squared observed in this specific run suggests that the chosen regularization parameters might have been too strong for the dataset's characteristics, potentially leading to an underfit model. Optimal performance often requires extensive cross-validation to find the best combination of these parameters.

Model Evaluation
Evaluating the performance of predictive models is crucial to understand their accuracy, reliability, and generalization capabilities. For this project, model performance was primarily assessed using three widely accepted statistical metrics:

R-squared (R 
2
 ):

Explanation: R-squared, also known as the coefficient of determination, measures the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where 1 indicates that the model perfectly explains the variability of the response data around its mean.

Interpretation: A higher R-squared value indicates a better fit of the model to the data, implying that the independent variables are good predictors of the dependent variable. However, a high R-squared alone doesn't guarantee a good model, as it can be inflated by overfitting.

Mean Squared Error (MSE):

Explanation: MSE is the average of the squares of the errors. An "error" in this context is the difference between the actual observed value and the value predicted by the model. Squaring the errors ensures that positive and negative errors do not cancel out and gives more weight to larger errors.

Interpretation: MSE is always non-negative, and a value closer to zero indicates a better fit. It is sensitive to outliers due to the squaring of errors.

Root Mean Squared Error (RMSE):

Explanation: RMSE is the square root of the MSE. It is one of the most commonly used metrics for evaluating regression models.

Interpretation: RMSE is in the same units as the target variable (in this case, stock price), making it more interpretable than MSE. A lower RMSE indicates that the model's predictions are, on average, closer to the actual values.

Performance Summary:

Model

R-squared

MSE

RMSE

Linear Regression (OLS)

0.993

17.35

4.16

Lasso Regression

0.7598

33.88

5.82

Ridge Regression

0.7443

36.06

6.00

Elastic Net Regression

0.6638

35.00

5.91

Observation:
The OLS model demonstrated an exceptionally high R-squared on the in-sample data, suggesting a very strong fit to the historical data it was trained on. Its low MSE and RMSE further reinforce its accuracy in predicting prices within the training period.

While regularization (Lasso, Ridge, Elastic Net) is crucial for preventing overfitting and improving generalization to unseen data, their performance on the test set (or out-of-sample prediction for 2025) varied. The OLS model, even after feature selection, still showed a strong predictive capability for the 2025 period, indicating that the selected features were highly informative and the relationships were predominantly linear within the observed range. The higher error metrics for the regularized models suggest that for this specific dataset and chosen alpha values, the penalty might have been too aggressive, or the benefits of regularization were not fully realized without more extensive hyperparameter tuning.

Assumption Validation (for OLS)
Why Validate Assumptions?
Validating the assumptions of Linear Regression (specifically OLS) is a critical step to ensure that the model's estimates are unbiased, efficient, and that the statistical inferences (like p-values and confidence intervals) drawn from the model are valid. Violations of these assumptions can lead to misleading conclusions about the relationships between variables and the reliability of predictions.

Linearity between Dependent and Independent Variables:

Explanation: This assumption states that there must be a linear relationship between the predictor variables and the response variable. If the relationship is non-linear, a linear model will not accurately capture it.

Method: Visual inspection using scatter plots, particularly seaborn.pairplot, which generates scatter plots for all pairs of variables.

Outcome: Visual analysis indicated a strong linear relationship between AAPL prices and its lagged values (AAPL(t-1)), as well as between AAPL and the S&P 500 index (^GSPC(t-1)). This suggests that the core relationships the model is trying to capture are indeed linear.

Homoscedasticity (Constant Variance of Residuals):

Explanation: This assumption implies that the variance of the errors (residuals) should be constant across all levels of the independent variables. If the variance of the residuals increases or decreases with the predicted values, it indicates heteroscedasticity, which can lead to inefficient coefficient estimates and incorrect standard errors.

Method: Plotting the residuals (actual - predicted values) against the fitted (predicted) values.

Outcome: The scatter plot of residuals against fitted values showed a "tube-like" structure centered around zero, with no discernible pattern (e.g., a funnel shape). This visual evidence suggests that the variance of the residuals was relatively constant across all levels of the predicted values, thus meeting the homoscedasticity assumption.

Multicollinearity (Low Correlation between Independent Variables):

Explanation: Multicollinearity occurs when independent variables in a regression model are highly correlated with each other. High multicollinearity can make it difficult to interpret the individual impact of each predictor and can lead to unstable and unreliable coefficient estimates.

Method: Calculated the Variance Inflation Factor (VIF) for each independent variable. VIF quantifies how much the variance of an estimated regression coefficient is inflated due to multicollinearity.

Rule of Thumb: A VIF value greater than 5 or 10 is generally considered indicative of significant multicollinearity.

Outcome: VIF values for AAPL(t-1) and ^GSPC(t-1) were approximately 7.63. This indicates a moderate level of multicollinearity. While not ideal (ideally VIF < 5), it was considered manageable for this project. The exploration of regularization models (Lasso, Ridge, Elastic Net) was partly aimed at mitigating the effects of this multicollinearity.

Normality of Residuals:

Explanation: This assumption posits that the residuals of the regression model should be approximately normally distributed. While OLS estimates are still unbiased even if residuals are not normal, normality is important for the validity of hypothesis tests and confidence intervals for the coefficients.

Method: Visual inspection using a histogram of residuals and a Quantile-Quantile (QQ) plot. A histogram helps to visualize the distribution's shape, while a QQ plot compares the quantiles of the residuals to the quantiles of a theoretical normal distribution.

Outcome: The histogram of residuals approximated a bell-shaped curve, and the points on the QQ plot generally followed the 45-degree line. This visual evidence suggests that the residuals were approximately normally distributed, supporting the validity of statistical inferences.

Autocorrelation of Residuals (Low Correlation between Consecutive Residuals):

Explanation: This assumption, particularly important for time-series data, states that the residuals should be independent of each other. If residuals are correlated (autocorrelated), it means that the error at one time point is related to the error at a previous time point, violating the independence assumption. This can lead to underestimated standard errors and misleading p-values.

Method: Durbin-Watson (DW) test. The DW statistic ranges from 0 to 4; a value close to 2 indicates no autocorrelation, values less than 2 suggest positive autocorrelation, and values greater than 2 suggest negative autocorrelation.

Outcome: The Durbin-Watson statistic was approximately 1.04. A value significantly less than 2 (like 1.04) suggests positive autocorrelation. This is a known limitation when applying standard OLS to time series data without specific time series modeling techniques (e.g., ARIMA, GARCH models, or incorporating autoregressive terms directly). It implies that the model's errors are not entirely random over time, and there might be uncaptured temporal dependencies in the data.

Future Predictions (2025)
The trained OLS model (specifically the one after feature selection, which showed robust performance on the historical data) was utilized to predict Apple stock prices for the first quarter of 2025. This step demonstrates the practical application of the developed model for forecasting unseen data.

Prediction Period: The model generated daily price forecasts for January 1, 2025, to March 31, 2025.

Visual Outcome: Plots comparing the actual 2025 prices with the predicted prices from the OLS model showed a reasonable alignment. The model was able to capture the general trend and direction of the stock price movements, although, as expected, it did not perfectly predict every daily fluctuation. This visual representation provides an intuitive understanding of the model's out-of-sample forecasting ability.

Conclusion & Limitations
This project successfully implemented and evaluated various linear regression models for Apple stock price prediction, emphasizing a structured approach from data preparation to rigorous validation. The Ordinary Least Squares (OLS) model, particularly after judicious feature selection, demonstrated a high R-squared and reasonable Root Mean Squared Error (RMSE) for out-of-sample predictions, indicating its capability to capture significant price movements. Regularized models (Lasso, Ridge, Elastic Net) were also explored, offering valuable insights into mitigating overfitting and multicollinearity, although their specific performance was sensitive to hyperparameter choices.

Key Drawbacks and Limitations:

Linearity Assumption: A fundamental limitation of linear regression is its assumption of linearity. Stock prices are inherently non-linear and are influenced by complex, dynamic, and often non-linear relationships. Purely linear models may struggle to fully capture these intricate patterns, leading to residual errors.

Autocorrelation in Residuals: As indicated by the Durbin-Watson statistic (approx. 1.04), the residuals exhibited positive autocorrelation. This suggests that the model's errors are not truly independent over time, violating a key OLS assumption. For more accurate time-series forecasting, dedicated time-series models (e.g., ARIMA, GARCH for volatility, or models incorporating autoregressive components) would be more appropriate to explicitly model these temporal dependencies.

External Factors: The models developed here primarily rely on historical price data and technical indicators. They do not account for crucial qualitative and quantitative external factors that significantly impact stock prices, such as:

News Events: Geopolitical developments, company-specific news (earnings reports, product launches), and economic announcements.

Macroeconomic Indicators: Inflation rates, GDP growth, interest rate changes (beyond what might be implicitly captured by index movements).

Market Sentiment: Investor psychology, fear, and greed, which often drive irrational market behavior.

Black Swan Events: Unpredictable and rare events with severe consequences.

Market Efficiency: The Efficient Market Hypothesis (EMH), particularly its strong form, posits that all available information (public and private) is already reflected in stock prices. If EMH holds, consistently predicting future prices using only historical data is theoretically impossible. This project operates under the assumption that some inefficiencies or predictable patterns might exist, at least in the short term.

Overfitting (for OLS): While OLS showed a very high R-squared in-sample, it is inherently more prone to overfitting than regularized models, especially when the number of features is large relative to the number of observations, or when features are highly correlated. The slightly lower R-squared of Lasso and Ridge on the test set, despite their in-sample penalty, might indicate better generalization due to their regularization properties.

Feature Set: The chosen technical indicators and cross-market variables represent only a small subset of the vast number of potential features. More advanced feature engineering, incorporating alternative data sources (e.g., sentiment analysis from news, satellite imagery for economic activity), or using deep learning models capable of automatic feature extraction, could potentially improve predictions.

This project serves as a foundational exploration of regression techniques for stock prediction. For real-world quantitative trading and investment strategies, more sophisticated machine learning algorithms (e.g., Recurrent Neural Networks like LSTMs, Transformer models) and robust time-series models, coupled with extensive feature engineering, dynamic risk management, and consideration of market microstructure, would be necessary.

Technologies Used
Python: The core programming language used for all data manipulation, model implementation, and visualization.

yfinance: A powerful Python library for fetching historical market data from Yahoo Finance, enabling easy access to stock prices and indices.

pandas: An essential library for data manipulation, cleaning, and analysis, providing DataFrames for structured data handling.

numpy: The fundamental package for numerical computing in Python, used for array operations and mathematical functions.

scikit-learn: A comprehensive machine learning library used for implementing various regression models (Lasso, Ridge, Elastic Net) and for calculating key model evaluation metrics like R-squared, MSE, and RMSE.

statsmodels: A Python module that provides classes and functions for the estimation of many different statistical models, including Ordinary Least Squares (OLS) regression, and for performing statistical tests like VIF and Durbin-Watson.

matplotlib & seaborn: Popular Python libraries for creating static, interactive, and animated visualizations, used extensively for plotting stock prices, predictions, and residual diagnostics.

How to Run
To run this project, follow these steps:

Prerequisites:

Ensure Python 3.x is installed on your system.

Have Jupyter Notebook or JupyterLab installed, as the project is structured as a Jupyter Notebook.

Install Libraries:

Open your terminal or command prompt and run the following command to install all necessary Python libraries:

pip install yfinance pandas numpy scikit-learn statsmodels matplotlib seaborn

Clone the Repository:

Clone this GitHub repository to your local machine:

git clone [Your GitHub Repo URL Here]
cd [Your Repo Name]

(Replace [Your GitHub Repo URL Here] with the actual URL of your GitHub repository, and [Your Repo Name] with the name of your cloned repository directory.)

Open Jupyter Notebook:

Navigate to the project directory in your terminal and launch Jupyter Notebook:

jupyter notebook

Navigate and Run:

In the Jupyter Notebook interface that opens in your web browser, navigate to the project folder.

Open the rgmodel.ipynb file (or Quant (3).pdf if you've converted it back to .ipynb format).

Execute all cells in the notebook sequentially to reproduce the analysis, model training, and predictions.

Contributing
Contributions are highly valued and welcome! If you have suggestions for improvements, new features, bug fixes, or alternative modeling approaches, please feel free to:

Open an Issue: Describe the bug, enhancement, or question you have.

Submit a Pull Request: Fork the repository, make your changes, and submit a pull request for review.

Your contributions help make this project better!

License
This project is licensed under the MIT License. This is a permissive open-source license, meaning you are free to use, modify, and distribute the code for both commercial and non-commercial purposes, provided you include the original copyright and license notice.

