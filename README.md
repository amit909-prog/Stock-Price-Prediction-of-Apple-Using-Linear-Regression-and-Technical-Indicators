
# Apple Stock Price Prediction using Regression Models

This repository features a robust quantitative analysis project focused on predicting Apple Inc. (AAPL) stock prices. It meticulously implements and compares various linear regression models (OLS, Lasso, Ridge, Elastic Net), leveraging extensive historical data and advanced feature engineering with technical indicators and cross-market variables. The project emphasizes rigorous statistical assumption validation, detailed performance evaluation using R-squared, MSE, and RMSE, and provides out-of-sample forecasts for 2025. It serves as a practical demonstration of applying econometric and machine learning techniques to financial time series data, highlighting model strengths, limitations, and the complexities of market prediction.

---

## 🗂 Table of Contents

* [📌 Motivation](#motivation)
* [📦 Libraries and Tools](#libraries-and-tools)
* [📊 Data Collection and Preprocessing](#data-collection-and-preprocessing)
* [🧮 Feature Engineering](#feature-engineering)
* [🔍 Linear Regression](#linear-regression)
* [📅 Predicting Apple Stock Price in 2025](#predicting-apple-stock-price-in-2025)
* [📉 Lasso Regression](#lasso-regression)
* [📐 Ridge Regression](#ridge-regression)
* [🧬 Elastic Net Regression](#elastic-net-regression)
* [✅ Model Comparison and Evaluation](#model-comparison-and-evaluation)
* [📌 Conclusion](#conclusion)
* [⚠️ Limitations & Future Improvements](#limitations--future-improvements)
* [How to Run](#how-to-run)
* [Contributing](#contributing)
* [License](#license)

---

## 📌 Motivation

Stock price prediction stands as one of the most challenging yet highly sought-after endeavors in quantitative finance. The inherent volatility, non-linearity, and influence of countless macroeconomic, geopolitical, and company-specific factors make it a complex problem. However, even marginal improvements in forecasting accuracy can yield significant advantages in trading strategies and investment decisions. This project is motivated by the desire to explore the applicability and limitations of traditional econometric models, specifically various forms of linear regression, in capturing the dynamics of a highly liquid stock like Apple Inc. (AAPL). By rigorously applying these models and validating their underlying assumptions, we aim to gain a deeper understanding of their predictive power and the critical considerations necessary for robust financial forecasting.

## 📦 Libraries and Tools

This project is developed entirely in Python, leveraging its rich ecosystem of data science and machine learning libraries.

* **Python:** The core programming language used for all data manipulation, model implementation, and visualization.
* **`yfinance`:** A powerful and user-friendly Python library for fetching historical market data from Yahoo Finance. This tool enabled efficient and reliable access to daily stock prices and index data.
* **`pandas`:** An essential library for data manipulation, cleaning, and analysis. It provides high-performance, easy-to-use data structures like DataFrames, which were fundamental for organizing and processing our time-series data.
* **`numpy`:** The foundational package for numerical computing in Python. It was used for array operations, mathematical functions, and efficient numerical computations required for statistical modeling.
* **`scikit-learn`:** A comprehensive and widely used machine learning library. It provided the implementations for Lasso, Ridge, and Elastic Net regression models, as well as essential functions for model selection (e.g., `train_test_split`) and performance evaluation metrics.
* **`statsmodels`:** A Python module that complements `scikit-learn` by offering classes and functions for the estimation of many different statistical models. It was particularly valuable for Ordinary Least Squares (OLS) regression due to its detailed statistical summaries and for performing crucial assumption diagnostic tests (like VIF and Durbin-Watson).
* **`matplotlib` & `seaborn`:** These are powerful Python libraries for creating static, interactive, and animated visualizations. They were extensively used throughout the project for plotting historical stock prices, visualizing actual vs. predicted values, and generating diagnostic plots for assumption validation (histograms, scatter plots, QQ plots).

## 📊 Data Collection and Preprocessing

The quality and relevance of the input data are paramount to the success of any predictive model. For this project, a meticulous approach was taken to data acquisition and initial preprocessing.

* **Data Source:** Historical stock data was efficiently acquired using the `yfinance` library, which provides access to Yahoo Finance's extensive financial datasets.
* **Primary Stock:** **Apple Inc. (AAPL)** was selected as the primary stock for prediction. AAPL is known for its high trading volume and significant market capitalization, providing a robust and liquid dataset for analysis.
* **Cross-Market Variables:** To capture broader market influences and inter-stock correlations, daily closing prices were also collected for a set of relevant tickers:
    * **Amazon (AMZN):** A major tech peer.
    * **Microsoft (MSFT):** Another significant tech peer.
    * **Invesco QQQ Trust (QQQ):** An ETF tracking the NASDAQ-100 Index, representing the performance of large non-financial companies listed on the Nasdaq.
    * **S&P 500 Index (^GSPC):** A key benchmark for the overall U.S. stock market, providing a measure of general market sentiment.
* **Date Range:**
    * **Training & Testing Data:** Historical daily 'Close' prices were collected from **January 1, 2020, to December 31, 2024**. This five-year period encompasses various market cycles, including periods of economic growth, downturns, and significant volatility, allowing the models to learn from diverse conditions.
    * **Future Prediction Data:** A separate, unseen dataset was acquired for the period from **January 1, 2025, to March 31, 2025**. This out-of-sample data is crucial for evaluating the models' true forecasting capabilities on data they have not encountered during training.
* **Data Fetched:** For all specified tickers, only the daily 'Close' prices were extracted, as this is the primary variable of interest for our prediction task.
* **Data Cleaning and Preprocessing:**
    * Initially, only the 'Close' price for each ticker was extracted, as this is the primary variable of interest for our prediction task.
    * During feature engineering (specifically, operations like `shift()` for lagged values and `rolling()` for moving averages), `NaN` (Not a Number) values are typically introduced at the beginning of the time series. These `NaN` values were systematically handled by applying the `df.dropna()` method, which removes any rows containing missing values. This ensures that the dataset used for model training is complete and free from missing observations, which could otherwise lead to errors or biased results in the regression models.

## 🧮 Feature Engineering

**Why Feature Engineering?**
Raw closing prices alone are often insufficient for accurate stock price prediction due to the complex, dynamic, and often non-linear nature of financial markets. Feature engineering is the art and science of transforming raw data into a set of features that better represent the underlying patterns, trends, and relationships within the market. This step is crucial for enhancing the predictive power of machine learning models by providing them with more informative and discriminative inputs. By creating new variables from existing ones, we aim to capture aspects like momentum, trend, and inter-market dependencies that simple price series might overlook.

**Features Created:**

1.  **Lagged Closing Prices:** These features represent the stock's or index's closing price from previous trading days. They are fundamental in time-series forecasting, as today's price is highly correlated with and often influenced by recent past prices.
    * `AAPL(t-1)`: Apple's closing price from the immediate previous day. This serves as a direct autoregressive component, acknowledging the strong temporal correlation in stock prices.
    * `AMZN(t-1)`, `MSFT(t-1)`, `QQQ(t-1)`, `^GSPC(t-1)`: Lagged closing prices of major tech peers (Amazon, Microsoft) and key market indices (QQQ for NASDAQ-100, ^GSPC for S&P 500). These act as **cross-market variables**, capturing broader market sentiment, sector-specific movements, and potential correlations that might influence AAPL's price. For instance, a general upward trend in the tech sector or the overall market could indicate a positive outlook for AAPL.

2.  **Moving Averages (Technical Indicators):** Moving averages are widely used technical indicators that smooth out price data over a specified period. They help in identifying trends and reducing noise from short-term price fluctuations.
    * `AAPL_MA_5`, `AMZN_MA_5`, `MSFT_MA_5`, `QQQ_MA_5`, `^GSPC_MA_5`: These represent the 5-day simple moving averages for each respective stock/index. A 5-day MA is a short-term indicator, useful for understanding recent price momentum. For example, if the current price is above its 5-day MA, it might suggest an upward short-term trend.

**Target Variable:**
* `Target`: This is the dependent variable that our models are designed to predict. It is defined as the **next day's Apple closing price (`df['AAPL'].shift(-1)`)**. This "shifted" value ensures that our models are trained to forecast a future price, not merely to fit to the current day's price, thus making the prediction task realistic.

**Data Cleaning Post-Engineering:**
After creating these features, any rows containing `NaN` values (which inevitably arise at the beginning of the dataset due to the `shift()` and `rolling()` operations, as there's no prior data to calculate the lagged values or initial moving averages) were systematically removed using `df.dropna()`. This ensures that the dataset used for subsequent model training and evaluation is complete and free from missing observations, which could otherwise lead to errors or biased results in the regression models.

## 🔍 Linear Regression

Linear regression is a fundamental statistical model that establishes a linear relationship between a dependent variable (the target) and one or more independent variables (features). This project explores several variants of linear regression, each with unique properties and applications. The data was split into a training set (95%) and a testing set (5%) using `shuffle=False` to preserve the time-series order, which is crucial for financial data.

### Ordinary Least Squares (OLS) Regression

* **Description:** Ordinary Least Squares (OLS) is the most basic and widely used form of linear regression. Its objective is to find the line (or hyperplane in multiple dimensions) that best fits the data by minimizing the sum of the squared differences between the observed values of the dependent variable and the values predicted by the model. Under certain assumptions, OLS provides the Best Linear Unbiased Estimator (BLUE) of the regression coefficients.
* **Implementation:** We utilized `statsmodels.api.OLS` for its comprehensive statistical summary. `statsmodels` provides detailed output including coefficient estimates, standard errors, t-statistics, p-values, R-squared, and various diagnostic statistics, which are excellent for understanding the underlying relationships and validating assumptions.
* **Initial Features & Outcome:** All 10 engineered features (5 lagged prices and 5 moving averages) were initially included in the OLS model.
    * **Initial R-squared:** `0.993`
    * **Explanation:** This exceptionally high R-squared value (0.993) suggests that the initial OLS model explains approximately 99.3% of the variance in the next-day AAPL price within the training data. While this indicates a very strong fit to the historical data, such high in-sample R-squared values can sometimes be a red flag for overfitting, especially with highly correlated time-series data.
* **Feature Selection Strategy:** After the initial OLS analysis, we examined the p-values associated with each feature's coefficient. Features with high p-values (typically > 0.05), indicating that their coefficients are not statistically significant, were considered for removal. The notebook specifically highlights `AAPL(t-1)` and `^GSPC(t-1)` as statistically significant features after this process.

### Lasso Regression

* **Description:** Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a linear regression model that incorporates **L1 regularization**. This technique modifies the OLS objective function by adding a penalty term that is proportional to the absolute value of the magnitude of the regression coefficients.
    * **Key Property:** The unique and powerful property of Lasso is its ability to shrink some coefficients exactly to zero. This means Lasso effectively performs **automatic feature selection**, making it highly valuable when dealing with a large number of features or when multicollinearity is suspected. It forces less important features out of the model entirely.
* **Implementation:** We used `sklearn.linear_model.Lasso(alpha=10)`. The `alpha` parameter is a hyperparameter that controls the strength of the regularization penalty. A higher `alpha` value imposes a stronger penalty, leading to more coefficients being shrunk to zero.
* **Outcome (on Test Data):**
    * R-squared: `0.7598`
    * MSE: `33.88`
    * RMSE: `5.82`
* **Drawbacks:**
    * **Bias-Variance Trade-off:** While Lasso successfully performs feature selection and helps reduce overfitting (by reducing model complexity and variance), it introduces a slight bias into the model's estimates. This can lead to a lower R-squared on the training data compared to an unregularized OLS model, as seen in our results (0.7598 vs. 0.992).
    * **Feature Selection in Correlated Groups:** When a group of features are highly correlated, Lasso tends to pick only one feature from the group and shrink the others to zero, which might not always be ideal if all correlated features carry some predictive information.

### Ridge Regression

* **Description:** Ridge Regression is another variant of linear regression that utilizes **L2 regularization**. It modifies the OLS objective function by adding a penalty term proportional to the square of the magnitude of the regression coefficients.
    * **Key Property:** This penalty shrinks the coefficients towards zero but rarely makes them exactly zero. Ridge Regression is particularly effective at handling **multicollinearity** by distributing the impact of correlated features across all of them, preventing any single feature from dominating due to high correlation with others. It stabilizes coefficient estimates in the presence of multicollinearity.
* **Implementation:** We implemented Ridge Regression using `sklearn.linear_model.Ridge(alpha=10)`. Similar to Lasso, `alpha` dictates the strength of the regularization.
* **Outcome (on Test Data):**
    * R-squared: `0.7443`
    * MSE: `36.06`
    * RMSE: `6.00`
* **Drawbacks:**
    * **No Feature Selection:** Unlike Lasso, Ridge does not perform automatic feature selection; it only shrinks coefficients, meaning all features remain in the model. This can be a disadvantage in very high-dimensional datasets where many features might be irrelevant.
    * **Bias Introduction:** Ridge also introduces a slight bias into the model's estimates to reduce variance, which can lead to a slightly lower R-squared on the training data compared to an unregularized OLS.

### Elastic Net Regression

* **Description:** Elastic Net Regression is a powerful and versatile hybrid linear regression model that combines both **L1 (Lasso) and L2 (Ridge) regularization**. It aims to leverage the strengths of both techniques: Lasso's ability to perform feature selection (by creating sparse models) and Ridge's capacity to handle multicollinearity by shrinking coefficients.
    * **Key Property:** Elastic Net is particularly useful when there are multiple correlated features. In such cases, Lasso might arbitrarily select one feature from a correlated group, while Elastic Net tends to select groups of correlated variables together, making it a more robust choice.
* **Implementation:** We used `sklearn.linear_model.ElasticNet(alpha=1, l1_ratio=0.5)`.
    * `alpha`: The overall regularization strength.
    * `l1_ratio`: A mixing parameter (between 0 and 1) that controls the balance between L1 and L2 penalties. A value of 0.5 indicates an equal mix of Lasso and Ridge penalties.
* **Outcome (on Test Data):**
    * R-squared: `0.6638`
    * MSE: `35.00`
    * RMSE: `5.91`
* **Drawbacks:**
    * **Hyperparameter Tuning:** Elastic Net has two hyperparameters (`alpha` and `l1_ratio`) that require careful tuning. Its performance can vary significantly depending on these choices, and finding the optimal combination often requires extensive cross-validation. The lower R-squared observed in this specific run suggests that the chosen regularization parameters might have been too strong for the dataset's characteristics, potentially leading to an underfit model.
    * **Complexity:** Being a combination of two methods, Elastic Net can be more computationally intensive and complex to interpret than pure Lasso or Ridge.

## ✅ Model Comparison and Evaluation

Evaluating the performance of predictive models is crucial to understand their accuracy, reliability, and generalization capabilities. For this project, model performance was primarily assessed using three widely accepted statistical metrics:

* **R-squared ($R^2$):**
    * **Explanation:** R-squared, also known as the coefficient of determination, measures the proportion of the variance in the dependent variable that can be predicted from the independent variables. It ranges from 0 to 1, where 1 indicates that the model perfectly explains the variability of the response data around its mean.
    * **Interpretation:** A higher R-squared value indicates a better fit of the model to the data, implying that the independent variables are good predictors of the dependent variable. However, a high R-squared alone doesn't guarantee a good model, as it can be inflated by overfitting.
* **Mean Squared Error (MSE):**
    * **Explanation:** MSE is the average of the squares of the errors. An "error" in this context is the difference between the actual observed value and the value predicted by the model. Squaring the errors ensures that positive and negative errors do not cancel out and gives more weight to larger errors.
    * **Interpretation:** MSE is always non-negative, and a value closer to zero indicates a better fit. It is sensitive to outliers due to the squaring of errors.
* **Root Mean Squared Error (RMSE):**
    * **Explanation:** RMSE is the square root of the MSE. It is one of the most commonly used metrics for evaluating regression models.
    * **Interpretation:** RMSE is in the same units as the target variable (in this case, stock price), making it more interpretable than MSE. A lower RMSE indicates that the model's predictions are, on average, closer to the actual values.

**Performance Summary on Test Data (Out-of-Sample):**

| **Model** | **R-squared** | **MSE** | **RMSE** |
| :-------------------- | :-------- | :-------- | :------- |
| Linear Regression (OLS) | `0.993` | `17.35` | `4.16` |
| Lasso Regression | `0.7598` | `33.88` | `5.82` |
| Ridge Regression | `0.7443` | `36.06` | `6.00` |
| Elastic Net Regression | `0.6638` | `35.00` | `5.91` |

**Analysis:**

* **OLS Dominance:** The **Ordinary Least Squares (OLS) model** stands out with an exceptionally high R-squared (0.992) and the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) on the test data. This indicates that it provides the best fit and highest predictive accuracy among the tested models for this specific dataset. Its RMSE of approximately $4.16 means that, on average, its predictions were within $4.16 of the actual Apple stock price for the out-of-sample period.
* **Regularized Models' Performance:** Lasso, Ridge, and Elastic Net, while valuable for mitigating overfitting and multicollinearity in general, showed lower R-squared values and higher error metrics compared to OLS. This suggests that for this particular dataset, the strong linear relationships captured by OLS (even after feature selection) were more dominant, and the regularization penalties might have been too aggressive, potentially leading to a slight underfitting or removal of genuinely informative features.
* **Trade-offs:** The comparison highlights the classic bias-variance trade-off. OLS, being less biased, might capture the training data's patterns more closely (leading to high R-squared). Regularized models introduce bias to reduce variance, aiming for better generalization. In this case, OLS with selected features appears to generalize well to the immediate out-of-sample data.

## Assumption Validation (for OLS)

**Why Validate Assumptions?**
Validating the assumptions of Linear Regression (specifically OLS) is a critical step to ensure that the model's estimates are unbiased, efficient, and that the statistical inferences (like p-values and confidence intervals) drawn from the model are valid. Violations of these assumptions can lead to misleading conclusions about the relationships between variables and the reliability of predictions.

1.  **Linearity between Dependent and Independent Variables:**
    * **Explanation:** This assumption states that there must be a linear relationship between the predictor variables and the response variable. If the relationship is non-linear, a linear model will not accurately capture it.
    * **Method:** Visual inspection using scatter plots, particularly `seaborn.pairplot`, which generates scatter plots for all pairs of variables.
    * **Outcome:** Visual analysis indicated a strong linear relationship between AAPL prices and its lagged values (`AAPL(t-1)`), as well as between AAPL and the S&P 500 index (`^GSPC(t-1)`). This suggests that the core relationships the model is trying to capture are indeed linear.
2.  **Homoscedasticity (Constant Variance of Residuals):**
    * **Explanation:** This assumption implies that the variance of the errors (residuals) should be constant across all levels of the independent variables. If the variance of the residuals increases or decreases with the predicted values, it indicates heteroscedasticity, which can lead to inefficient coefficient estimates and incorrect standard errors.
    * **Method:** Plotting the residuals (actual - predicted values) against the fitted (predicted) values.
    * **Outcome:** The scatter plot of residuals against fitted values showed a "tube-like" structure centered around zero, with no discernible pattern (e.g., a funnel shape). This visual evidence suggests that the variance of the residuals was relatively constant across all levels of the predicted values, thus meeting the homoscedasticity assumption.
3.  **Multicollinearity (Low Correlation between Independent Variables):**
    * **Explanation:** Multicollinearity occurs when independent variables in a regression model are highly correlated with each other. High multicollinearity can make it difficult to interpret the individual impact of each predictor and can lead to unstable and unreliable coefficient estimates.
    * **Method:** Calculated the Variance Inflation Factor (VIF) for each independent variable. VIF quantifies how much the variance of an estimated regression coefficient is inflated due to multicollinearity.
    * **Rule of Thumb:** A VIF value greater than 5 or 10 is generally considered indicative of significant multicollinearity.
    * **Outcome:** VIF values for `AAPL(t-1)` and `^GSPC(t-1)` were approximately `7.63`. This indicates a moderate level of multicollinearity. While not ideal (ideally VIF < 5), it was considered manageable for this project. The exploration of regularization models (Lasso, Ridge, Elastic Net) was partly aimed at mitigating the effects of this multicollinearity.
4.  **Normality of Residuals:**
    * **Explanation:** This assumption posits that the residuals of the regression model should be approximately normally distributed. While OLS estimates are still unbiased even if residuals are not normal, normality is important for the validity of hypothesis tests and confidence intervals for the coefficients.
    * **Method:** Visual inspection using a histogram of residuals and a Quantile-Quantile (QQ) plot. A histogram helps to visualize the distribution's shape, while a QQ plot compares the quantiles of the residuals to the quantiles of a theoretical normal distribution.
    * **Outcome:** The histogram of residuals approximated a bell-shaped curve, and the points on the QQ plot generally followed the 45-degree line. This visual evidence suggests that the residuals were approximately normally distributed, supporting the validity of statistical inferences.
5.  **Autocorrelation of Residuals (Low Correlation between Consecutive Residuals):**
    * **Explanation:** This assumption, particularly important for time-series data, states that the residuals should be independent of each other. If residuals are correlated (autocorrelated), it means that the error at one time point is related to the error at a previous time point, violating the independence assumption. This can lead to underestimated standard errors and misleading p-values.
    * **Method:** Durbin-Watson (DW) test. The DW statistic ranges from 0 to 4; a value close to 2 indicates no autocorrelation, values less than 2 suggest positive autocorrelation, and values greater than 2 suggest negative autocorrelation.
    * **Outcome:** The Durbin-Watson statistic was approximately `1.04`. A value significantly less than 2 (like 1.04) suggests positive autocorrelation. **This is a known limitation when applying standard OLS to time series data without specific time series modeling techniques (e.g., ARIMA, GARCH models, or incorporating autoregressive terms directly).** It implies that the model's errors are not entirely random over time, and there might be uncaptured temporal dependencies in the data.

## 📅 Predicting Apple Stock Price in 2025

The trained OLS model (specifically the one after feature selection, which showed robust performance on the historical data) was utilized to predict Apple stock prices for the first quarter of 2025. This step demonstrates the practical application of the developed model for forecasting unseen data.

* **Prediction Period:** The model generated daily price forecasts for **January 1, 2025, to March 31, 2025**. This period represents entirely unseen data for the model, providing a realistic assessment of its forecasting ability.
* **Visual Outcome:** Plots comparing the actual 2025 prices with the predicted prices from the OLS model showed a **reasonable alignment**. The model was able to capture the general trend and direction of the stock price movements, although, as expected, it did not perfectly predict every daily fluctuation (which is expected in volatile markets), its ability to follow the overall trajectory demonstrates its practical utility in identifying potential future price paths. This visual representation offers an intuitive understanding of the model's out-of-sample forecasting capability.

## 📌 Conclusion

This project successfully implemented and evaluated various linear regression models for Apple stock price prediction, emphasizing a structured approach from data acquisition and feature engineering to rigorous assumption validation and comprehensive performance assessment. The **Ordinary Least Squares (OLS) model**, particularly after judicious feature selection, emerged as the top performer, demonstrating an exceptionally high R-squared and the lowest Root Mean Squared Error (RMSE) on out-of-sample predictions. This indicates its strong capability to capture significant linear patterns within the financial time series. While regularized models (Lasso, Ridge, Elastic Net) offer valuable tools for mitigating overfitting and multicollinearity, the OLS model, in this specific context, proved to be highly effective. The project provides a foundational understanding of applying econometric and machine learning techniques to financial data, highlighting the importance of data quality, feature engineering, and assumption validation in building predictive models.

## ⚠️ Limitations & Future Improvements

While the project provides valuable insights, it's crucial to acknowledge its limitations and identify avenues for future improvements to enhance model robustness and accuracy.

**Key Drawbacks and Limitations:**

* **Linearity Assumption:** A fundamental limitation of all linear regression models is their inherent assumption of linearity. Stock prices are influenced by complex, dynamic, and often non-linear relationships that a purely linear model may struggle to fully capture. This can lead to residual errors that represent unmodeled non-linearities.
* **Autocorrelation in Residuals:** As indicated by the Durbin-Watson statistic (approx. 1.04), the residuals exhibited positive autocorrelation. This suggests that the model's errors are not truly independent over time, violating a key OLS assumption. For more accurate time-series forecasting, dedicated time-series models (e.g., ARIMA, GARCH for volatility, or models incorporating autoregressive components) would be more appropriate to explicitly model these temporal dependencies.
* **Exclusion of External Factors:** The current models primarily rely on historical price data and technical indicators. They do not account for crucial qualitative and quantitative external factors that significantly impact stock prices, such as:
    * **News Events:** Geopolitical developments, company-specific news (e.g., earnings reports, product launches), and economic announcements.
    * **Macroeconomic Indicators:** Broader economic health indicators beyond just market indices (e.g., unemployment rates, consumer confidence, industrial production).
    * **Market Sentiment:** Investor psychology, fear, and greed, which often drive irrational or herd-like market behavior.
    * **Black Swan Events:** Unpredictable and rare events with severe consequences (e.g., pandemics, major natural disasters, sudden policy shifts).
* **Efficient Market Hypothesis (EMH):** The project operates under the assumption that some inefficiencies or predictable patterns might exist in stock prices. However, the EMH (especially its strong form) suggests that all available information is already reflected in prices, making consistent prediction from historical data alone theoretically impossible in the long run.
* **Overfitting (for OLS):** Despite its high R-squared, OLS is inherently more prone to overfitting than regularized models, especially when the number of features is large relative to the number of observations, or when features are highly correlated. While feature selection was applied, the risk remains.
* **Feature Set Scope:** The chosen technical indicators and cross-market variables represent only a small subset of the vast number of potential features that could influence stock prices.

**Future Improvements:**

* **Advanced Time-Series Models:** Implement and compare more sophisticated time-series models like ARIMA, GARCH (for volatility modeling), or Prophet to better capture temporal dependencies and heteroscedasticity.
* **Machine Learning Algorithms:** Explore non-linear machine learning models such as:
    * **Random Forests/Gradient Boosting Machines (GBMs):** Ensemble methods that can capture complex non-linear relationships and interactions between features.
    * **Support Vector Machines (SVMs):** Effective for both linear and non-linear regression tasks.
    * **Deep Learning (RNNs, LSTMs, Transformers):** These neural network architectures are specifically designed for sequential data and can learn highly complex patterns from time series.
* **Sentiment Analysis:** Integrate sentiment data derived from news articles, social media, or financial reports to capture the impact of market sentiment on stock prices.
* **Fundamental Data:** Incorporate company fundamental data (e.g., P/E ratios, revenue growth, debt-to-equity) for long-term prediction.
* **Hyperparameter Tuning:** Systematically tune hyperparameters for regularized models (Lasso, Ridge, Elastic Net) and other machine learning models using techniques like GridSearchCV or RandomizedSearchCV to optimize their performance.
* **Volatility Forecasting:** Develop dedicated models for predicting volatility, as volatility is a key input for options pricing and risk management.
* **Risk Management & Portfolio Optimization:** Extend the project to include components for portfolio optimization based on predicted returns and risks, and implement backtesting strategies.




