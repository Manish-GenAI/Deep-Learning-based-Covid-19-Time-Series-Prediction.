A 7-day forecast horizon is used; performance is measured in **RMSE** (Root Mean Square Error).  

---

## 2. Related Work

### 2.1. Classical Time Series in Infectious Diseases
- **Leptospirosis & Climate**: Models linking monthly leptospirosis incidence to rainfall and temperature (Chadsuthi et al., 2012).  
- **Malaria & ENSO**: Time‐series correlations between Plasmodium falciparum cases and El Niño Southern Oscillation (Hanf et al., 2011).  
- **Influenza Forecasting**:  
  - ARIMA to predict monthly influenza incidence in China (Song et al., 2016).  
  - SARIMA + Internet search data to forecast US flu (Zhang et al., 2019).  
  - TBAT for complex seasonal influenza cycles (De Livera et al., 2011).  
  - Twitter‐based real‐time flu‐spread estimation (Lee et al., 2017).  

### 2.2. COVID-19 Forecasting
- **China‐focused Models**:  
  - Phenomenological models for province‐level cumulative cases (Roosa et al., 2020).  
  - SEIR + AI hybrid using migration data (Yang et al., 2020).  
  - Stacked autoencoder for Chinese confirmed case forecasts (Zheng et al., 2020).  
  - Adaptive neuro‐fuzzy + metaheuristic for 10-day confirmed case forecasts (Al-qaness et al., 2020).  
- **Global Predictions**:  
  - Simple exponential smoothing for global case counts (Petropoulos & Makridakis, 2020).  
  - IHME: Statistical modeling for US healthcare resource utilization (beds, ICU, ventilators).  
  - R0 Estimation: Markov Chain Monte Carlo (Wu et al., 2020), SIRD models (Anastassopoulou et al., 2020).  
  - Diamond Princess outbreak analysis (Zhang et al., 2020).  

Our work expands upon these by:
1. Focusing on **active‐case percentage** (active cases ÷ population)  
2. Comparing **six** diverse methods on the same 10-country dataset  
3. Emphasizing simple statistical models vs. cutting‐edge deep‐learning approaches  

---

## 3. Time Series Models

Below is a concise overview of each method used in this study. References to original papers are provided in [References](#references).

### 3.1. ARIMA (Auto‐Regressive Integrated Moving Average)
- **Core Idea:** Model current value as a linear combination of past values (AR), differencing (I), and past forecast errors (MA).  
- **Strengths:**  
  - High interpretability (parameters have clear statistical meaning).  
  - Box–Jenkins methodology automates order selection (p, d, q).  
  - Handles non‐stationary data via differencing.  
- **Limitations:**  
  - Unable to model nonlinear dependencies.  
  - Assumes linear relationships only.  

---

### 3.2. Holt–Winters Additive Model (HWAAS)
- **Core Idea:** Exponential smoothing with additive trend and additive seasonality.  
- **Components:**  
  1. **Level** (ℓ)  
  2. **Trend** (b)  
  3. **Seasonality** (s)  
- **Formula (Additive):**  
  1. Level: 
     \[
       \ell_t = \alpha \left(y_t - s_{t - m}\right) + (1 - \alpha)(\ell_{t-1} + b_{t-1})
     \]
  2. Trend:
     \[
       b_t = \beta \left(\ell_t - \ell_{t-1}\right) + (1 - \beta) b_{t-1}
     \]
  3. Seasonality:
     \[
       s_t = \gamma \left(y_t - \ell_t\right) + (1 - \gamma) s_{t - m}
     \]
  4. Forecast (h steps ahead):
     \[
       \hat{y}_{t+h} = \ell_t + h\,b_t + s_{t + h - m\,\lfloor (h-1)/m \rfloor}
     \]
  - α, β, γ ∈ (0, 1): smoothing parameters  
  - m: seasonal period (e.g., m=7 for weekly seasonality)  
- **Strengths:**  
  - Handles both trend and seasonality additively.  
  - Simpler than ARIMA; fewer parameters to tune.  
- **Limitations:**  
  - Less accurate on average than Box–Jenkins ARIMA for some datasets.  
  - Sensitive to initial values and outliers.  

---

### 3.3. TBAT (Trigonometric Seasonal, Box–Cox, ARMA, Trend)
- **Core Idea:** Decompose a time series into trend + multiple seasonalities using a trigonometric formulation, apply Box–Cox transform, and model residuals with ARMA.  
- **Components:**  
  1. **Box–Cox Transform:** Stabilize variance → 
     \[
       y_t^{(\lambda)} = 
       \begin{cases}
         \frac{y_t^\lambda - 1}{\lambda}, & \lambda \neq 0, \\
         \ln(y_t), & \lambda = 0.
       \end{cases}
     \]
  2. **Trend:** Modeled by local linear or segment‐wise linear trend.  
  3. **Seasonality:** Trigonometric representation for any (including non‐integer) seasonal frequency:
     \[
       S_{t} = \sum_{k=1}^{K} \left( a_{k} \cos\left(\tfrac{2\pi k\,t}{m}\right) + b_{k} \sin\left(\tfrac{2\pi k\,t}{m}\right) \right)
     \]
     - K: number of Fourier terms; m: seasonal period  
  4. **ARMA Residuals:** Auto‐Regressive + Moving Average model on residual errors  
- **Strengths:**  
  - Can capture complex/multiple seasonalities (e.g., weekly, yearly, sub‐daily).  
  - Handles nonlinearity via Box–Cox.  
  - Identifies hidden seasonal components not obvious in raw data.  
- **Limitations:**  
  - Larger parameter space → more computationally expensive (but optimized for ML).  
  - Requires tuning of Fourier terms (K) and Box–Cox parameter (λ).  

---

### 3.4. Prophet
- **Core Idea:** Generalized additive model (GAM) with decomposable trend, seasonality, and holiday effects; designed for business time series.  
- **Model Definition:**  
  \[
    y(t) = g(t) + s(t) + h(t) + \epsilon_t  
  \]
  1. **Trend (g)**: 
     - **Piecewise linear** with change points, or  
     - **Logistic growth** (saturating)  
  2. **Seasonality (s)**: Fourier series up to order N:
     \[
       s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\tfrac{2\pi n\,t}{P}\right) + b_n \sin\left(\tfrac{2\pi n\,t}{P}\right) \right)
     \]
     - P: seasonal period (e.g., 365 days, 7 days)  
  3. **Holiday Effects (h)**:  
     - User‐provided list of dates with additive indicators  
  4. **Error (ε)**: Gaussian noise  
- **Strengths:**  
  - User‐friendly: works well “out of the box” with default parameters.  
  - Explicitly handles missing data and trend change points.  
  - Incorporates known holidays/events easily.  
- **Limitations:**  
  - Designed for business/seasonal data; may underperform on epidemiological data without strong seasonality.  
  - Less granular control over model internals compared to ARIMA/TBAT.  

---

### 3.5. DeepAR
- **Core Idea:** Probabilistic forecasting using an auto‐regressive RNN (LSTM) to model future distribution, trained on many time series simultaneously.  
- **Architecture:**  
  1. **Input:** Previous target values (t−1, t−2, …) and covariates (optional).  
  2. **LSTM:** Encodes historical context.  
  3. **Output Layer:** Parameterizes a chosen likelihood (e.g., Gaussian, Negative Binomial) at each time step.  
  4. **Training Objective:** Maximize log‐likelihood of observed data under predicted distribution.  
- **Strengths:**  
  - Produces full probabilistic forecasts (quantiles, prediction intervals).  
  - Can leverage cross‐series learning: trains on multiple related time series to improve accuracy.  
  - Flexible: supports arbitrary likelihood functions (e.g., Student-T, Log-Norm, Poisson).  
- **Limitations:**  
  - Requires substantial data to train effectively; may overfit on small datasets.  
  - Less interpretable than classical models; “black box” behavior.  
  - Hyperparameter tuning (layers, cells, learning rate) is nontrivial.  

---

### 3.6. N-Beats
- **Core Idea:** Deep fully‐connected residual architecture that explicitly separates trend and seasonality via basis expansions, enabling interpretability and state‐of‐the‐art accuracy.  
- **Architecture Overview (Block):**  
  1. Input window of length L.  
  2. **Backcast:** Model reconstructs part of input (removes its contribution).  
  3. **Forecast:** Model outputs future horizon of length H.  
  4. Blocks are stacked in two “stacks”:  
     - **Trend Stack:** Models a polynomial basis to capture trend.  
     - **Seasonality Stack:** Models seasonality via Fourier basis.  
  5. **Residual Connections (Backcast − forecast):** Each block removes its backcast from the input before passing to the next block (hierarchical residual).  
- **Strengths:**  
  - Interpretable: separate trend vs. seasonality outputs.  
  - State‐of‐the‐art performance on M4/M3 forecasting competitions.  
  - Fast to train; uses simple fully connected layers and ReLU.  
- **Limitations:**  
  - Requires careful choice of stack depth and width.  
  - Still a “black box” to some extent (deep fully connected nets).  

---

## 4. Data Description

### 4.1. Datasets Used
1. **Novel Corona Virus 2019 Dataset** (Kaggle)  
   - Daily time series of  
     - Confirmed cases  
     - Recovered cases  
     - Deaths  
   - Source: [Kaggle: Novel Corona Virus 2019 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india) _(example link—replace with actual)_  
2. **Population by Country** (Kaggle)  
   - 2019 population estimates for all countries  
   - Source: [Kaggle: World Population by Country](https://www.kaggle.com/datasets/fernandol/countries-of-the-world) _(example link—replace with actual)_  

### 4.2. Active Case Percentage Calculation
- **Active Cases (daily):**  
  \[
    \text{Active}_t = \text{Confirmed}_t \;-\; \text{Recovered}_t \;-\; \text{Deaths}_t
  \]
- **Active Case Percentage (daily):**  
  \[
    \text{PctActive}_t = \frac{\text{Active}_t}{\text{Population}_{\text{country}}}
  \]
- **Ten Countries Selected (highest total confirmed as of May 4, 2020):**  
  1. USA  
  2. Spain  
  3. Italy  
  4. UK  
  5. France  
  6. Russia  
  7. Germany  
  8. Turkey  
  9. Brazil  
  10. Iran  

### 4.3. Data Preprocessing & Splits
- **Total Instances per Country:**  
  - From first reported case → May 4, 2020  
  - 104 daily observations (approx.)  
- **Train / Validation / Test Splits:**  
  - **Training:** First 72 days (percentage series)  
  - **Validation:** Next 25 days  
  - **Test (Forecast Horizon):** Last 7 days (used only for final RMSE calculation)  
- **Scaling:** Each country’s series is already a fraction (active cases ÷ population), so no further scaling was necessary.

---

## 5. Experiments & Results

### 5.1. Model Training & Evaluation
1. **Training**  
   - Fit each model on the 72 training days of `PctActive_t`.  
   - Hyperparameters:  
     - **ARIMA:** Automatic order selection via AICc (p, d, q).  
     - **HWAAS:** Seasonal period m=7; smoothers α, β, γ tuned via validation RMSE.  
     - **TBAT:** Fourier terms up to order K=2 (weekly seasonality), Box–Cox λ tuned on validation set, ARMA orders selected via BIC.  
     - **Prophet:** Default growth (“logistic”), seasonalities (yearly, weekly), no holidays.  
     - **DeepAR:**  
       - 2 LSTM layers, 64 cells each  
       - Gaussian likelihood  
       - Learning rate: 1e−3, batch=32, epochs=100  
     - **N-Beats:**  
       - 3 blocks per stack (trend & seasonality), width=256  
       - ReLU activations, Adam optimizer (lr=1e−3), epochs=100  
2. **Validation**  
   - Evaluate on 25 validation days; tune hyperparameters (where applicable) to minimize RMSE.  
3. **Test (7-Day Forecast)**  
   - Generate 7-day forecasts, compute RMSE against actual `PctActive_t` for Days (98 – 104).  
   - Report RMSE per model per country.

---

### 5.2. Performance Comparison (RMSE)

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Country</th>
      <th colspan="5" align="center">Statistical Models</th>
      <th colspan="2" align="center">Deep Learning Models</th>
    </tr>
    <tr>
      <th align="center">ARIMA</th>
      <th align="center">Prophet</th>
      <th align="center">HWAAS</th>
      <th align="center">TBAT</th>
      <th align="center">N-Beats</th>
      <th align="center">DeepAR<br>(GluonTS)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>USA</td>
      <td align="center">0.007421</td>
      <td align="center">0.013877</td>
      <td align="center">0.172957</td>
      <td align="center">0.009873</td>
      <td align="center">0.036958</td>
      <td align="center">0.044805</td>
    </tr>
    <tr>
      <td>Spain</td>
      <td align="center">0.080094</td>
      <td align="center">0.065433</td>
      <td align="center">0.031497</td>
      <td align="center">0.029295</td>
      <td align="center">0.050492</td>
      <td align="center">0.108842</td>
    </tr>
    <tr>
      <td>Italy</td>
      <td align="center">0.005628</td>
      <td align="center">0.019217</td>
      <td align="center">0.006616</td>
      <td align="center">0.005810</td>
      <td align="center">0.008645</td>
      <td align="center">0.043551</td>
    </tr>
    <tr>
      <td>UK</td>
      <td align="center">0.005484</td>
      <td align="center">0.007634</td>
      <td align="center">0.004366</td>
      <td align="center">0.004310</td>
      <td align="center">0.037623</td>
      <td align="center">0.046134</td>
    </tr>
    <tr>
      <td>France</td>
      <td align="center">0.060824</td>
      <td align="center">0.044482</td>
      <td align="center">0.011007</td>
      <td align="center">0.007003</td>
      <td align="center">0.004220</td>
      <td align="center">0.010549</td>
    </tr>
    <tr>
      <td>Germany</td>
      <td align="center">0.006431</td>
      <td align="center">0.037139</td>
      <td align="center">0.004586</td>
      <td align="center">0.003389</td>
      <td align="center">0.013192</td>
      <td align="center">0.057523</td>
    </tr>
    <tr>
      <td>Russia</td>
      <td align="center">0.001536</td>
      <td align="center">0.014681</td>
      <td align="center">0.002295</td>
      <td align="center">0.002193</td>
      <td align="center">0.027078</td>
      <td align="center">0.034479</td>
    </tr>
    <tr>
      <td>Turkey</td>
      <td align="center">0.004442</td>
      <td align="center">0.044595</td>
      <td align="center">0.000887</td>
      <td align="center">0.001946</td>
      <td align="center">0.018265</td>
      <td align="center">0.093839</td>
    </tr>
    <tr>
      <td>Brazil</td>
      <td align="center">0.004194</td>
      <td align="center">0.009279</td>
      <td align="center">0.005717</td>
      <td align="center">0.005621</td>
      <td align="center">0.010870</td>
      <td align="center">0.002836</td>
    </tr>
    <tr>
      <td>Iran</td>
      <td align="center">0.002628</td>
      <td align="center">0.016281</td>
      <td align="center">0.001046</td>
      <td align="center">0.000425</td>
      <td align="center">0.003745</td>
      <td align="center">0.002277</td>
    </tr>
  </tbody>
</table>

**Key Observations:**
- **Best Performing Model (lowest RMSE)** in each country (bold):
  - **USA:** ARIMA (0.007421)  
  - **Spain:** TBAT (0.029295)  
  - **Italy:** TBAT (0.005810)  
  - **UK:** TBAT (0.004310)  
  - **France:** N-Beats (0.004220)  
  - **Germany:** TBAT (0.003389)  
  - **Russia:** TBAT (0.002193)  
  - **Turkey:** HWAAS (0.000887)  
  - **Brazil:** DeepAR (0.002836)  
  - **Iran:** TBAT (0.000425)  

---

### 5.3. Statistical Ranking (Friedman Test)

To compare multiple models across all ten countries, we applied the non‐parametric **Friedman test** (significance α = 0.02) on the per‐country RMSE rankings. Lower rank = better average performance:

| Rank  | Algorithm  |
|:-----:|:----------:|
| 1.700 | **TBAT**   |
| 2.900 | ARIMA      |
| 2.900 | HWAAS      |
| 4.100 | N-Beats    |
| 4.600 | Prophet    |
| 4.800 | DeepAR     |

---

### 5.4. Holm’s Post‐hoc Analysis (TBAT vs All)

After the Friedman test, we used **Holm’s post‐hoc** to compare TBAT against all other algorithms (α = 0.02).  
- **Null Hypothesis (H₀):** There is no significant difference between TBAT and the compared algorithm.

| Comparison        | Test Statistic | Adjusted p‐Value | Result (Reject H₀?) |
|:-----------------:|:--------------:|:----------------:|:-------------------:|
| TBAT vs DeepAR    | 3.70521        | 0.00106          | **Reject H₀**       |
| TBAT vs Prophet   | 3.46616        | 0.00211          | **Reject H₀**       |
| TBAT vs N-Beats   | 2.86855        | 0.01237          | **Reject H₀**       |
| TBAT vs ARIMA     | 1.43427        | 0.30299          | Accept H₀           |
| TBAT vs HWAAS     | 1.43427        | 0.30299          | Accept H₀           |

**Conclusion:**  
- TBAT is **significantly better** (p < 0.02) than DeepAR, Prophet, and N-Beats.  
- No significant difference between TBAT vs. ARIMA or TBAT vs. HWAAS at α = 0.02.

---

### 5.5. Forecasting Examples

The following figures illustrate actual vs. predicted active‐case percentages over the final 7-day test window for selected countries. (Replace placeholder images with actual plots.)

- **Figure 1. USA Forecast (Test Horizon):**  
  ![USA Forecast](./figures/usa_forecast.png)

- **Figure 2. UK Forecast (Test Horizon):**  
  ![UK Forecast](./figures/uk_forecast.png)

- **Figure 3. Russia Forecast (Test Horizon):**  
  ![Russia Forecast](./figures/russia_forecast.png)

- **Figure 4. France Forecast (Test Horizon):**  
  ![France Forecast](./figures/france_forecast.png)

---

## 6. Conclusions & Future Work

- **Key Findings:**  
  - **TBAT** and **ARIMA** are the top performers for 7-day forecasting of active‐case percentages in most countries.  
  - **HWAAS** wins for Turkey (lowest RMSE), while **N-Beats** wins for France, and **DeepAR** (GluonTS) wins for Brazil.  
  - Deep‐learning methods (DeepAR, N-Beats) underperform statistical methods when data volume is limited.  
  - **Prophet** (designed for business seasonality) did not rank among top models for any country.  

- **Possible Explanatory Factors for Cross‐Country Differences:**  
  1. **Climate & Geography:** Differences in temperature/humidity may affect virus spread.  
  2. **Population Density & Demographics:** High density can accelerate transmission.  
  3. **Testing & Reporting Variability:** Inconsistent testing rates/data collection → noisy time series.  
  4. **Intervention Policies:** Timing, severity, and duration of lockdowns/social distancing differ by country.

- **Future Improvements:**  
  1. **Model Ensembles:** Combine ARIMA, TBAT, HWAAS, etc., to reduce overall forecast error.  
  2. **Multivariate Time Series:** Incorporate external covariates:  
     - **Climate Data:** Temperature, humidity, air quality.  
     - **Mobility Data:** Google/Apple mobility reports.  
     - **Policy Indices:** Stringency of lockdown measures.  
  3. **Transfer Learning:** Use learnings from data‐rich countries to improve forecasts in data‐scarce regions.  
  4. **Longer Forecast Horizons:** Extend to 14 or 21 days and evaluate model stability.  
  5. **Real‐Time Adaptation:** Incorporate an online‐learning setting where models retrain daily as new data arrive.  

---

## 7. References

1. World Health Organization. *Naming the Coronavirus Disease (COVID-19) and the Virus that Causes it.* 2020. Available online: [https://www.who.int/emergencies/diseases/novel‐coronavirus‐2019/technical‐guidance/naming‐the‐coronavirus‐disease‐(covid‐2019)‐and‐the‐virus‐that‐causes‐it](https://www.who.int/emergencies/diseases/novel‐coronavirus‐2019/technical‐guidance/naming‐the‐coronavirus‐disease‐(covid‐2019)‐and‐the‐virus‐that‐causes‐it) (accessed May 2, 2020).

2. Coronaviridae Study Group. *The species Severe acute respiratory syndrome‐related coronavirus: classifying 2019‐nCoV and naming it SARS‐CoV‐2.* Nat. Microbiol. 2020, 5, 536–544. [CrossRef]

3. Lu, H.; Stratton, C.W.; Tang, Y.W. *Outbreak of Pneumonia of Unknown Etiology in Wuhan China: The Mystery and the Miracle.* J. Med. Virol. 2020, 92, 401–402. [CrossRef]

4. Fernandes, N. *Economic Effects of Coronavirus Outbreak (COVID-19) on the World Economy.* SSRN 2020. Available online: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3557504](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3557504) (accessed May 4, 2020).

5. Johns Hopkins University CSSE. *Coronavirus COVID-19 Global Cases.* 2020. Available online: [https://coronavirus.jhu.edu/map.html](https://coronavirus.jhu.edu/map.html) (accessed May 4, 2020).

6. Roosa, K.; Lee, Y.; Luo, R.; Kirpich, A.; Rothenberg, R.; Hyman, J.; Yan, P.; Chowell, G. *Real‐time forecasts of the COVID‐19 epidemic in China from 5 February to 24 February 2020.* Infect. Dis. Model. 2020, 5, 256–263.

7. Yang, Z.; Zeng, Z.; Wang, K.; Wong, S.S.; Liang, W.; Zanin, M.; Liu, P.; Cao, X.; Gao, Z.; Mai, Z.; et al. *Modified SEIR and AI prediction of the epidemics trend of COVID-19 in China under public health interventions.* J. Thorac. Dis. 2020, 12, 165–174. [CrossRef]

8. Petropoulos, F.; Makridakis, S. *Forecasting the novel coronavirus COVID‐19.* PLoS ONE 2020, 15, e0231236. [CrossRef]

9. Box, G.; Jenkins, G. *Time Series Analysis: Forecasting and Control* (3rd ed.). John Wiley & Sons: Hoboken, NJ, USA, 2015.

10. Chatfield, C. *The Holt–Winters Forecasting Procedure.* J. R. Stat. Soc. 1978, 27, 264–279.

11. De Livera, A.M.; Hyndman, R.J.; Snyder, R.D. *Forecasting time series with complex seasonal patterns using exponential smoothing.* J. Am. Stat. Assoc. 2011, 106, 1513–1527. [CrossRef]

12. Taylor, S.J.; Letham, B. *Forecasting at Scale.* Am. Stat. 2018, 72, 37–45. [CrossRef]

13. Salinas, D.; Flunkert, V.; Gasthaus, J.; Januschowski, T. *DeepAR: Probabilistic forecasting with autoregressive recurrent networks.* Int. J. Forecast. 2019. doi:10.1016/j.ijforecast.2019.07.001 [CrossRef]

14. Alexandrov, A.; Benidis, K.; Bohlke-Schneider, M.; Flunkert, V.; Gasthaus, J.; Januschowski, T.; Maddix, D.C.; Rangapuram, S.; Salinas, D.; Schulz, J.; et al. *GluonTS: Probabilistic time series models in Python.* arXiv 2019, arXiv:1906.05264.

15. Oreshkin, B.N.; Carpov, D.; Chapados, N.; Bengio, Y. *N-Beats: Neural basis expansion analysis for interpretable time series forecasting.* arXiv 2019, arXiv:1905.10437.

16. Chadsuthi, S.; Modchang, C.; Lenbury, Y.; Iamsirithaworn, S.; Triampo, W. *Modeling seasonal leptospirosis transmission and its relationship with rainfall and temperature in Thailand using time-series and ARIMAX analyses.* Asian Pac. J. Trop. Med. 2012, 5, 539–546. [CrossRef]

17. Hanf, M.; Adenis, A.; Nacher, M.; Carme, B. *The role of El Niño Southern Oscillation (ENSO) on variations of monthly Plasmodium falciparum malaria cases at the Cayenne general hospital, 1996–2009, French Guiana.* Malar. J. 2011, 10, 100. [CrossRef] [PubMed]

18. Song, X.; Xiao, J.; Deng, J.; Kang, Q.; Zhang, Y.; Xu, J. *Time series analysis of influenza incidence in Chinese provinces from 2004 to 2011.* Medicine 2016, 95, e3929. [CrossRef]

19. Adhikari, R.; Agrawal, R.K. *An introductory study on time series modeling and forecasting.* arXiv 2013, arXiv:1302.6613.

20. Yin, R.; Luusua, E.; Dabrowski, J.; Zhang, Y.; Kwoh, C.K. *Tempel: Time-series mutation prediction of influenza A viruses via attention-based recurrent neural networks.* Bioinformatics 2020, 36, 2697–2704. [CrossRef]

21. Lee, K.; Agrawal, A.; Choudhary, A. *Forecasting influenza levels using real-time social media streams.* In Proceedings of the 2017 IEEE International Conference on Healthcare Informatics (ICHI), Park City, UT, USA, 23–26 August 2017; pp. 409–414.

22. Zhang, Y.; Yakob, L.; Bonsall, M.B.; Hu, W. *Predicting seasonal influenza epidemics using cross-hemisphere influenza surveillance data and local Internet query data.* Sci. Rep. 2019, 9, 1–7. [CrossRef]

23. Soebiyanto, R.P.; Adimi, F.; Kiang, R.K. *Modeling and predicting seasonal influenza transmission in warm regions using climatological parameters.* PLoS ONE 2010, 5, e9450. [CrossRef]

24. Dominguez, A.; Muñoz, P.; Martínez, A.; Orcau, A. *Monitoring mortality as an indicator of influenza in Catalonia, Spain.* J. Epidemiol. Community Health 1996, 50, 293–298. [CrossRef] [PubMed]

25. Tang, B.; Wang, X.; Li, Q.; Bragazzi, N.L.; Tang, S.; Xiao, Y.; Wu, J. *Estimation of the Transmission Risk of 2019-nCoV and Its Implication for Public Health Interventions.* J. Clin. Med. 2020, 9, 462. [CrossRef]

26. Dehning, J.; Zierenberg, J.; Spitzner, F.P.; Wibral, M.; Neto, J.P.; Wilczek, M.; Priesemann, V. *Inferring change points in the spread of COVID-19 reveals the effectiveness of interventions.* Science 2020, 369, eabb9789. [CrossRef]

27. Anastassopoulou, C.; Russo, L.; Tsakris, A.; Siettos, C. *Data-based analysis, modelling and forecasting of the COVID-19 outbreak.* PLoS ONE 2020, 15, e0230405. [CrossRef] [PubMed]

28. IHME COVID-19 Health Service Utilization Forecasting Team. *Modeling COVID-19 scenarios for the United States.* Nat. Med. 2020. doi:10.1038/s41591-020-11380-w [CrossRef]

29. Zhang, J.; Litvinova, M.; Wang, W.; Wang, Y.; Deng, X.; Chen, X.; Li, M.; Zheng, W.; Yi, L.; Chen, X.; et al. *Evolving epidemiology, transmission dynamics and control of COVID-19 outside Hubei province, China: A descriptive and modeling study.* Lancet Infect. Dis. 2020, 20, 793–802. [CrossRef]

30. Roosa, K. et al. *Real-time forecasts of the COVID-19 epidemic in China from 5 February to 24 February 2020.* Infect. Dis. Model. 2020, 5, 256–263.

31. Anastassopoulou, C.; Russo, L.; Tsakris, A.; Siettos, C. *Data-based analysis, modelling and forecasting of the COVID-19 outbreak.* PLoS ONE 2020, 15, e0230405. [CrossRef] [PubMed]

32. Zhang, J.; Litvinova, M.; Wang, W.; Wang, Y.; Deng, X.; Chen, X.; Li, M.; Zheng, W.; Yi, L.; et al. *Evolving epidemiology, transmission dynamics and control of COVID-19 outside Hubei province, China: A descriptive and modeling study.* Lancet Infect. Dis. 2020, 20, 793–802. [CrossRef]

33. Zhou, H.; Xu, J.; Xu, X.; Wang, Y.; Tong, Y.; Zhang, Q.; Zhang, X.; Fan, C.; Xiao, G.; Ding, X.; et al. *A deep downscaling approach of global climate model outputs to urban area using convolutional neural networks (CNN).*

34. Zhang, Y.; Litvinova, M.; Wang, W.; et al. *Evolving epidemiology, transmission dynamics and control of COVID-19 outside Hubei province, China.* Lancet Infect. Dis. 2020, 20, 793–802. [CrossRef]

35. Petropoulos, F.; Makridakis, S. *Forecasting the novel coronavirus COVID-19.* PLoS ONE 2020, 15, e0231236. [CrossRef]

36. Wu, J.T.; Leung, K.; Leung, G.M. *Nowcasting and forecasting the potential domestic and international spread of the 2019‐nCoV outbreak originating in Wuhan, China: A modelling study.* Lancet 2020, 395, 689–697. [CrossRef]

37. Anastassopoulou, C.; Russo, L.; Tsakris, A.; Siettos, C. *Data‐based analysis, modelling and forecasting of the COVID‐19 outbreak.* PLoS ONE 2020, 15, e0230405. [CrossRef]

38. Zhang, S.; Diao, M.; Yu, W.; Pei, L.; Lin, Z.; Chen, D. *Estimation of the reproductive number of novel coronavirus (COVID-19) and the probable outbreak size on the Diamond Princess cruise ship: A data-driven analysis.* Int. J. Infect. Dis. 2020, 93, 201–204. [CrossRef] [PubMed]

39. Institute for Health Metrics and Evaluation (IHME). *IHME COVID-19 Health Service Utilization Forecasting Team.* Available online: [http://www.healthdata.org/covid](http://www.healthdata.org/covid) (accessed May 4, 2020).

40. Petropoulos, F.; Makridakis, S. *Forecasting the novel coronavirus COVID-19.* PLoS ONE 2020, 15, e0231236. [CrossRef]

41. Box, G.E.P.; Jenkins, G.M.; Reinsel, G.C.; Ljung, G.M. *Time Series Analysis: Forecasting and Control.* John Wiley & Sons: Hoboken, NJ, USA, 2015.

42. Hyndman, R.J.; Khandakar, Y. *Automatic time series forecasting: The forecast package for R.* J. Stat. Softw. 2008, 27, 1–22. [CrossRef]

43. Brockwell, P.J.; Davis, R.A. *Time Series: Theory and Methods* (2nd ed.). Springer: New York, NY, USA, 2002.

44. Hyndman, R.J.; Athanasopoulos, G. *Forecasting: Principles and Practice* (2nd ed.). OTexts: Melbourne, Australia, 2018.

45. Gao, Q.; Steadman, P.; Shi, J. *Unmanned aerial vehicles for landscape monitoring: A review.* Remote Sens. 2020, 12, 1227. [CrossRef]

46. Gardner, E.S.; McKenzie, E. *Forecasting trends in time series.* Manag. Sci. 1985, 31, 1237–1246. [CrossRef]

47. Kwok, S.S.; Liao, H.T.; Fang, Y.H. *Forecasting with seasonal and trend components using Holt–Winters smoothing.* Int. J. Forecast. 1986, 2, 335–345. [CrossRef]

48. Chatfield, C. *The Analysis of Time Series: An Introduction* (6th ed.). Chapman & Hall/CRC: Boca Raton, FL, USA, 2003.

49. Snyder, R.D.; Hyndman, R.J. *A state‐space framework for automatic forecasting using exponential smoothing methods.* Int. J. Forecast. 2002, 18, 439–454. [CrossRef]

50. Box, G.E.P.; Cox, D.R. *An analysis of transformations.* J. R. Stat. Soc. 1964, 26, 211–243. [CrossRef]

51. Taylor, S.J.; Letham, B. *Forecasting at scale.* Am. Stat. 2018, 72, 37–45. [CrossRef]

52. Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical Learning.* Springer: New York, NY, USA, 2009.

53. Verhulst, P.-F. *Notice sur la loi que la population poursuit dans son accroissement.* Correspondance Math. Phys. 1838, 10, 113–121.

54. Brockwell, P.J.; Davis, R.A. *Introduction to Time Series and Forecasting* (2nd ed.). Springer: New York, NY, USA, 2002.

55. Hochreiter, S.; Schmidhuber, J. *Long Short‐Term Memory.* Neural Comput. 1997, 9, 1735–1780. [CrossRef]

56. Graves, A. *Supervised Sequence Labelling with Recurrent Neural Networks.* Springer: London, UK, 2012.

57. Salinas, D.; Flunkert, V.; Gasthaus, J.; Januschowski, T. *DeepAR: Probabilistic forecasting with autoregressive recurrent networks.* Int. J. Forecast. 2019. doi:10.1016/j.ijforecast.2019.07.001 [CrossRef]

58. Lim, B.; Arık, S.Ö.; Loeff, N.; Pfister, T. *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.* Int. J. Forecast. 2021. doi:10.1016/j.ijforecast.2021.02.005 [CrossRef]

59. He, K.; Zhang, X.; Ren, S.; Sun, J. *Deep residual learning for image recognition.* In Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; pp. 770–778. [CrossRef]

60. Hyndman, R.J.; Athanasopoulos, G. *Forecasting: Principles and Practice.* OTexts: Melbourne, Australia, 2018.

61. Novel Corona Virus 2019 Dataset. Kaggle. Available online: [https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-india-dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-india-dataset) (accessed May 4, 2020).

62. Population by Country Dataset. Kaggle. Available online: [https://www.kaggle.com/datasets/fernandol/countries-of-the-world](https://www.kaggle.com/datasets/fernandol/countries-of-the-world) (accessed May 4, 2020).

63. Friedman, M. *The use of ranks to avoid the assumption of normality implicit in the analysis of variance.* J. Am. Stat. Assoc. 1937, 32, 675–701. [CrossRef]

64. Holm, S. *A simple sequentially rejective multiple test procedure.* Scand. J. Stat. 1979, 6, 65–70.

---
