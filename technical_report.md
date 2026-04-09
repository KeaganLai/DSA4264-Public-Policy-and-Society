<h1 align="center">Effect of Proximity to “Good” Schools on HDB Resale Prices</h1>

## 1. Introduction

### 1.1 Background
In Singapore, primary school admission prioritises children living within 1 km, creating strong demand for housing near “good” schools. This demand is capitalised into higher HDB resale prices, potentially reducing affordability and increasing spatial inequality by linking educational access to housing wealth.
### 1.2 Research Question
This study aims to estimate the effect of proximity to “good” primary schools on HDB resale prices. Here, “good schools” are defined as schools with high perceived school desirability rather than purely objective academic performance. Specifically, it examines whether flats nearer to high-demand schools command a premium, controlling for other factors. Using the 1 km admission cutoff, it identifies whether price differences are driven by proximity rather than underlying housing characteristics, providing policy-relevant evidence on education’s impact on housing markets.

This study’s success is defined by ensuring the hedonic and RDD results are correctly interpreted. Findings must be policy-relevant from an MND perspective, highlighting implications for affordability and spatial inequality. Results should also be communicated clearly, with simple, intuitive takeaways that are easily understood by non-technical stakeholders.


## 2. Data

### 2.1 Overview
The analysis uses HDB resale transactions across Singapore (2013–2025), merged with school registration and geospatial accessibility data. Each observation represents a resale transaction matched with housing, school, and locational characteristics. 

A detailed summary of all datasets, variables and data sources is provided in Appendix A.

### 2.2 HDB Resale Transaction Data
HDB resale transaction data is sourced from data.gov.sg, and flat addresses are geocoded using the OneMap API, with only a small number of observations with missing coordinates removed.

### 2.3 Housing Price Index
To account for inflation, resale prices are deflated using the HDB Resale Price Index (RPI). Real resale prices are constructed as:

$$
\text{RealPrice}_{it} = \frac{\text{NominalPrice}_{it}}{RPI_t / 100}
$$

where i indexes flats and t indexes time.

### 2.4 School Registration and School Characteristics Data
School desirability is measured using Primary 1 registration outcomes. The number of applicants and vacancies in Phases 2B and 2C are used, as they better reflect parental demand than earlier priority-based phases, which depend on institutional ties like siblings, alumni, and staff affiliation.

Ballot intensity and persistence capture short- and long-term demand, while participation in Gifted Education Programme (GEP) and Special Assistance Plan (SAP) proxy school quality. School locations are obtained via OneMap with manual corrections where needed.

### 2.5 Accessibility and Amenity Data
To control for neighbourhood accessibility, geospatial data on transport and amenities are used. Bus stop and MRT locations come from LTA DataMall, hawker centres from data.gov.sg, and shopping malls are separately geocoded.

Distances are calculated using the Haversine formula, which measures great-circle distance and accounts for Earth’s curvature, providing more accurate city-wide proximity measures than Euclidean distance.

Distance to the Central Business District is proxied by Raffles Place MRT station, reflecting its role as Singapore’s primary financial centre (National Library Board, 2023). 

## 3. Data Preprocessing and Feature Engineering

### 3.1 Dependent Variable
The main dependent variable is the logarithm of real resale price per square foot:

$$
\text{RealPricePSF}_{it} = \frac{\text{RealPrice}_{it}}{\text{FloorArea}_{it}}
$$

$$
y_{it} = \log(\text{RealPricePSF}_{it})
$$

Price per square foot normalises prices by unit size, ensuring comparability across flats and avoiding confounding size effects with location or school proximity.

The log transformation reduces skewness, limits outlier influence, and stabilises estimates, while allowing coefficients to be interpreted as approximate percentage changes, which are more meaningful in a housing context.

### 3.2 Construction of the School Desirability Index (SDI)
The SDI captures both the intensity and persistence of excess demand for each school.

### 3.2.1 Ballot Intensity

Ballot intensity is measured as the log ratio of applicants to vacancies in Phases 2B and 2C:

$$
BI_{st} = \log \left( \frac{Applicants^{2B}_{st} + Applicants^{2C}_{st}}{Vacancies^{2B}_{st} + Vacancies^{2C}_{st}} \right)
$$

**where:**

- Applicants<sup>2B</sup><sub>st</sub> and Applicants<sup>2C</sup><sub>st</sub> denote the number of applicants  
- Vacancies<sup>2B</sup><sub>st</sub> and Vacancies<sup>2C</sup><sub>st</sub> denote the number of vacancies  
- for school <em>s</em> in year <em>t</em>

An across-phase ballot indicator is added to identify schools oversubscribed in both phases, reflecting consistently high demand:

$$
APB_{st} =
\begin{cases}
1 & \text{if } Applicants^{2B}_{st} > Vacancies^{2B}_{st} \text{ and } Applicants^{2C}_{st} > Vacancies^{2C}_{st} \\
0 & \text{otherwise}
\end{cases}
$$

These are then combined, where BI is weighted more heavily as it captures how strong demand is, while the indicator captures widespread demand:

$$
BI^*_{st} = 0.7 \cdot z(BI_{st}) + 0.3 \cdot APB_{st}
$$

**where:**
- $z(\cdot)$: standardisation  



### 3.2.2 Persistence of Demand

To measure how consistently a school experiences excess demand, we compute a rolling ballot frequency:

$$
RBF_{st} = \frac{1}{5} \sum_{\tau = t-4}^{t} Ballot_{s\tau}
$$

**where:**
- $Ballot_{s\tau}$: indicator equal to 1 if balloting occurred in year $\tau$, otherwise 0  
- averages ballot occurrences in Phase 2B or 2C over the past five years

### 3.2.3 School Desirability Index

The SDI combines both dimensions, capturing current demand and its persistence over time:

$$
SDI_{st} = \frac{BI^*_{st} + z(RBF_{st})}{2}
$$

### 3.3 Construction of the Good School Index (GSI)

The Good School Index (GSI) assumes housing prices reflect perceived rather than purely objective school quality. Schools seen as offering better opportunities attract stronger admission demand which, under Singapore’s distance-based admission system, is capitalised into higher nearby housing prices.

The index combines demand indicators with GEP and SAP participation to capture both parental preferences and school characteristics. Greater weight is assigned to SDI, as it directly reflects demand intensity and households’ willingness to pay.

$$
GSI_{st} = 0.6 \cdot SDI_{st} + 0.2 \cdot GEP_s + 0.2 \cdot SAP_s
$$

Schools are classified as “good” if they fall in the upper tail of the GSI distribution. The baseline uses the 80th percentile, with 75, 85, and 90 tested for robustness.

### 3.4 School Proximity Variables

Firstly, school density measures count the number of primary schools within specified distance bands from each flat:

$$
countall_{0\text{-}1km}\,i = \sum_{s} \mathbf{1}(d_{is} \leq 1)
$$

$$
countall_{1\text{-}2km}\,i = \sum_{s} \mathbf{1}(1 < d_{is} \leq 2)
$$

**where:**
- $d_{is}$ denotes the distance between flat $i$ and school $s$

Secondly, school desirability measures count the number of schools above each GSI percentile threshold $q \in \{75, 80, 85, 90\}$ within these distance bands, capturing exposure to highly desirable schools:


$$
count_{0\text{-}1km,\,good_q}\,i = \sum_{s} \mathbf{1}(GSI_{st} \geq q_t \,\cap\, d_{is} \leq 1)
$$

$$
count_{1\text{-}2km,\,good_q}\,i = \sum_{s} \mathbf{1}(GSI_{st} \geq q_t \,\cap\, 1 < d_{is} \leq 2)
$$

Binary indicators are also constructed to reflect whether at least one such school exists within each band, providing a more interpretable measure of access:


$$
d_{0\text{-}1km,\,goodq}\,i = \mathbf{1}(count_{0\text{-}1km,\,goodq}\,i > 0)
$$

$$
d_{1\text{-}2km,\,goodq}\,i = \mathbf{1}(count_{1\text{-}2km,\,goodq}\,i > 0)
$$

### 3.5 Mature Estate Indicator

A binary variable indicating whether a flat is in a mature town. The towns can be found in Appendix B.

### 3.6 Flat Type Mapping

Flat type is converted into an ordinal variable based on size, with larger flats assigned higher values to better capture their relationship with resale price.

### 3.7 Categorisation of Storey Range

Storey range is converted to a midpoint floor level, then normalised within each town using percentile ranks to account for differences in building height. Flats are grouped into:

- LOW_IN_ESTATE: Bottom 33% of floors within the town  
- MID_IN_ESTATE: Middle 33%  
- HIGH_IN_ESTATE: Top 33%

## 4. Methodology
This study adopts a two-pronged empirical strategy combining a hedonic regression model with a regression discontinuity design (RDD). The hedonic regression serves as the primary framework to estimate the relationship between HDB resale prices and proximity to desirable primary schools across the full sample, controlling for structural and locational factors. 

To strengthen causal interpretation, an RDD is implemented around the 1 km admission cutoff. By comparing flats just inside and outside the boundary, it identifies the local causal effect of priority admission eligibility. Together, the hedonic model captures broad, market-wide price gradients associated with school proximity, while the RDD isolates the causal effect of the admission policy at the cutoff, helping to distinguish policy-driven effects from purely correlational patterns. 

The validity and interpretation of these empirical strategies depend on several key assumptions.

Firstly, the hedonic model assumes that, after controlling for observable characteristics, school proximity is exogenous to unobserved neighbourhood quality. While this may not fully hold in practice, it is necessary for interpreting the estimated premium as a school-related effect.

Secondly, school desirability is proxied using the Good School Index (GSI). This assumes that the index captures both observed parental demand and underlying school characteristics that shape perceptions. If this proxy is mis-specified, the classification of “good schools” and the interpretation of estimated proximity premiums would be affected.

### 4.1 Hedonic Regression Model Specification

The baseline specification is:

$$
\log(Price_i) = \beta_0 + \beta_1 d^{0\text{-}1km}_i + \beta_2 d^{1\text{-}2km}_i + \gamma X_i + \varepsilon_i
$$

**where:**
- $\log(Price_i)$: log real price per square foot  
- $d^{0\text{-}1km}_i$: indicator for flats within 1 km of a desirable school  
- $d^{1\text{-}2km}_i$: indicator for flats within 1–2 km of a desirable school  
- $X_i$: vector of control variables  
- $\varepsilon_i$: error term  

The coefficients $\beta_1$ and $\beta_2$ represent percentage price premiums for being within each distance band, holding all other factors constant.

To improve model design and assess robustness, the hedonic regression is estimated sequentially.  The baseline includes flat characteristics, followed by locational controls.

Finally, school availability measures (e.g., number of nearby primary schools) are included to distinguish proximity to desirable schools from general access. This stepwise approach tests whether the estimated proximity premium remains stable with additional controls, strengthening the credibility of the results.

### 4.2 Regression Discontinuity Design

The baseline RDD specification is:

$$
\log(Price_i) = a + \tau \cdot \mathbf{1}(dist_i \leq 1) + f(dist_i) + \varepsilon_i
$$

**where:**
- $\log(Price_i)$: log real price per square foot  
- $\mathbf{1}(dist_i \leq 1)$: indicator for flats within 1 km of a desirable school (treatment)  
- $dist_i$: distance to the nearest desirable school (running variable, centred at 1 km cutoff)  
- $f(dist_i)$: linear function of distance  
- $\varepsilon_i$: error term  

The coefficient $\tau$ captures the local treatment effect at the 1 km cutoff, interpreted as the causal impact of eligibility for the Primary 1 priority admission policy on resale prices. It reflects the price difference between flats just within and just outside the boundary.

To improve identification and account for potential confounding, the RDD is extended as follows:

$$
\log(Price_i) = a + \tau \cdot \mathbf{1}(dist_i \leq 1) + f(dist_i) + \gamma_s + \beta X_i + \varepsilon_i
$$

**where:**
- $\gamma_s$: school fixed effects  
- $X_i$: vector of control variables  

This specification strengthens the design by controlling for school-level heterogeneity using fixed effects and improving inference through clustering at the school level.

## 5. Results

### 5.1 Hedonic Regression Model

#### 5.1.1 Empirical Results

<div align="center">

| Model   | Specification     | 0–1km Premium (%) | 1–2km Premium (%) | Adj R² | N       |
|--------|------------------|------------------|------------------|--------|---------|
| Model 1 | FE only          | 0.73**           | 1.66**           | 0.778  | 291,419 |
| Model 2 | + Accessibility  | 0.87**           | 0.75**           | 0.822  | 291,419 |
| Model 3 | + School Density | 1.16**           | 0.41**           | 0.823  | 291,419 |

</div>

<p align="center"><strong>Table 1: Results of Hedonic Regression</strong></p>

<p align="center"><strong>Notes:</strong> ** indicates statistical significance at the 5% level (p &lt; 0.05)</p>

Model 1 produces counterintuitive results, with a larger premium for 1–2 km than 0–1 km, suggesting omitted variable bias from unaccounted locational factors.

After introducing accessibility controls (Model 2), the 0-1 km premium rises while the 1-2 km premium falls, indicating that part of the earlier effect was driven by transport and amenities. 

With school density controls (Model 3), the estimates stabilise, with the premium concentrated within 1 km and a smaller effect for 1-2 km.

Model 3 is adopted as the preferred specification, as it accounts for both locational accessibility and school availability, thereby isolating the premium associated with proximity to desirable schools in a manner consistent with the institutional admission framework.

### 5.1.2 Robustness Checks

#### 5.1.2.1 Alternative Definitions of “Good School” (Threshold Robustness)

To test sensitivity to the definition of “good schools”, the hedonic regression was re-estimated using SDI thresholds of 75, 80, 85, and 90.

<div align="center">

| Threshold | 0–1km Premium (%) | 1–2km Premium (%) | Adj. R² | N       |
|----------|------------------|------------------|---------|---------|
| 75       | 0.46**          | 0.80**          | 0.8227  | 291,419 |
| 80       | 1.16**          | 0.41**          | 0.8229  | 291,419 |
| 85       | 1.27**          | 0.34**          | 0.8230  | 291,419 |
| 90       | 0.67**          | 0.12**          | 0.8226  | 291,419 |

<p><strong>Table 2: Results of Threshold Robustness</strong></p>

</div>

At the 75 threshold, results are less precise, with the 1–2 km premium exceeding the 0–1 km premium, suggesting the classification is too broad and captures general school access rather than the true school quality.

At thresholds 80 and 85, estimates are more consistent, with a stronger premium within 1 km and a smaller effect for 1–2 km.

At the 90 threshold, both premiums decline, likely due to fewer schools being included.

Overall, results are robust to threshold variation, with 80-85 providing the best balance between capturing desirable schools and preserving sufficient variation.

#### 5.1.2.2 Alternative Weights in GSI

This robustness check tests whether the estimated premium depends on the GSI weighting scheme, while keeping SDI as the dominant component relative to GEP and SAP.

<div align="center">

| GSI Weighting Scheme        | 0–1km Premium (%) | 1–2km Premium (%) | Adj. R² | N       |
|----------------------------|------------------|------------------|---------|---------|
| 0.7 SDI, 0.15 GEP, 0.15 SAP | 1.17**          | 0.50**          | 0.8230  | 291,419 |
| 0.8 SDI, 0.1 GEP, 0.1 SAP   | 1.09**          | 0.49**          | 0.8229  | 291,419 |
| 0.6 SDI, 0.2 GEP, 0.2 SAP   | 1.16**          | 0.41**          | 0.8229  | 291,419 |

<p><strong>Table 4: Results of Alternative Weights in GSI</strong></p>

</div>

The similar estimates suggest that results are not sensitive to the GSI weighting scheme, as long as SDI remains dominant. The consistency across specifications supports the use of an SDI-led GSI, capturing parental demand while incorporating school characteristics through GEP and SAP.

### 5.1.3 Heterogeneity Analysis

#### 5.1.3.1 Flat-Type Heterogeneity

<div align="center">

| Flat Type Group           | 0–1km Premium (%)  | 1–2km Premium (%)   |
|--------------------------|----------|----------|
| 1–3 room flats           | 0.43** | −2.22** |
| 4-room and above flats   | 1.33** | 1.28** |

<p><strong>Table 5: Heterogeneity in School Proximity Premium</strong></p>

</div>

To assess variation by household type, interaction terms between school proximity and flat type are included. Larger flats are interpreted as a proxy for family households, who are more sensitive to school access.

Results show a stronger premium for larger flats. In particular, the interaction terms for larger flat types are positive and statistically significant, indicating that households residing in these units are willing to pay a higher premium to be near desirable primary schools. In contrast, for 1–3 room flats, the premium is smaller within 1 km and negative for 1–2 km.

Overall, the school premium varies by household type and is driven more by family-oriented buyers.

#### 5.1.3.2 Mature Estates Heterogeneity

<div align="center">

| Estate Type Group   | 0–1km Premium (%)   | 1–2km Premium (%)   |
|--------------------|----------|----------|
| Non-mature estates | 1.14** | 0.17**   |
| Mature estates     | 1.25 | 1.27** |

<p><strong>Table 6: Results of Mature vs Non-Mature Estates</strong></p>

</div>

To examine heterogeneity across estate types, interaction terms between school proximity and mature estates are included.

The results show that within the 0–1 km band, the price premium is similar across both groups. While mature estates exhibit a slightly higher premium, the difference is not statistically significant.In contrast, within the 1–2 km band, the premium is significantly higher in the mature estates than in the non-mature estates. This indicates that while the immediate proximity premium is comparable across estate types, the influence of school proximity extends over a wider distance in mature estates


### 5.1.4 Hedonic Results Overview

The hedonic regression results provide consistent evidence that proximity to desirable primary schools is capitalised into HDB resale prices. The model estimates an average premium of about 1.16% within 1 km and 0.41% within 1–2 km. 

Robustness checks confirm the stability of this finding, with premiums remaining consistent across alternative definitions of “good” schools and GSI weighting schemes. Additional analysis using school counts further shows that the marginal value of additional good schools is concentrated within the 1 km band, reinforcing the importance of close proximity.

Heterogeneity results indicate that the premium is stronger for larger flats, suggesting that school-related housing demand is driven mainly by family households. While the 1 km premium is similar across estate types, it persists over a wider range in mature estates.

Overall, while the hedonic results identify an average premium across distance bands, this effect is not uniform across households or spatial contexts. 

### 5.2 Regression Discontinuity Design
### 5.2.1 Empirical Results of Baseline RDD Model

<div align="center">

| Model    | Specification | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|--------------|--------------------|--------------------|
| Baseline | τ            | -1.83**           | -4.36**           |

<p><strong>Table 7: Baseline RDD Results</strong></p>

</div>

The baseline RDD estimates indicate a statistically significant negative price discontinuity at both distance-boundary cutoffs. Because treatment is coded as 1($dist_i$ ≤ c), a negative τ suggests that resale flats located just inside the 1 km zone are cheaper than otherwise similar flats just outside the boundary.

#### 5.2.1.1 Robustness checks

Covariate balance tests re-run the same RDD using pre-treatment or predetermined covariates as outcomes. If the design is locally comparable, these covariates should be continuous at the cutoff.

<div align="center">

| Covariate (1 km cutoff)        | Jump   | Significance     |
|--------------------------------|--------|------------------|
| Floor area                     | 3.80%  | **Significant**  |
| Remaining lease                | -0.20% | Not significant  |
| Distance to MRT                | 0.66%  | Not significant  |
| Distance to mall               | -2.21% | **Significant**  |
| Distance to hawker             | 0.98%  | Not significant  |
| Distance to bus stop           | 0.21%  | **Significant**  |
| Count of all schools (0–1 km)  | 1.10%  | **Significant**  |
| Count of all schools (1–2 km)  | -1.10% | **Significant**  |

<p><strong>Table 8: Baseline RDD Covariate Balance Test Results</strong></p>

</div>

Several significant jumps indicate imperfect local balance in the baseline design, so baseline causal interpretation should be treated cautiously. Refer to Appendix C for detailed robustness checks on the baseline RDD.

### 5.2.2 Empirical Results of Extended RDD Model

The extended RDD introduces school fixed effects and clustered standard errors to tighten identification by comparing flats around boundaries relative to the same nearest desirable school.

<div align="center">

| Model    | Specification | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|--------------|--------------------|--------------------|
| Extended | τ            | -1.71             | 0.60              |

<p><strong>Table 9: Extended RDD Results</strong></p>

</div>

The extended-model estimate is statistically insignificant at the 5% level. Relative to the baseline, this suggests weaker evidence of a stable discontinuity once school-level heterogeneity is absorbed.

### 5.2.3 Robustness Checks

#### 5.2.3.1 Alternative Definitions of “Good” Schools (Threshold Robustness)

<div align="center">

| Threshold | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|------------------|------------------|
| 75       | -0.77           | 5.55            |
| 80       | -1.71           | 0.60            |
| 85       | -1.52           | 0.76            |
| 90       | -0.55           | 0.90            |

<p><strong>Table 10: Extended RDD with Different “Good” School Thresholds</strong></p>

</div>

While the estimates are statistically insignificant across all thresholds, the direction of coefficients remains broadly consistent. This suggests that the null result is robust to alternative definitions of school desirability.

#### 5.2.3.2 Bandwidth

<div align="center">

| Bandwidth (km) | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------------|------------------|------------------|
| 0.3            | -2.54**         | 0.99            |
| 0.5            | -1.71           | 0.60            |
| 0.8            | -1.04           | -0.36           |
| 1.0            | -0.52           | 0.24            |

<p><strong>Table 11: Extended RDD with Various Bandwidths</strong></p>

</div>

Only the narrowest bandwidth yields a statistically significant estimate. This indicates weak bandwidth robustness in the extended specification.

#### 5.2.3.3 Control Variables

<div align="center">

| Model    | Specification | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|--------------|--------------------|--------------------|
| Extended | τ            | -1.71             | 0.60              |
| Adjusted | + Controls   | -0.87             | -0.74             |

<p><strong>Table 12: Comparing Extended RDD and Added Controls</strong></p>

</div>

Adding control variables weakens the 1 km estimate and reverses the sign at 2 km, while remaining statistically insignificant.

#### 5.2.3.4 Covariate Balance

<div align="center">

| Covariate (1km cutoff)           | Jump   | Significance      |
|---------------------------------|--------|-------------------|
| Floor Area                      | 2.88%  | Significant       |
| Remaining lease                 | -0.17% | Not significant   |
| Distance to MRT                 | 1.84%  | Not significant   |
| Distance to mall                | -4.38% | Not significant   |
| Distance to hawker              | -1.76% | Not significant   |
| Distance to bus stop            | 0.18%  | Not significant   |
| Count of all schools (0–1km)    | 1.09%  | Significant       |
| Count of all schools (1–2km)    | -1.05% | Significant       |

<p><strong>Table 13: Extended RDD Covariate Balance Test Result</strong></p>

</div>

Covariate balance improves relative to the baseline for most amenity variables, though some discontinuities remain. This indicates imperfect local balance, so causal interpretation should be treated with caution. Refer to Appendix D for additional robustness checks supporting the extended specification.

### 5.2.4 RDD Results Overview

Across all specifications, the RDD finds no statistically significant price discontinuity at the 1 km cutoff. Results are robustly insignificant across alternative definitions of school desirability. However, estimates are sensitive to bandwidth choice and the inclusion of controls, and balance tests indicate residual covariate differences at the cutoff. Overall, there is no strong or stable evidence that eligibility for the 1 km priority rule leads to a price discontinuity.

## 6. Policy Implications

From a Ministry of National Development (MND) perspective, the findings show that education factors influence housing outcomes. The hedonic results suggest perceived school quality is capitalised into prices, potentially reducing affordability.

While the RDD finds no causal effect of the 1 km policy, the link between school desirability and demand remains. Price differences likely reflect neighbourhood characteristics and school reputation, pointing to spatial inequality driven by concentrated demand rather than policy boundaries.

Heterogeneity results show stronger effects for larger flats, indicating that family-oriented households are more sensitive to school access and drive demand. In addition, while the effect within 1 km is similar across estates, the effect extends further in mature estates, suggesting broader school-related demand.

These findings highlight the need for integrated planning between housing and education. For MND, this includes addressing uneven demand by improving estate attractiveness and working with the Ministry of Education to better manage school demand and perceptions. 

Targeted housing supply measures can also help alleviate pressure. Increasing the supply of larger flats through BTO in high-demand areas may address the needs of households most affected by school access considerations, while improving school quality and neighbourhood qualities in non-mature estates can help redistribute demand more evenly.

Overall, policy should focus on managing the spatial and demographic concentration of school-related housing demand, rather than relying on adjustments to admission thresholds alone.

## 7. Limitations
Accessibility measures use current amenity locations rather than historical infrastructure at the time of the transaction. School relocations and boundary changes are not captured, introducing measurement error, especially for earlier years, which may attenuate estimates. Future work could incorporate historical data to improve accuracy.

Hedonic regression may suffer from omitted variable bias, as unobserved neighbourhood factors (e.g., prestige, community) may affect both prices and school desirability. Hence, estimated premiums may partly reflect these factors.

Housing prices are also shaped by cooling measure policies, including the Total Debt Servicing Ratio (TDSR), which restricts borrowing and limits buyers’ ability to bid for preferred locations. This can dampen price variation, meaning the estimated proximity premium may understate true demand for school access.

## 8. Future Recommendations
Firstly, approach HDB and LTA for historical amenity and infrastructure data to be incorporated and matched to the transaction year, so that accessibility measures reflect the actual environment buyers faced at the time of purchase. Historical school relocations could likewise be included to reduce measurement error in school proximity variables. 

Secondly, the measurement of “good” schools could be refined by incorporating additional indicators of school quality or parental perceptions, allowing a clear distinction between perceived prestige and underlying educational quality. 

Finally, future research could model housing policy interventions more explicitly by examining whether school-related price premiums vary across different regulatory regimes. 

These extensions would allow policymakers to better distinguish between demand-driven premiums and policy-induced effects, improving the targeting of housing and education interventions.

---
Word count: 2865 (Excluding headers, tables, references and appendix)

## 9. References
National Library Board. (2023). Raffles Place. 

https://www.nlb.gov.sg/main/article-detail?cmsuuid=b62dd9d7-ea3c-42b4-a3b7-fd97d601cea4

## 10. Appendix
### Appendix A

<div align="center">

| Dataset                          | Source                          | Link                                                                 | Key variables                                           | Purpose                                      |
|----------------------------------|----------------------------------|----------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------|
| HDB resale flat transactions     | data.gov.sg                     | https://data.gov.sg/collections/189/view                            | resale price, flat type, floor area, town, lease, storey range | Main housing dataset                         |
| HDB Resale Price Index (RPI)     | Housing & Development Board     | https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics | quarterly price index                                   | Deflation of nominal housing prices          |
| Primary school registration outcomes | Aggregated school registration sources | https://sgschooling.com/year/                                       | applicants, vacancies, ballot outcomes                 | Construction of school desirability measures |
| School characteristics (GEP, SAP)| School-level information        | https://www.property2b2c.com/school-ranking/bespoke                 | programme participation indicators                     | Proxy for institutional prestige             |
| Bus stop locations               | LTA DataMall                    | https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html   | latitude, longitude                                    | Accessibility control                        |
| MRT station data                 | LTA DataMall                    | https://datamall.lta.gov.sg/content/datamall/en/static-data.html    | station coordinates                                    | Accessibility control and CBD reference      |
| Hawker centre dataset            | data.gov.sg                     | https://data.gov.sg/datasets?query=hawker                           | location coordinates                                   | Neighbourhood amenity proxy                  |
| Shopping mall locations          | Compiled list and geocoded      | https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore   | mall coordinates                                       | Commercial accessibility proxy               |
| Geocoding service                | OneMap API                      | https://www.onemap.gov.sg                                           | latitude, longitude                                    | Spatial matching of flats and schools        |

</div>

### Appendix B

<div align="center">

| Mature Estate |  |  |
|--------------|--------------|--------------|
| Ang Mo Kio   | Bedok        | Bishan       |
| Bukit Merah  | Bukit Timah  | Central Area |
| Geylang      | Kallang/Whampoa | Marine Parade |
| Queenstown   | Serangoon    | Toa Payoh    |

</div>

### Appendix C

<div align="center">

| Threshold | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|------------------|------------------|
| 75       | -0.12           | 6.43**          |
| 80       | -1.83**         | -4.36**         |
| 85       | -2.04**         | -3.95**         |
| 90       | -0.68**         | -3.52**         |

<p><strong>Baseline RDD estimates vary in magnitude and sign across thresholds, indicating instability to school desirability definitions.</strong></p>

</div>


<div align="center">

| Bandwidth (km) | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------------|------------------|------------------|
| 0.3            | -3.55**         | -4.38**         |
| 0.5            | -1.83**         | -4.36**         |
| 0.8            | -1.16**         | -3.95**         |
| 1.0            | -0.79**         | -3.52**         |

<p><strong>For the base RDD, the negative discontinuity weakens as bandwidth increases, indicating a stronger local effect near the cutoff.</strong></p>

</div>


<div align="center">

| Model    | Specification | 0–1 km Premium (%) | 1–2 km Premium (%) |
|----------|--------------|--------------------|--------------------|
| Baseline | τ            | -1.83%**           | -4.36%**           |
| Adjusted | + Controls   | -0.32%**           | 0.02%              |

<p><strong>Adding controls reduces the discontinuity and removes significance, indicating the baseline effect reflects compositional differences rather than a true boundary effect.</strong></p>

</div>

### Appendix D


<div align="center">

| Donut Hole (km) | Premium |
|------------------|---------|
| 0.05             | -1.39%  |
| 0.10             | -0.39%  |

<p><strong>The extended RDD donut test shows that excluding observations near the cutoff does not affect results, indicating no sorting bias and no causal effect.</strong></p>

</div>


<div align="center">

| Placebo Cutoff (km) | Premium |
|---------------------|---------|
| 0.8                 | 0.58%   |
| 1.2                 | 1.50%   |

<p><strong>The extended RDD placebo test shows no spurious discontinuities, supporting model validity but confirming no causal effect at the cutoff.</strong></p>

</div>
