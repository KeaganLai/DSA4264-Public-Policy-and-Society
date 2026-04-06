# Report Findings for Policy Chat RAG

Generated from `Report(2).docx` for repo-native RAG usage.

## Quick facts for policy Q&A

- Preferred hedonic specification: Model 3 (with accessibility and school-density controls).
- Main premium estimate (threshold 80): within 0 to 1 km = **1.16%**, within 1 to 2 km = **0.41%**.
- Interpretation: average premiums are positive and stronger closer to desirable schools.
- RDD interpretation: no stable, statistically robust causal jump exactly at the 1 km policy boundary.

## Acronym definitions

- SDI = School Desirability Index (demand-based school desirability metric in this project).
- GSI = Good School Index.
- RDD = Regression Discontinuity Design.
- GEP = Gifted Education Programme.
- SAP = Special Assistance Plan.

## 1 Introduction

### 1.1 Background

- In Singapore, primary school admission prioritises children living within 1 km, creating strong demand for housing near “good” schools. This demand is capitalised into higher HDB resale prices, potentially reducing affordability and increasing spatial inequality by linking educational access to housing wealth.

### 1.2 Research Question

- This study aims to estimate the effect of proximity to “good” primary schools on HDB resale prices. Here, “good schools” are defined as schools with high perceived school desirability rather than purely objective academic performance. Specifically, it examines whether flats nearer to high-demand schools command a premium, controlling for other factors. Using the 1 km admission cutoff, it identifies whether price differences are driven by proximity rather than underlying housing characteristics, providing policy-relevant evidence on education’s impact on housing markets.
- The study evaluates its findings based on three criteria: whether the estimated proximity premium is robust across alternative specifications and definitions of school desirability, whether the regression discontinuity design provides credible causal evidence at the 1 km boundary, and whether the results yield meaningful insights for housing and education policy.

## 2 Data

### 2.1 Overview

- The analysis uses HDB resale transactions across Singapore (2013–2025), merged with school registration and geospatial accessibility data. The final dataset contains 291,419 observations and 51 variables. Each observation represents a resale transaction matched with housing, school, and locational characteristics.
- A detailed summary of all datasets, variables and data sources is provided in Appendix A.

### 2.2 HDB Resale Transaction Data

- HDB resale transaction data is sourced from data.gov.sg, and flat addresses are geocoded using the OneMap API, with only a small number of observations with missing coordinates removed.

### 2.3 Housing Price Index

- To account for inflation, resale prices are deflated using the HDB Resale Price Index (RPI). Real resale prices are constructed as:
- where indexes flats and indexes time.

### 2.4 School Registration and School Characteristics Data

- School desirability is measured using Primary 1 registration outcomes. The number of applicants and vacancies in Phases 2B and 2C are used, as they better reflect parental demand than earlier priority-based phases, which depend on institutional ties like siblings, alumni, and staff affiliation.
- Ballot intensity and persistence capture short- and long-term demand, while Gifted Education Programme (GEP) and Special Assistance Plan (SAP) participation proxy school quality. School locations are obtained via OneMap with manual corrections where needed.

### 2.5 Accessibility and Amenity Data

- To control for neighbourhood accessibility, geospatial data on transport and amenities are used. Bus stop and MRT locations come from LTA DataMall, hawker centres from data.gov.sg, and shopping malls are separately geocoded.
- Distances are calculated using the Haversine formula, which measures great-circle distance and accounts for Earth’s curvature, providing more accurate city-wide proximity measures than Euclidean distance.
- Distance to the Central Business District is proxied by Raffles Place MRT station, reflecting its role as Singapore’s primary financial centre (National Library Board, 2023).

## 3 Data Preprocessing and Feature Engineering

### 3.1 Dependent Variable

- The main dependent variable is the logarithm of real resale price per square foot:
- Price per square foot normalises prices by unit size, ensuring comparability across flats and avoiding confounding size effects with location or school proximity.
- The log transformation reduces skewness, limits outlier influence, and stabilises estimates, while allowing coefficients to be interpreted as approximate percentage changes, which are more meaningful in a housing context.

### 3.2 Construction of the School Desirability Index (SDI)

- The SDI captures both the intensity and persistence of excess demand for each school.

#### 3.2.1 Ballot Intensity

- Ballot intensity is measured as the log ratio of applicants to vacancies in Phases 2B and 2C:
- where and denotes number of applicants, and and denotes number of vacancies for school in year .
- An across-phase ballot indicator is added to identify schools oversubscribed in both phases, reflecting consistently high demand:
- These are then combined, where BI is weighted more heavily as it captures how strong demand is, while the indicator captures widespread demand:
- Where:
- : standardisation

#### 3.2.2 Persistence of Demand

- To measure how consistently a school experiences excess demand, we compute a rolling ballot frequency:
- where:
- : indicator is 1 if balloting occurred in year , else 0
- averages ballot occurrences in phase 2B or 2C over the past five years

#### 3.2.3 School Desirability Index

- The SDI combines both dimensions, capturing current demand and its persistence over time:

### 3.3 Construction of the Good School Index (GSI)

- The Good School Index (GSI) assumes housing prices reflect perceived rather than purely objective school quality. Schools seen as offering better opportunities attract stronger admission demand, which, under Singapore’s distance-based admission system, is capitalised into higher nearby housing prices.
- The index combines demand indicators with GEP and SAP participation to capture both parental preferences and school characteristics. Greater weight is assigned to SDI, as it directly reflects demand intensity and households’ willingness to pay.
- Schools are classified as “good” if they fall in the upper tail of the GSI distribution. The baseline uses the 80th percentile, with 75, 85, and 90 tested for robustness.

### 3.4 School Proximity Variables

- Firstly, school density measures count the number of primary schools within specified distance bands from each flat:
- where denotes the distance between flat and school .
- Secondly, school desirability measures count the number of schools above each GSI percentile threshold within these distance bands, capturing exposure to highly desirable schools:
- Binary indicators are also constructed to reflect whether at least one such school exists within each band, providing a more interpretable measure of access:

### 3.5 Mature Estate Indicator

- A binary variable indicating whether a flat is in a mature town. The towns can be found in Appendix B.

### 3.6 Flat Type Mapping

- Flat type is converted into an ordinal variable based on size, with larger flats assigned higher values to better capture its relationship with resale price.

### 3.7 Categorisation of Storey Range

- Storey range is converted to a midpoint floor level, then normalised within each town using percentile ranks to account for differences in building height. Flats are grouped into:
- LOW_IN_ESTATE: Bottom 33% of floors within the town
- MID_IN_ESTATE: Middle 33%
- HIGH_IN_ESTATE: Top 33%

## 4 Methodology(485+144)

- This study adopts a two-pronged empirical strategy combining a hedonic regression model with a regression discontinuity design (RDD). The hedonic regression serves as the primary framework to estimate the relationship between HDB resale prices and proximity to desirable primary schools across the full sample, controlling for structural and locational factors. However, estimates may still reflect omitted variable bias due to unobserved neighbourhood quality.
- To strengthen causal interpretation, an RDD is implemented around the 1 km admission cutoff. By comparing flats just inside and outside the boundary, it identifies the local causal effect of priority admission eligibility. Together, the hedonic model captures broad, market-wide price gradients associated with school proximity, while the RDD isolates the causal effect of the admission policy at the cutoff, helping to distinguish policy-driven effects from purely correlational patterns.
- The validity and interpretation of these empirical strategies depend on several key assumptions.
- Second, school desirability is proxied using the Good School Index (GSI). This assumes that the index captures both observed parental demand and underlying school characteristics that shape perceptions. If this proxy is mis-specified, the classification of “good schools” and the interpretation of estimated proximity premiums would be affected.

### 4.1 Hedonic Regression Model Specification

- The baseline specification is:
- Where:
- : Log real price per square foot
- : Indicator for flats within 1 km of a desirable school
- : Indicator for flats within 1–2 km of a desirable school
- : Vector for control variables
- : error term
- The coefficients and represent percentage price premiums for being within each distance band, holding all other factors constant.
- To improve model design and assess robustness, the hedonic regression is estimated sequentially. The baseline includes flat characteristics, followed by locational controls.
- Finally, school availability measures (e.g., number of nearby primary schools) are included to distinguish proximity to desirable schools from general access. This stepwise approach tests whether the estimated proximity premium remains stable with additional controls, strengthening the credibility of the results.

### 4.2 Regression Discontinuity Design

- The baseline RDD specification is:
- Where:
- : Log real price per square foot
- : Indicator for flats within 1 km of a desirable school (treatment)
- : Distance to the nearest desirable school (running variable, centred at 1 km cutoff)
- : Linear function of distance
- : error term
- The coefficient captures the local treatment effect at the 1 km cutoff, interpreted as the causal impact of eligibility for the Primary 1 priority admission policy on resale prices. It reflects the price difference between flats just within and just outside the boundary.
- To improve identification and account for potential confounding, the RDD is extended as follows:
- Where:
- : School fixed effects
- : Vector of control variables
- This specification strengthens the design by controlling for school-level heterogeneity using fixed effects and improving inference through clustering at the school level.

## 5 Results

### 5.1 Hedonic Regression Model(762-150)

#### 5.1.1 Empirical Results

- Model
- Specification
- 0–1km Premium (%)
- 1–2km Premium (%)
- Adj R²
- Model 1
- FE only
- 0.73**
- 1.66**
- 0.778
- 291,419
- Model 2
- + Accessibility
- 0.87**
- 0.75**
- 0.822
- 291,419
- Model 3
- + School Density
- 1.16**
- 0.41**
- 0.823
- 291,419
- Table 1: Results of Hedonic Regression
- Notes: ** denotes statistical significance at the 5% level (p < 0.05)
- Model 1 produces counterintuitive results, with a larger premium for 1-2 km than 0-1 km, suggesting omitted variable bias from unaccounted locational factors.
- After introducing accessibility controls (Model 2), the 0-1 km premium rises while the 1-2 km premium falls, indicating that part of the earlier effect was driven by transport and amenities.
- With school density controls (Model 3), the estimates stabilise, with the premium concentrated within 1 km and a smaller effect for 1-2 km.
- Model 3 is adopted as the preferred specification, as it accounts for both locational accessibility and school availability, thereby isolating the premium associated with proximity to desirable schools in a manner consistent with the institutional admission framework.

#### 5.1.2 Robustness checks

#### 5.1.2.1 Alternative Definitions of “Good School” (Threshold Robustness)

- To test sensitivity to the definition of “good schools”, the hedonic regression was re-estimated using SDI thresholds of 75, 80, 85, and 90.
- Threshold
- Premium (0–1 km)
- Premium (1–2 km)
- Adj. R²
- 0.46%**
- 0.80%**
- 0.8227
- 291,419
- 1.16%**
- 0.41%**
- 0.8229
- 291,419
- 1.27%**
- 0.34%**
- 0.8230
- 291,419
- 0.67%**
- 0.12%**
- 0.8226
- 291,419
- Table 2: Results of Threshold Robustness
- At the 75 threshold, results are less precise, with the 1–2 km premium exceeding the 0–1 km premium, suggesting the classification is too broad and captures general school access rather than the true school quality.
- At thresholds 80 and 85, estimates are more consistent, with a stronger premium within 1 km and a smaller effect for 1–2 km.
- At the 90 threshold, both premiums decline, likely due to fewer schools being included.
- Overall, results are robust to threshold variation, with 80-85 providing the best balance between capturing desirable schools and preserving sufficient variation.

#### 5.1.2.2 Alternative Weights in GSI

- This robustness check tests whether the estimated premium depends on the GSI weighting scheme, while keeping SDI as the dominant component relative to GEP and SAP.
- GSI Weighting Scheme
- Premium within 0–1 km
- Premium within 1–2 km
- Adj. R²

### 0.7 SDI, 0.15 GEP, 0.15 SAP

- 1.17%**
- 0.50%**
- 0.8230
- 291,419

### 0.8 SDI, 0.1 GEP, 0.1 SAP

- 1.09%**
- 0.49%**
- 0.8229
- 291,419

### 0.6 SDI, 0.2 GEP, 0.2 SAP

- 1.16%**
- 0.41%**
- 0.8229
- 291,419
- Table 4: Results of Alternative weights in GSI
- The similar estimates suggest results are not sensitive to GSI weighting, as long as SDI remains dominant. The consistency across specifications supports using an SDI-led GSI, capturing parental demand while incorporating school characteristics through GEP and SAP.

#### 5.1.3 Heterogeneity Analysis

#### 5.1.3.1 Flat-Type Heterogeneity

- Flat Type Group
- 0–1 km
- 1–2 km
- 1–3 room flats
- +0.43%**
- −2.22%**
- 4-room and above flats
- +1.33%**
- +1.28%**
- Table 5: Heterogeneity in School Proximity Premium
- To assess variation by household types, interaction terms between school proximity and flat type are included. Larger flats are interpreted as a proxy for family households, who are more sensitive to school access.
- Results show a stronger premium for larger flats. In particular, the interaction terms for larger flat types are positive and statistically significant, indicating that households residing in these units are willing to pay a higher premium to be near desirable primary schools. In contrast, for 1–3 room flats, the premium is smaller within 1 km and negative for 1–2 km.
- Overall, the school premium varies by household type and is driven more by family-oriented buyers.

#### 5.1.3.2 Mature Estates Heterogeneity

- Estate Type Group
- 0–1 km
- 1–2 km
- Non-mature estates
- +0.93%**
- +0.09%
- Mature estates
- +1.19%**
- +1.82%**
- Table 6: Results of mature estates vs non-mature estates
- To examine heterogeneity across estate types, the sample is split into mature and non-mature estates, with regressions estimated separately while retaining town fixed effects.
- The results show that within the 0–1 km band, the price premium is similar across both groups. In contrast, within the 1–2 km band, a statistically significant premium is observed only in mature estates, while the effect in non-mature estates is small and not statistically significant. This indicates that while the immediate proximity premium is comparable across estate types, the influence of school proximity extends over a wider distance in mature estates

#### 5.1.4 Hedonic Results Overview

- The hedonic regression results provide consistent evidence that proximity to desirable primary schools is capitalised into HDB resale prices. The model estimates an average premium of about 1.16% within 1 km and 0.41% within 1–2 km.
- Robustness checks confirm the stability of this finding, with premiums remaining consistent across alternative definitions of “good” schools and GSI weighting schemes. Additional analysis using school counts further shows that the marginal value of additional good schools is concentrated within the 1 km band, reinforcing the importance of close proximity.
- Heterogeneity results indicate that the premium is stronger for larger flats, suggesting that school-related housing demand is driven mainly by family households. While the 1 km premium is similar across estate types, it persists over a wider range in mature estates.
- Overall, while the hedonic results identify an average premium across distance bands, this effect is not uniform across households or spatial contexts.

### 5.2 Regression Discontinuity Design(610)5.2.1 Empirical Results of Baseline RDD Model

- Model
- Specification
- 0–1km Premium (%)
- 1–2km Premium (%)
- Baseline
- -1.83**
- -4.36**
- Table X: Baseline RDD Results
- The baseline RDD estimates indicate a statistically significant negative price discontinuity at both distance-boundary cutoffs. Because treatment is coded as , a negative therefore suggests resale flats located just inside the 1km zone are cheaper than otherwise similar flats just outside the boundary.

#### 5.2.1.2 Covariate Balance

- Covariate balance tests re-run the same RDD using pre-treatment or predetermined covariates as outcomes. If the design is locally comparable, these covariates should be continuous at the cutoff.
- Covariate (1km cutoff)
- Jump
- Significance
- Floor Area
- 3.80%
- Significant
- Remaining lease
- -0.20%
- Not significant
- Distance to MRT
- 0.66%
- Not significant
- Distance to mall
- -2.21%
- Significant
- Distance to hawker
- 0.98%
- Not significant
- Distance to bus stop
- 0.21%
- Significant
- Count of all schools (0-1km)
- 1.10%
- Significant
- Count of all schools (1-2km)
- -1.10%
- Significant
- Table X: Baseline RDD Covariate Balance Test Result
- Several significant jumps indicate imperfect local balance in the baseline design, so baseline causal interpretation should be treated cautiously. Refer to Appendix X for detailed robustness checks on the baseline RDD.

#### 5.2.2 Empirical Results of Extended RDD Model

- The extended RDD introduces school fixed effects and clustered standard errors to tighten identification by comparing flats around boundaries relative to the same nearest desirable school.
- Model
- Specification
- 0–1km Premium
- 1–2km Premium
- Extended
- -1.71%
- 0.60%
- Table X: Extended RDD Result
- None of the extended-model estimates is statistically significant at 5%. Relative to baseline, this indicates weaker evidence of a stable discontinuity once school-level heterogeneity is absorbed.

#### 5.2.3 Robustness checks5.2.3.1 Alternative Definitions of “Good” Schools (Threshold Robustness)

- Threshold
- Premium (0–1 km)
- Premium (1–2 km)
- -0.77%
- 5.55%
- -1.71%
- 0.60%
- -1.52%
- 0.76%
- -0.55%
- 0.90%
- Table X: Extended RDD with different “Good” School Thresholds
- While estimates are statistically insignificant across all thresholds, the direction of coefficients remains broadly consistent. This suggests that the null result is robust to alternative definitions of school desirability. However, the lack of statistical significance indicates insufficient evidence of a causal price discontinuity.5.2.3.2 Bandwidth(17)
- Bandwidth (km)
- Premium (0–1 km)
- Premium (1–2 km)
- -2.54%**
- 0.99%
- -1.71%
- 0.60%
- -1.04%
- -0.36%
- -0.52%
- 0.24%
- Table X: Extended RDD with Various Bandwidth
- Only the narrowest 1 km estimate is significant. This indicates weak bandwidth robustness in the extended specification.

#### 5.2.3.3 Control Variables

- Model
- Specification
- 0–1km Premium (%)
- 1–2km Premium (%)
- Extended
- -1.71%
- 0.60%
- Adjusted
- +Controls
- -0.87%
- -0.74%
- Table X: Comparing Extended RDD and Added Controls
- Adding controls weakens the 1 km estimate and flips the sign at 2 km, while remaining statistically insignificant. This points to poor covariate balance, suggesting imperfect local balance in the baseline design, so extended causal interpretation should be treated cautiously.

#### 5.2.3.4 Covariate Balance

- Covariate (1km cutoff)
- Jump
- Significance
- Floor Area
- 2.88%
- Significant
- Remaining lease
- -0.17%
- Not significant
- Distance to MRT
- 1.84%
- Not significant
- Distance to mall
- -4.38%
- Not significant
- Distance to hawker
- -1.76%
- Not significant
- Distance to bus stop
- 0.18%
- Not significant
- Count of all schools (0-1km)
- 1.09%
- Significant
- Count of all schools (1-2km)
- -1.05%
- Significant
- Table X: Extended RDD Covariate Balance Test Result
- Balance improves relative to baseline for most amenity covariates, though discontinuities remain for floor area and school-density counts. Refer to Appendix X for further robustness checks supporting the improved model.

#### 5.2.4 RDD Results Overview

- Across all specifications, the RDD finds no statistically significant price discontinuity at the 1 km cutoff. Results are robustly insignificant across alternative definitions of school desirability and donut specifications, and placebo tests show no spurious effects. However, estimates are sensitive to bandwidth choice and the inclusion of controls, and balance tests indicate residual covariate differences at the cutoff. Overall, there is no strong or stable evidence that eligibility for the 1 km priority rule leads to a price discontinuity.

## 6 Policy Implications

- From a Ministry of National Development (MND) perspective, the findings highlight how education-related factors can influence housing market outcomes. The hedonic results suggest that perceived school quality is capitalised into housing values and may reduce affordability.
- While the RDD finds no statistically significant causal effect of the 1 km policy, the broader relationship between school desirability and demand remains evident. This suggests that price differences are driven more by underlying neighbourhood characteristics and perceived school reputation than the policy itself. This indicates that spatial inequality may arise from concentrated demand around certain schools and estates, rather than discrete policy effects at the boundary.
- The heterogeneity analysis further shows that this demand is not uniform. The school effect is stronger for larger flats, indicating that family-oriented households are more sensitive to school access and drive a disproportionate share of price pressure. In addition, while the effect within 1 km is similar across estates, it persists over a wider range in mature estates, suggesting that school-related demand extends beyond the immediate priority zone in these areas.
- These findings underscore the importance of integrated planning between housing and education agencies. For MND, this includes addressing uneven demand across locations by improving estate attractiveness and working with the Ministry of Education to better manage school demand and perceptions.
- Targeted housing supply measures can also help alleviate pressure. Increasing the supply of larger flats in high-demand areas may address the needs of households most affected by school access considerations, while improving school quality and neighbourhood qualities in non-mature estates can help redistribute demand more evenly.
- Overall, policy should focus on managing the spatial and demographic concentration of school-related housing demand, rather than relying on adjustments to admission thresholds alone.

## 7 Limitations and Constraints

- Accessibility measures rely on current amenity locations rather than historical infrastructure at the time of transaction. School relocations and boundary changes are not captured, introducing measurement error, especially for earlier years, which may attenuate estimates. Future work could incorporate historical data to improve accuracy.
- Hedonic regression may suffer from omitted variable bias, as unobserved neighbourhood factors (e.g., prestige, environment, community) may affect both prices and school desirability. Hence, estimated premiums may partly reflect these factors.
- Housing prices in Singapore are influenced by policy interventions such as cooling measures, including the Total Debt Servicing Ratio (TDSR), which restricts borrowing and limits buyers’ ability to bid for preferred locations. These constraints can dampen price variation, reducing the extent to which willingness to pay for proximity to desirable primary schools is reflected in resale prices. As a result, the estimated school proximity premium may understate the underlying demand for school access.

## 8 Conclusion

- This study examines the relationship between proximity to desirable primary schools and HDB resale prices using hedonic regression and RDD. Hedonic results show a clear price premium near “good” schools, reflecting strong demand. However, the RDD finds no statistically significant causal effect at the 1 km admission boundary, suggesting the premium is driven by broader neighbourhood factors rather than the policy.
- From a policy perspective, the findings highlight how perceived school quality shapes housing demand and contributes to spatial inequality, even without a discrete policy-induced price effect. While the 1 km priority rule does not create a sharp price discontinuity, demand pressures around desirable schools may still influence housing outcomes. Future research could incorporate richer measures of school quality, historical infrastructure data, and alternative identification strategies to better isolate causal effects.

## 9 References

- National Library Board. (2023). Raffles Place. https://www.nlb.gov.sg/main/article-detailcmsuuid=b62dd9d7-ea3c-42b4-a3b7-fd97d601cea4

## 10 Appendix

- Appendix A
- Dataset
- Source
- Link
- Key variables
- Purpose
- HDB resale flat transactions
- data.gov.sg
- https://data.gov.sg/collections/189/view
- resale price, flat type, floor area, town, lease, storey range
- Main housing dataset
- HDB Resale Price Index (RPI)
- Housing & Development Board
- https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics
- quarterly price index
- Deflation of nominal housing prices
- Primary school registration outcomes
- Aggregated school registration sources
- https://sgschooling.com/year/
- applicants, vacancies, ballot outcomes
- Construction of school desirability measures
- School characteristics (GEP, SAP)
- School-level information
- https://www.property2b2c.com/school-ranking/bespoke
- programme participation indicators
- Proxy for institutional prestige
- Bus stop locations
- LTA DataMall
- https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html
- latitude, longitude
- Accessibility control
- MRT station data
- LTA DataMall
- https://datamall.lta.gov.sg/content/datamall/en/static-data.html
- station coordinates
- Accessibility control and CBD reference
- Hawker centre dataset
- data.gov.sg
- https://data.gov.sg/datasetsquery=hawker
- location coordinates
- Neighbourhood amenity proxy
- Shopping mall locations
- Compiled list and geocoded
- https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore
- mall coordinates
- Commercial accessibility proxy
- Geocoding service
- OneMap API
- https://www.onemap.gov.sg
- latitude, longitude
- Spatial matching of flats and schools
- Appendix B
- Mature Estate
- Ang Mo Kio
- Bedok
- Bishan
- Bukit Merah
- Bukit Timah
- Central Area
- Geylang
- Kallang/Whampoa
- Marine Parade
- Queenstown
- Serangoon
- Toa Payoh
- Appendix
- Threshold
- Premium (0–1 km)
- Premium (1–2 km)
- -0.12%
- 6.43%**
- -1.83%**
- -4.36%**
- -2.04%**
- -3.95%**
- -0.68%**
- -3.52%**
- Baseline RDD estimates vary in magnitude and sign across thresholds, indicating instability to school desirability definitions.
- Bandwidth (km)
- Premium (0–1 km)
- Premium (1–2 km)
- -3.55%**
- -4.38%**
- -1.83%**
- -4.36%**
- -1.16%**
- -3.95%**
- -0.79%**
- -3.52%**
- For the base RDD, the negative discontinuity weakens as bandwidth increases, indicating a stronger local effect near the cutoff.
- Model
- Specification
- 0–1km Premium (%)
- 1–2km Premium (%)
- Baseline
- -1.83**
- -4.36**
- Adjusted
- +Controls
- -0.32%**
- 0.02%
- Adding controls reduces the discontinuity and removes significance, indicating the baseline effect reflects compositional differences rather than a true boundary effect.
- Appendix
- Donut Hole (km)
- Premium
- 0.05
- -1.39%
- 0.10
- -0.39%
- The extended RDD donut test shows that excluding observations near the cutoff does not affect results, indicating no sorting bias and no causal effect.
- Placebo Cutoff (km)
- Premium
- 0.58%
- 1.50%
- The extended RDD placebo test shows no spurious discontinuities, supporting model validity but confirming no causal effect at the cutoff.
