# Academic Structure Discussion - Materials Science and Mathematical Methods

## Original Question

I have one doubt regarding how to properly distribute the concepts I have been working on lately (this past few years). I have thought I could enumerate some of them and could you help me distribute the concepts and give them an academic structure? If you feel like there is something missing just add it or suggest it. Shall I begin?

I have been working on simulating phase transformation, hot working and mechanical properties mainly. These three topics have been tackled the following way:
- Hot working: I have optimized a model which simulates torsion tests stress strain curves. This optimization has been done following least squares curve fitting. This stress strain behavior has also been done with a diverse set of medium carbon steels.
   - torsion tests for a variety of  low and medium carbon steels. Implementation and optimization of a model that integrates Estrin-Mecking and Avrami's equations.
- Phase transformation: Phase transformation modelling taking into account homogeneous and heterogeneous chemical microstructure.
   - Implementation of Kirkaldy-based phase transformation models for the prediction of final phase volume fractions in 42CrMo4 steel.
   - Use of Nonlinear Least Squares methods and Evolutionary Algorithms (EAs) for parameter optimization.
   - Extension of the models for chemical heterogeneity consideration.
   - Kinetics of phase transformation (bainite, ferrite) 
Mechanical properties: 
   - Estimation of mechanical properties by means of diverse analytical methods.
   - Modelling of the stress-strain behavior of tempered martensitic materials for medium carbon steels.
   - Development of X-Ray spectra analysis tools for mechanical properties calculation (crystallite size, dislocation density).
      - This required the implementation and study of the williamson-hall and warren-averbach traditional methods and their modified counterparts
   - Study of correlations and significance of mechanical properties and process attributes in relation to asymmetry phenomena.
   - Development of data models by means of machine learning methods (regularized regression, random forest, gradient boosting, SVM, kNN, ...) 

The thing is in the development of this models, either when optimizing or designing surrogate models I have been working with various machine learning techniques. I have tackled:
- Diverse optimization algorithms: Simulated annealing, Enhanced stochastic evolution (ESE) algorithm, ...
- Machine learning regression techniques: Linear regression, regularized regression (Lasso, Ridge, Elastic net), Random forest, Gradient boosting, kNN regression, SVM regression.
- Feature selection (recursive feature selection and sequential feature selection)
- Techniques for outlier detection:
   - One dimensional: IQR, Z-score, robust Z-score
   - Multidimensional: Mahalanobis distance, kNN, local outlier factor, Isolation forest.
- Other preprocessing techniques like PCA, clustering via different techniques:  K-means, DBSCAN, HDBSCAN (and the techniques to measure how many clusters use (Elbow rule, silhouette method)
- DOE with analytical methods. Which uses different metrics derived from geometry and statistics:
   - Space filling criterions: Like minimax and maximins distances, MaxPro criterion, Riesz-energy distance.
   - Projection metrics: MaxPro criterion, usage of Latin Hypercube Designs, ...
   - Uniformity/independence: L2 star and centered discrepancies. 

The thing is that I am working on material science, but I also am a mathematician, so I have interest in not only describing the physical processes which I am trying to emulate, but to give special focus to the numerical methods/mathematical concepts which fuel the procedures/algorithms that I implement.

How would you summarize all of this in a table of contents, for example? Again if you feel like there is something I should add, or an intermediary concept that I could define or append on a section, just add it. 

Imagine I intend to write a kind of book and I want to design its structure

---

## Proposed Academic Structure

Based on your work, I'd suggest a structure that balances your dual expertise in materials science and mathematics. Here's a comprehensive table of contents that gives proper weight to both domains:

# **Computational Methods for Materials Science: A Mathematical Approach to Phase Transformation, Hot Working, and Mechanical Property Prediction**

## **Part I: Foundations**

### **Chapter 1: Introduction**
1.1 Motivation and Scope
1.2 The Intersection of Materials Science and Computational Mathematics
1.3 Overview of the Integrated Approach
1.4 Structure of this Work

### **Chapter 2: Mathematical Foundations**
2.1 Optimization Theory
   - 2.1.1 Unconstrained Optimization
   - 2.1.2 Constrained Optimization
   - 2.1.3 Convergence Criteria and Stopping Conditions
2.2 Least Squares Methods
   - 2.2.1 Linear Least Squares
   - 2.2.2 Nonlinear Least Squares
   - 2.2.3 Regularization Techniques
2.3 Statistical Learning Theory
   - 2.3.1 Bias-Variance Tradeoff
   - 2.3.2 Cross-Validation
   - 2.3.3 Model Selection Criteria
2.4 Numerical Analysis Considerations
   - 2.4.1 Numerical Stability
   - 2.4.2 Computational Complexity
   - 2.4.3 Error Analysis

### **Chapter 3: Materials Science Background**
3.1 Steel Metallurgy Essentials
   - 3.1.1 Carbon and Alloying Elements
   - 3.1.2 Microstructural Phases
   - 3.1.3 Processing Routes
3.2 Phase Transformation Fundamentals
   - 3.2.1 Thermodynamics of Phase Transformations
   - 3.2.2 Nucleation and Growth Mechanisms
   - 3.2.3 TTT and CCT Diagrams
3.3 Hot Working Processes
   - 3.3.1 Dynamic Recovery and Recrystallization
   - 3.3.2 Flow Stress Behavior
   - 3.3.3 Microstructural Evolution During Deformation
3.4 Mechanical Properties of Steels
   - 3.4.1 Strength, Ductility, and Toughness
   - 3.4.2 Microstructure-Property Relationships
   - 3.4.3 X-Ray Diffraction Fundamentals

---

## **Part II: Data Preprocessing and Experimental Design**

### **Chapter 4: Data Quality and Preprocessing**
4.1 Data Collection and Validation
4.2 Outlier Detection Methods
   - 4.2.1 Univariate Methods
      - 4.2.1.1 Interquartile Range (IQR) Method
      - 4.2.1.2 Z-Score and Robust Z-Score
   - 4.2.2 Multivariate Methods
      - 4.2.2.1 Mahalanobis Distance
      - 4.2.2.2 k-Nearest Neighbors (kNN) Based Detection
      - 4.2.2.3 Local Outlier Factor (LOF)
      - 4.2.2.4 Isolation Forest
   - 4.2.3 Comparative Analysis and Selection Criteria
4.3 Data Transformation and Normalization
   - 4.3.1 Scaling Methods
   - 4.3.2 Box-Cox and Power Transformations
4.4 Dimensionality Reduction
   - 4.4.1 Principal Component Analysis (PCA)
   - 4.4.2 Variance Explained and Component Selection
   - 4.4.3 Interpretation in Physical Context

### **Chapter 5: Clustering and Pattern Recognition**
5.1 Clustering Fundamentals
5.2 Partitioning Methods
   - 5.2.1 K-Means Clustering
   - 5.2.2 Cluster Number Selection: Elbow Method
   - 5.2.3 Cluster Number Selection: Silhouette Method
5.3 Density-Based Methods
   - 5.3.1 DBSCAN Algorithm
   - 5.3.2 HDBSCAN: Hierarchical Density-Based Clustering
   - 5.3.3 Parameter Selection and Sensitivity Analysis
5.4 Applications in Materials Data Analysis

### **Chapter 6: Design of Experiments (DOE) for Materials Testing**
6.1 Classical DOE Methods
   - 6.1.1 Factorial Designs
   - 6.1.2 Response Surface Methodology
6.2 Space-Filling Designs
   - 6.2.1 Latin Hypercube Sampling (LHS)
   - 6.2.2 Quasi-Random Sequences
6.3 Optimality Criteria for Computer Experiments
   - 6.3.1 Distance-Based Criteria
      - 6.3.1.1 Minimax Distance
      - 6.3.1.2 Maximin Distance
      - 6.3.1.3 Riesz Energy Distance
   - 6.3.2 Projection Properties
      - 6.3.2.1 Maximum Projection (MaxPro) Criterion
   - 6.3.3 Uniformity Measures
      - 6.3.3.1 L2-Star Discrepancy
      - 6.3.3.2 Centered L2 Discrepancy
6.4 Adaptive Sampling Strategies
6.5 Implementation and Case Studies

---

## **Part III: Optimization Algorithms**

### **Chapter 7: Deterministic Optimization Methods**
7.1 Gradient-Based Methods
   - 7.1.1 Steepest Descent
   - 7.1.2 Newton and Quasi-Newton Methods
   - 7.1.3 Levenberg-Marquardt Algorithm for Nonlinear Least Squares
7.2 Direct Search Methods
7.3 Trust Region Methods
7.4 Convergence Analysis and Numerical Considerations

### **Chapter 8: Stochastic and Metaheuristic Optimization**
8.1 Simulated Annealing
   - 8.1.1 Thermodynamic Analogy
   - 8.1.2 Cooling Schedules
   - 8.1.3 Implementation and Parameter Tuning
   - 8.1.4 Convergence Properties
8.2 Evolutionary Algorithms
   - 8.2.1 Genetic Algorithms Fundamentals
   - 8.2.2 Enhanced Stochastic Evolution (ESE) Algorithm
      - 8.2.2.1 Mathematical Formulation
      - 8.2.2.2 Advantages Over Standard EAs
      - 8.2.2.3 Implementation Details
   - 8.2.3 Differential Evolution
   - 8.2.4 Selection, Crossover, and Mutation Operators
8.3 Hybrid Optimization Strategies
8.4 Comparative Performance Analysis
8.5 Applications to Materials Model Parameter Identification

---

## **Part IV: Machine Learning Methods**

### **Chapter 9: Regression Techniques**
9.1 Linear Regression
   - 9.1.1 Ordinary Least Squares (OLS)
   - 9.1.2 Assumptions and Diagnostics
   - 9.1.3 Residual Analysis
9.2 Regularized Regression
   - 9.2.1 Ridge Regression (L2 Regularization)
   - 9.2.2 Lasso Regression (L1 Regularization)
   - 9.2.3 Elastic Net
   - 9.2.4 Hyperparameter Selection via Cross-Validation
9.3 Support Vector Regression (SVR)
   - 9.3.1 Kernel Methods
   - 9.3.2 ε-Insensitive Loss Function
   - 9.3.3 Parameter Tuning
9.4 k-Nearest Neighbors (kNN) Regression
   - 9.4.1 Distance Metrics
   - 9.4.2 Neighbor Selection Strategies
9.5 Polynomial and Spline Regression

### **Chapter 10: Tree-Based Ensemble Methods**
10.1 Decision Trees for Regression
   - 10.1.1 Splitting Criteria
   - 10.1.2 Pruning Strategies
10.2 Random Forest Regression
   - 10.2.1 Bootstrap Aggregating (Bagging)
   - 10.2.2 Feature Randomness
   - 10.2.3 Out-of-Bag Error Estimation
   - 10.2.4 Variable Importance Measures
10.3 Gradient Boosting Methods
   - 10.3.1 Gradient Boosting Machines (GBM)
   - 10.3.2 XGBoost, LightGBM, and CatBoost
   - 10.3.3 Regularization in Boosting
   - 10.3.4 Learning Rate and Number of Iterations
10.4 Comparative Analysis and When to Use Each Method

### **Chapter 11: Feature Engineering and Selection**
11.1 Feature Engineering Strategies
   - 11.1.1 Polynomial Features
   - 11.1.2 Interaction Terms
   - 11.1.3 Domain-Specific Feature Construction
11.2 Feature Selection Methods
   - 11.2.1 Filter Methods
      - 11.2.1.1 Correlation Analysis
      - 11.2.1.2 Mutual Information
   - 11.2.2 Wrapper Methods
      - 11.2.2.1 Recursive Feature Elimination (RFE)
      - 11.2.2.2 Sequential Feature Selection (Forward/Backward)
   - 11.2.3 Embedded Methods
      - 11.2.3.1 Lasso-Based Selection
      - 11.2.3.2 Tree-Based Feature Importance
11.3 Multicollinearity and Variance Inflation Factor (VIF)
11.4 Feature Selection in High-Dimensional Spaces

### **Chapter 12: Model Evaluation and Validation**
12.1 Performance Metrics for Regression
   - 12.1.1 MSE, RMSE, MAE, MAPE
   - 12.1.2 R² and Adjusted R²
   - 12.1.3 Domain-Specific Metrics
12.2 Cross-Validation Strategies
   - 12.2.1 k-Fold Cross-Validation
   - 12.2.2 Leave-One-Out Cross-Validation
   - 12.2.3 Time Series Cross-Validation
12.3 Overfitting and Underfitting Detection
12.4 Learning Curves and Validation Curves
12.5 Statistical Significance Testing
12.6 Uncertainty Quantification in Predictions

---

## **Part V: Applications to Hot Working**

### **Chapter 13: Hot Working Process Modeling**
13.1 Flow Stress Behavior in Hot Working
13.2 Torsion Testing: Experimental Methodology
13.3 Physical Mechanisms During Hot Deformation
   - 13.3.1 Work Hardening
   - 13.3.2 Dynamic Recovery
   - 13.3.3 Dynamic Recrystallization

### **Chapter 14: Estrin-Mecking and Avrami Formulation**
14.1 Estrin-Mecking Model for Work Hardening and Recovery
   - 14.1.1 Dislocation Density Evolution
   - 14.1.2 Mathematical Formulation
   - 14.1.3 Physical Interpretation of Parameters
14.2 Avrami Equation for Dynamic Recrystallization
   - 14.2.1 Johnson-Mehl-Avrami-Kolmogorov (JMAK) Theory
   - 14.2.2 Time Exponent and Physical Meaning
14.3 Integrated Model Formulation
   - 14.3.1 Coupling Work Hardening and Recrystallization
   - 14.3.2 State Variable Evolution
14.4 Constitutive Equations for Flow Stress

### **Chapter 15: Parameter Optimization for Hot Working Models**
15.1 Problem Formulation
   - 15.1.1 Objective Function Definition
   - 15.1.2 Parameter Bounds and Constraints
15.2 Optimization Strategy
   - 15.2.1 Sensitivity Analysis
   - 15.2.2 Multi-Start Approaches
   - 15.2.3 Global vs Local Optimization
15.3 Applications to Low and Medium Carbon Steels
   - 15.3.1 Material Specifications
   - 15.3.2 Experimental Data
   - 15.3.3 Optimization Results
   - 15.3.4 Model Validation
15.4 Comparative Analysis Across Steel Grades
15.5 Predictive Capability and Extrapolation

---

## **Part VI: Applications to Phase Transformation**

### **Chapter 16: Phase Transformation Modeling Framework**
16.1 Overview of Phase Transformation Models
16.2 Kinetics of Diffusional Transformations
   - 16.2.1 Ferrite Formation
   - 16.2.2 Pearlite Formation
   - 16.2.3 Bainite Formation
16.3 Chemical Driving Force and Undercooling
16.4 Nucleation and Growth Kinetics

### **Chapter 17: Kirkaldy-Based Models**
17.1 Kirkaldy's Diffusion Model
   - 17.1.1 Theoretical Foundation
   - 17.1.2 Diffusion Equations in Multicomponent Systems
17.2 Phase Fraction Evolution
   - 17.2.1 Interface Velocity
   - 17.2.2 Carbon Diffusion Control
17.3 Model Implementation
   - 17.3.1 Numerical Solution Techniques
   - 17.3.2 Time Integration Schemes
17.4 Application to 42CrMo4 Steel
   - 17.4.1 Material Characterization
   - 17.4.2 Experimental Validation Data
   - 17.4.3 Model Predictions vs Experiments

### **Chapter 18: Chemical Heterogeneity in Phase Transformation**
18.1 Sources of Chemical Heterogeneity
   - 18.1.1 Microsegregation During Solidification
   - 18.1.2 Dendritic Structure and Interdendritic Regions
18.2 Modeling Approaches for Heterogeneous Microstructures
   - 18.2.1 Statistical Distribution of Composition
   - 18.2.2 Multi-Domain Models
   - 18.2.3 Volume Averaging Methods
18.3 Extended Kirkaldy Model for Chemical Heterogeneity
   - 18.3.1 Model Formulation
   - 18.3.2 Discretization of Composition Space
   - 18.3.3 Integration Over Composition Distribution
18.4 Impact on Transformation Kinetics and Final Microstructure
18.5 Validation and Case Studies

### **Chapter 19: Parameter Identification for Phase Transformation Models**
19.1 Inverse Problem Formulation
19.2 Nonlinear Least Squares Approach
   - 19.2.1 Gauss-Newton Method
   - 19.2.2 Levenberg-Marquardt Algorithm
   - 19.2.3 Jacobian Calculation
19.3 Evolutionary Algorithm Approach
   - 19.3.1 Problem Encoding
   - 19.3.2 Fitness Function Design
   - 19.3.3 Algorithm Implementation
19.4 Hybrid Optimization Strategies
19.5 Comparison of Optimization Methods
19.6 Parameter Identifiability and Uncertainty
19.7 Results and Discussion

---

## **Part VII: Applications to Mechanical Properties**

### **Chapter 20: Analytical Estimation of Mechanical Properties**
20.1 Empirical and Semi-Empirical Relationships
20.2 Rule of Mixtures for Multiphase Materials
20.3 Hall-Petch Relationship
20.4 Strengthening Mechanisms
   - 20.4.1 Solid Solution Strengthening
   - 20.4.2 Precipitation Strengthening
   - 20.4.3 Dislocation Strengthening
   - 20.4.4 Grain Boundary Strengthening
20.5 Integration of Multiple Strengthening Contributions

### **Chapter 21: Stress-Strain Modeling of Tempered Martensitic Steels**
21.1 Martensitic Transformation and Tempering
21.2 Microstructure of Tempered Martensite
21.3 Constitutive Modeling Approaches
   - 21.3.1 Phenomenological Models
   - 21.3.2 Physically-Based Models
21.4 Model Development for Medium Carbon Steels
   - 21.4.1 Model Formulation
   - 21.4.2 Parameter Identification
   - 21.4.3 Validation Against Experimental Data
21.5 Influence of Tempering Temperature and Time
21.6 Predictive Capability and Applications

### **Chapter 22: X-Ray Diffraction Analysis for Microstructural Characterization**
22.1 X-Ray Diffraction Fundamentals
   - 22.1.1 Bragg's Law
   - 22.1.2 Peak Broadening Mechanisms
22.2 Traditional Methods for Line Profile Analysis
   - 22.2.1 Williamson-Hall Method
      - 22.2.1.1 Theoretical Basis
      - 22.2.1.2 Separation of Size and Strain Broadening
      - 22.2.1.3 Limitations
   - 22.2.2 Warren-Averbach Method
      - 22.2.2.1 Fourier Analysis of Diffraction Profiles
      - 22.2.2.2 Separation of Size and Strain Effects
      - 22.2.2.3 Implementation Considerations
22.3 Modified Methods
   - 22.3.1 Modified Williamson-Hall (mWH) Method
   - 22.3.2 Modified Warren-Averbach (mWA) Method
   - 22.3.3 Accounting for Dislocation Contrast Factors
22.4 Implementation of Analysis Tools
   - 22.4.1 Software Architecture
   - 22.4.2 Peak Fitting Algorithms
   - 22.4.3 Background Subtraction
   - 22.4.4 Kα2 Stripping
22.5 Crystallite Size Determination
22.6 Dislocation Density Calculation
   - 22.6.1 Relationship Between Strain and Dislocation Density
   - 22.6.2 Wilkens Model
22.7 Correlation with Mechanical Properties
22.8 Case Studies and Applications

### **Chapter 23: Asymmetry Phenomena in Materials Processing**
23.1 Definition and Manifestations of Asymmetry
23.2 Sources of Asymmetry in Processing
   - 23.2.1 Thermal Asymmetry
   - 23.2.2 Mechanical Asymmetry
   - 23.2.3 Microstructural Asymmetry
23.3 Statistical Analysis of Asymmetry
   - 23.3.1 Correlation Analysis
   - 23.3.2 Significance Testing (ANOVA, t-tests)
   - 23.3.3 Multivariate Analysis
23.4 Relationship to Mechanical Properties
23.5 Process Optimization to Minimize Asymmetry

### **Chapter 24: Machine Learning for Mechanical Property Prediction**
24.1 Data Collection and Database Structure
24.2 Feature Engineering for Materials Data
   - 24.2.1 Composition Features
   - 24.2.2 Processing Parameters
   - 24.2.3 Microstructural Features
24.3 Model Development and Comparison
   - 24.3.1 Linear and Regularized Models
   - 24.3.2 Tree-Based Models
   - 24.3.3 Support Vector Machines
   - 24.3.4 k-Nearest Neighbors
24.4 Feature Selection and Importance Analysis
24.5 Model Performance and Validation
24.6 Physical Interpretation of ML Models
24.7 Surrogate Modeling for Computational Efficiency
24.8 Active Learning and Model Updating

---

## **Part VIII: Integration and Advanced Topics**

### **Chapter 25: Integrated Process-Microstructure-Property Modeling**
25.1 Multi-Scale Modeling Framework
25.2 Coupling Hot Working, Phase Transformation, and Property Models
25.3 Sequential vs Concurrent Coupling
25.4 Computational Workflow
25.5 Case Study: Complete Processing Chain Simulation

### **Chapter 26: Uncertainty Quantification and Sensitivity Analysis**
26.1 Sources of Uncertainty
   - 26.1.1 Measurement Uncertainty
   - 26.1.2 Model Form Uncertainty
   - 26.1.3 Parameter Uncertainty
26.2 Probabilistic Approaches
   - 26.2.1 Monte Carlo Methods
   - 26.2.2 Latin Hypercube Sampling for Uncertainty Propagation
26.3 Global Sensitivity Analysis
   - 26.3.1 Variance-Based Methods (Sobol Indices)
   - 26.3.2 Morris Screening Method
26.4 Bayesian Calibration and Inference
26.5 Applications to Materials Models

### **Chapter 27: Model Validation and Verification**
27.1 Verification vs Validation
27.2 Validation Metrics and Criteria
27.3 Hierarchical Validation Approach
27.4 Extrapolation and Domain of Applicability
27.5 Documentation and Best Practices

### **Chapter 28: Future Directions**
28.1 Physics-Informed Machine Learning
   - 28.1.1 Integrating Physical Constraints
   - 28.1.2 Physics-Informed Neural Networks (PINNs)
28.2 Deep Learning for Materials Science
   - 28.2.1 Neural Networks for Complex Mappings
   - 28.2.2 Transfer Learning
28.3 Multi-Fidelity Modeling
28.4 Real-Time Process Monitoring and Control
28.5 Materials Informatics and High-Throughput Approaches
28.6 Digital Twins for Materials Processing

---

## **Part IX: Appendices**

### **Appendix A: Mathematical Tools**
A.1 Matrix Algebra
A.2 Calculus of Variations
A.3 Probability and Statistics Review
A.4 Numerical Integration Methods
A.5 Differential Equation Solvers

### **Appendix B: Steel Grades and Compositions**
B.1 Low Carbon Steels
B.2 Medium Carbon Steels
B.3 42CrMo4 Specification
B.4 Heat Treatment Protocols

### **Appendix C: Experimental Techniques**
C.1 Torsion Testing Procedures
C.2 Dilatometry
C.3 Hardness Testing
C.4 Tensile Testing
C.5 X-Ray Diffraction Experimental Setup

### **Appendix D: Software Implementation**
D.1 Programming Languages and Environments
D.2 Open-Source Libraries Used
D.3 Code Structure and Documentation
D.4 Reproducibility Guidelines

### **Appendix E: Dataset Descriptions**
E.1 Hot Working Dataset
E.2 Phase Transformation Dataset
E.3 Mechanical Properties Dataset
E.4 Data Availability Statement

---

## **Suggested Additional Topics/Concepts to Consider:**

1. **Hyperparameter Optimization Techniques**: Grid search, random search, Bayesian optimization (especially for ML models)

2. **Model Interpretability**: SHAP values, LIME, partial dependence plots for ML models

3. **Neural Networks**: Even if you haven't used them extensively, a brief section comparing traditional ML to deep learning could be valuable

4. **Time-Temperature Transformation**: More explicit treatment of cooling rate effects

5. **Finite Element Integration**: If your models connect to FEM simulations

6. **Texture and Anisotropy**: If relevant to your work

7. **Multi-Objective Optimization**: If you've dealt with competing objectives

8. **Transfer Learning**: Applying models trained on one steel grade to another

9. **Data Augmentation**: Synthetic data generation for sparse experimental regions

10. **Benchmarking Section**: Standard test problems to compare optimization algorithms

---

## **Summary**

This structure emphasizes your mathematical rigor while maintaining strong connections to the physical metallurgy. Each chapter can be written with mathematical depth while always connecting back to the materials science application. The organization allows you to:

- Establish both mathematical and materials science foundations early
- Present computational methods (preprocessing, optimization, ML) before applications
- Show three major application domains with consistent mathematical treatment
- Integrate concepts in advanced topics
- Provide comprehensive reference material in appendices

The dual perspective throughout the book will make it valuable to both materials scientists wanting to understand computational methods and mathematicians/computer scientists interested in materials applications.