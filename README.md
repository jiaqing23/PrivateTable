# Private Table Library

This library is to help library users to implement secure ways to handle data without knowing much knowledge of differential privacy. The functionalities supported by the library include differential privacy statistical function query, privacy tracker to calculate privacy budget consumed, differential privacy machine learning optimizer, and federated learning algorithm.

## Fuctionalities
### Private Mechanism
- Laplace Mechanism
- Gaussian Mechanism
- Histogram Mechanism
- Exponential Mechanism

### Privacy Budget Tracker
- `SimplePrivacyBudgetTracker`: Uses simple composition theorem
- `AdvancedPrivacyBudgetTracker`: Uses advance composition theorem
- `MomentPrivacyBudgetTracker`: Uses moment accountant

### Statistical Function
- mean
- standard deviation
- variance
- min
- max
- median
- mode
- categorical histogram
- numerical histogram

### Machine Learning Optimizer
- `PrivateSGD`

### Federated Learning
> Currently this part only support Tensorflow.
- Federated Averaging (FedAvg)
- DP-FedAvg