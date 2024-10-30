# Coffee Roasting Curve Prediction Model

This project aims to develop a neural network model for inline time-series prediction of the coffee roasting curve, laying the foundation for an automated coffee roasting tool. The goal is to move beyond the conventional PID control, which only minimizes the difference from a set value, toward a system that simulates the decision-making process of a skilled human operator.

### Project Overview
By predicting the coffee roasting curve the model enables more precise control of burner output, targeting an optimal roast profile. Traditional methods for system modelling fail to provide satisfactory results due to non-stationarity, non-linearity, and partial observability of the system dynamics (different coffees respond differently to the burner output, also temperature readings are only an approximation of the real state of the system). By utilizing data-driven approach, where we use historical data of coffee roasting to train the RNN model we can achieve higher accuracy in predicting the systems behaviour, thus allowing us to produce higher quality roast.


![Figure 1: Coffee Roasting Curve Prediction](fig_19.png)

### Model Description
Current version of model utilizes LSTM network with Masked Attention Mechanism running in the autoregressive mode. As an input model takes timeseries data consisting of two temperature curves (Bean Temperature - BT, and Environment Temperature - ET) along with timeseries of burner value accross the roasting process. During Inference Time, burner value is used as exogenous input which we can modulate in order to simulate system dynamics (response of temperature curves on the burner settings)


### Objectives
Accurate Prediction of Roasting Curve
Generate a reliable prediction of the coffee roasting curve to anticipate the roast progression.

### Future Steps
Burner Control Model: Implement a model that utilizes the curve prediction to adjust burner output and achieve specified roast goals.
Refinement of Prediction Model: Fine-tune the model's predictive capabilities with additional roasting data for improved accuracy and adaptability to different machines.