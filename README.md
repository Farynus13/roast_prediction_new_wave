# Coffee Roasting Curve Prediction Model

This project aims to develop a neural network model for inline time-series prediction of the coffee roasting curve, laying the foundation for an automated coffee roasting tool. The goal is to move beyond the conventional PID control, which only minimizes the difference from a set value, toward a system that simulates the decision-making process of a skilled human operator.

### Project Overview
By predicting the coffee roasting curve the model enables more precise control of burner output, targeting an optimal roast profile. Traditional methods for coffee roasting system modelling<sup>[[1]](https://doi.org/10.1016/j.rineng.2024.102575)</sup> fail to provide satisfactory results due to non-stationarity, non-linearity, and partial observability of the system dynamics (different coffees respond differently to the burner output, also temperature readings are only an approximation of the real state of the system). By utilizing data-driven approach, where we use historical data of coffee roasting to train the RNN model we can achieve higher accuracy in predicting the systems behaviour, thus allowing us to produce higher quality roast.


![Figure 1: Coffee Roasting Curve Prediction](/media/roast_prediction.gif)

*In black dashed lines we can observe the model's output given as an input last n seconds of the temperature curves along the burner changes.
We can see an almost perfect fit to the real roast curve's ET and BT. Thanks to the added Attention Mechanisms model focuses strongly on the changes made in the burner value, which is especially visible in the ET curve*

### Model Description
Current version of model utilizes LSTM <sup>[[2]](https://doi.org/10.1162/neco.1997.9.8.1735)</sup> network with Masked Attention Mechanism<sup>[[3]](https://doi.org/10.48550/arXiv.1706.03762)</sup> running in the autoregressive mode. As an input model takes timeseries data consisting of two temperature curves (Bean Temperature - BT, and Environment Temperature - ET) along with timeseries of burner value accross the roasting process. During Inference Time, burner value is used as exogenous input which we can modulate in order to simulate system dynamics (response of temperature curves on the burner settings)


### Objectives
Accurate Prediction of Roasting Curve

Generate a reliable prediction of the coffee roasting curve to anticipate the roast progression.

### Installation
Install dependencies:

```sh
pip install -U -r requirements.txt
```

### Training
1. First prepare dataset consisting of .alog roast profiles (Artisan file with recorded roast).
2. Gather all your roasts in the "data" folder
3. To preproces your dataset run:
```sh
python data_preparation.py
```
It will create data.npy and meta_data.npy files containing our preprocessed dataset

4. To train run:
```sh
python training.py
  -- model_path path_to_your_model
  -- start_epoch epoch
```
  - model_path: where to save your model
  - start_epoch: if passed > 0 it will load saved model from your path and restart training from the provided epoch,
  training pipeline uses scheduled sampling<sup>[[4]](https://doi.org/10.48550/arXiv.1506.03099)</sup> which is dependent on the epoch
### Testing
1. To test run:
```sh
python test.py
  -- model_path path_to_your_model
  -- plot [True/False]
```
  - model_path: path to the trained model
  - plot: whether to show prediction plots during execution, if False, prediction plots will only be saved to the 'figs' folder
Running this script will run evaluation, generate prediction plots and output calculated loss for the model under evaluation 



### Future Steps
Burner Control Model: Implement a model that utilizes the curve prediction to adjust burner output and achieve specified roast goals.

Refinement of Prediction Model: Fine-tune the model's predictive capabilities with additional roasting data for improved accuracy and adaptability to different machines.

---
### References

[[1]H.G. Schwartzberg, Modeling bean heating during batch roasting of coffee beans, J. Welti-Chanes, G. V Barbosa-Canovas, J.M. Aguilera (Eds.), Engineering and Food for the 21st Century (first ed.), CRC Press, Boca Raton (2002), p. 1104](https://doi.org/10.1016/j.rineng.2024.102575)

[[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.](https://doi.org/10.1162/neco.1997.9.8.1735)

[[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 5998-6008.](https://doi.org/10.48550/arXiv.1706.03762)

[[4] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://doi.org/10.48550/arXiv.1506.03099)
