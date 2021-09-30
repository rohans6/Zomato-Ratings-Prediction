# Zomato-Ratings-Prediction
# Project Overview:-
* Build a Web Application to predict Zomato Ratings for your Restaurant in Banglaore.
* Bangalore being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world.  It  has become difficult for new restaurants to compete with established restaurants.
* Analysed Features like which are responsible to predict Ratings like:-Location,Price,Cuisine etc and prepared a model that can predict ratings for restaurants given features.
* Build an API using flask
# Features used for Prediction:-
* Online Order:- whether online ordering is available in the restaurant or not
* Book Table:-   table book option available or not
* Votes:-        contains total number of rating for the restaurant as of the today's date
* Location:-     contains the neighborhood in which the restaurant is located
* Rest Type:-    Type of Restaurant
# EDA:-
Looked at different plots to better understand the data. Some of plots from analysis:-
![Onine Order and Book Table](ob.png)
![Histograms of Ratings and Average Cost of 2 People](h.png)
![Top 10 Cuisines](c.png)
# Eval Metric and Modelling
R2_Score was Evaluation Metric
ExtraTressRegressor(n_estimators=1450) performed best with 0.94 R2 Score.
# Deployment
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl.
2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.
3. Navigate to URL http://localhost:5000
You will be able to see home-page.
Enter valid entries and click on predict button to get rating for your restaurant
