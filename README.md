# NEW PROJECT SETTINGS

### LOCAL ENVIRONMENT
you can use jupyter notebook or if you are use the py file you need to create a virtual environment.
`Python 3.10.4` is used at the time of testing. just make sure you have `Python 3.5+` installed on your machine just to avoid any library compatibility issue.

To handle multiple Python version you can use [Pyenv](https://realpython.com/intro-to-pyenv/)

### Steps to create virtual env:

1- create a new virtual environment in current directory.

    python -m venv env

2- then you need to active it using following command (linux or MacOs users)

    source env/bin/activate

for windows user
    
    .\env\Scripts\activate

3- now you need to install all the dependent libraries used in the code.
    
    pip install -r requirements.txt

4- once everything is install and ready, now you can use and test the code.

    python crypto_price_prediction_streamlit.py



------------------------------

# OLD
####
install anaconda (https://www.anaconda.com/products/distribution)
####
pip install matplotlib   
####
pip install plotly
####
 pip install python-binance
####
 pip install -U scikit-learn 
####
 conda install -c conda-forge keras
####

To start the app crypto_price_prediction_streamlit.py, you will need to install streamtlit using 
pip install streamlit

And start the app with 

streamlit run crypto_price_prediction_streamlit.py

###
The chart of the output of the prediction model is displayed using plotly. To see the chart in jupyter notebook, you will need to run the following command on your terminal and restart jupyter notebook

jupyter labextension install jupyterlab-plotly
