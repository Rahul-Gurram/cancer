import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/cancerLog.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        radius_mean = flask.request.form['radius_mean']
        radius_se = flask.request.form['radius_se']
        compactness_se = flask.request.form['compactness_se']
        texture_worst = flask.request.form['texture_worst']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[radius_mean, radius_se, compactness_se, texture_worst]],
                                       columns=['radius_mean', 'bmi', 'compactness_se', 'texture_worst'],
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Radius_Mean':radius_mean,
                                                     'Radius_Se':radius_se,
                                                     'Compactness_Se':compactness_se,
                                                     'Texture_Worst':texture_worst},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)