from random import randint
from time import strftime
from flask import Flask, render_template, flash, request, Markup
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.fields.html5 import DateField
import threading
from get_tweets import get_combined_tweets
from stockmarket import get_prediction_by_date



DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'


all_predictions = {}
prediction_threads = []

class Prediction():

    def __init__(self,prediction_id,form_data):
        self.company=form_data['company']
        self.code=form_data['code']
        self.main_handles=form_data['main_handles'].split(",")
        self.other_handles=form_data['other_handles'].split(",")
        self.keywords=form_data['keywords'].split(",")
        self.related_companies=form_data['related'].split(",") # Related Companies
        self.predict_date = form_data['date']
        self.prediction_id = form_data['prediction_id']
        self.predicted_value = None
        self.status = "Created"
    



class ReusableForm(Form):
    dt = DateField('DatePicker', format='%Y-%m-%d')
    name = TextField('Name:', validators=[validators.required()])
    surname = TextField('Surname:', validators=[validators.required()])

def get_time():
    time = strftime("%Y-%m-%dT%H:%M")
    return time

def write_to_disk(name, surname, email):
    data = open('file.log', 'a')
    timestamp = get_time()
    data.write('DateStamp={}, Name={}, Surname={}, Email={} \n'.format(timestamp, name, surname, email))
    data.close()

@app.route("/show_predicted_value/<prediction_id>", methods=['GET', 'POST'])
def show_predicted_value(prediction_id):
    global all_predictions
    # Function to fetch predicted value and return to user
    # 
    if prediction_id in all_predictions:
        predicted_value = all_predictions[prediction_id].predicted_value
        
        if predicted_value is None:
            status = all_predictions[prediction_id].status
            return "Prediction for id:{} in process,status:{} please give some more time".format(prediction_id,status)
        
        return all_predictions[prediction_id].predicted_value

    else:
        return "Prediction ID not found, add prediction again"


def predict_value(prediction_id):
    global all_predictions

    prediction = all_predictions[prediction_id]


    main_handles = prediction.main_handles
    other_handles = prediction.other_handles
    keywords = prediction.keywords
    end_date= prediction.predict_date
    stock_code = prediction.code
    pages = 20

    # combined_tweets =  get_combined_tweets(       main_handles,other_handles,keywords,end_date,pages)
    # print("number of tweets collected:{}".format(len(combined_tweets)))

    prediction.status = "Tweets done"
    prediction.predicted_value = str(get_prediction_by_date(end_date, stock_code))
    print(prediction.predicted_value)
    
    prediction.status = "Done"


    # Main function to get predicted value
    
    # TODO Gather tweeter data

    pass


def start_prediction_process(prediction_id):
    global all_predictions
    try:
        prediction_obj = all_predictions[prediction_id]
    except KeyError:
        flash_msg = "ID :{} not found in all_predictions :{}".format(prediction_id,list(all_predictions.keys())) 
        print(flash_msg)
        flash(flash_msg,category="error")
    except Exception as err:
        print(str(err))

    # start a thread to do prediction
    thread = threading.Thread(target=predict_value, args=(prediction_id,))
    thread.daemon = True
    thread.start()

    prediction_threads.append(thread)
    flash(Markup("prediction started for ID:{}, check result in  <a href='/show_predicted_value/{}' class='alert-link'>here</a>".format(prediction_id,prediction_id)))
    
    # for thrd in prediction_threads:
    #     thrd.join()

    print("print prediction thread started")



@app.route('/', methods=['GET'])
def landing():
    return """
    <title>HomePage</title>
    <h3> Welcome to UVengers Stock Prediction App </h3>
    <p> <a href ="/addCompany   ">click here</a> to add company </p>
    """


@app.route("/addCompany", methods=['GET', 'POST'])
def addCompany():
    global all_predictions

    form = ReusableForm(request.form)

    #print(form.errors)
    if request.method == 'POST':
        company=request.form['company']

        if company is None or company is "":
            flash("Please enter company ",'error')
            return render_template('index.html', form=form)

        code=request.form['code']
        if code is None or code is "":
            flash("Please enter code ",'error')
            return render_template('index.html', form=form)

        main_handles=request.form['main_handles']

        if main_handles is None or main_handles is "":
            flash("Please enter main_handles ",'error')
            return render_template('index.html', form=form)

        related_companies=request.form['related'] # Related Companies

        date = request.form['date']
        if date is None or date is "":
            flash("Please enter date ",'error')
            return render_template('index.html', form=form)




        prediction_id = request.form['prediction_id']
        prediction = Prediction(prediction_id,request.form)

        if prediction_id is None or prediction_id is "":
            flash("Please enter prediction ID ",'error')
            return render_template('index.html', form=form)
        
        if prediction_id in all_predictions:
            flash("prediction ID not unique",'error')
            return render_template('index.html', form=form)
        

        all_predictions[prediction_id] = prediction
        print(list(all_predictions.keys()))
        start_prediction_process(prediction_id)

        # print(request.form)


        # if form.validate():
        #     write_to_disk(name, surname, email)
        #     flash('Hello: {} {}'.format(name, surname))

        # else:
        #     flash('Error: All Fields are Required')

    return render_template('index.html', form=form)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
