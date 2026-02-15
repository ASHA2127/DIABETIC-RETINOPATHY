import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request, redirect
import sqlite3
import cv2
import shutil
import re




connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()
command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT, approved INTEGER DEFAULT 0)"""
cursor.execute(command)
connection.commit()

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute("SELECT name, email, mobile, approved FROM user WHERE approved != -1")
    pending_users = cursor.fetchall()
    return render_template('admin_dashboard.html', pending_users=pending_users)

@app.route('/approve_user/<string:username>', methods=['POST'])
def approve_user(username):
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute("UPDATE user SET approved = 1 WHERE name = ?", (username,))
    connection.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/reject_user/<string:username>', methods=['POST'])
def reject_user(username):
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    cursor.execute("DELETE FROM user WHERE name = ?", (username,))
    connection.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        name = request.form['name']
        password = request.form['password']
        query = "SELECT name, password, approved FROM user WHERE name = ? AND password = ? AND approved = 1"
        cursor.execute(query, (name, password))
        result = cursor.fetchall()
        if len(result) == 0:
            return render_template('user_register.html', msg='Sorry, Incorrect Credentials or Not Approved Yet, Try Again')
        else:
            return render_template('userlog.html')
    return render_template('user_register.html')

@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT, approved INTEGER DEFAULT 0)"""
        cursor.execute(command)
        connection.commit()
        # Remove the duplicate command execution
        # cursor.execute(command)
        # cursor.execute(command)
        # connection.commit()
        
        if not name or not password or not mobile or not email:
            return render_template('user_register.html', msg='All fields are required')
        if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return render_template('user_register.html', msg='Invalid email format')
        cursor.execute("INSERT INTO user (name, password, mobile, email) VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()
        return redirect(url_for('userlog'))

    return render_template('user_register.html')


@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')

@app.route('/results', methods=['GET', 'POST'])
def image():
    if request.method == 'GET':
        return render_template('results.html')
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)


        
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'DiabeticRetinopathy-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 5, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        Tre=""
        Tre1=""
        diet=""
        diet1=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

           
                
                
            if np.argmax(model_out) == 0:
                str_label = "Mild"
                print("The predicted image of the Mild is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Mild is with a accuracy of {}%".format(model_out[0]*100)

                diet=" The Diet chart for Mild"
                diet1=["Control carbs: Avoid white bread, rice, and sugar-heavy snacks.",
                "Add lutein & zeaxanthin: Kale, spinach, broccoli – boost retinal health.",
                "Stay hydrated: Drink at least 2–2.5L of water daily.",
                "Snack smart: Handful of almonds or boiled eggs instead of sugary snacks."]

                Tre="Treatment for Mild diabetic"
                Tre1=["Continue tight glycemic control: Prevent further damage to retinal blood vessels.",
                "Monitor progression: Eye exams every 6–12 months or as advised.",
                "Control comorbidities: Manage hypertension, kidney function, and lipids.",
                "No active treatment yet, but early intervention strategies discussed with the ophthalmologist."]
                
                

            elif np.argmax(model_out) == 1:
                str_label = "Moderate"
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[1]*100)
                diet = "The diet for  Moderate are:\n\n "
                diet1 = ["Limit sodium: Reduce processed foods, aim for <1500mg/day.",
                "Vitamin C & E sources: Bell peppers, sunflower seeds, and oranges.",
                "Low-fat dairy or plant-based calcium options to support vascular integrity.",
                "Avoid trans fats: Skip fried/packaged foods that worsen circulation."]

                Tre="Treatment for Moderate:"
                Tre1=["Closer monitoring: Eye exams every 3–6 months.",
                "Possible use of medications: Like ACE inhibitors or ARBs to protect retina (if hypertensive).",
                "Lifestyle reinforcement: Continue optimal control of blood glucose, BP, and lipids.",
                "Early referral to a retina specialist: For baseline imaging and long-term care plan."]

            elif np.argmax(model_out) == 2:
                str_label = "Proliferate_DR"
                print("The predicted image of the normal is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the normal is with a accuracy of {}%".format(model_out[2]*100)
                diet = "The diet chat for  Proliferate_DR are:\n\n "
                diet1 = [" Intensify antioxidant intake: Add supplements if advised (A, C, E, zinc).",
                "Focus on anti-inflammatory foods: Turmeric, ginger, flaxseeds.",
                "No added sugar: Even fruit juices should be avoided; stick to whole fruits.",
                "Consult dietitian regularly: As food tolerances may change with treatment."]

                Tre="Treatment for proliferate_DR:"
                Tre1=["More frequent monitoring: Every 2–4 months.",
                "OCT and fluorescein angiography: Imaging to assess retinal damage.",
                "Possible anti-VEGF injections: To reduce macular edema if present.",
                "Laser photocoagulation: May be considered to treat leaking vessels."]


            elif np.argmax(model_out) == 3:
                str_label = "Severe"
                print("The predicted image of the Moderate is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Moderate is with a accuracy of {}%".format(model_out[3]*100)
                rem = "The remedies for  Severe are:\n\n "
                rem1 = ["Strict glycemic control: Smaller, frequent balanced meals (protein + fiber).",
                "Boost zinc: Pumpkin seeds, lentils, whole grains – supports retina.",
                "Green tea: Rich in antioxidants; 1–2 cups/day can help reduce inflammation.",
                "Avoid high-cholesterol foods: Like red meats and full-fat dairy."]

                Tre="Treatment for severe:"
                Tre1=["Panretinal photocoagulation (PRP): Laser treatment to shrink abnormal blood vessels.",
                "Anti-VEGF injections: Regular injections (e.g., Avastin, Lucentis) to control vessel growth.",
                "Vitrectomy: Surgery if there's vitreous hemorrhage or retinal detachment.",
                "Very close follow-up: Every few weeks to monitor response and adjust treatment"]
                                
                           

            elif np.argmax(model_out) == 4:
                str_label  = "normal"
                print("The predicted image of the normal is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the normal is with a accuracy of {}%".format(model_out[4]*100)
                   
             
            
    

               
           
       
        return render_template('results.html', status=str_label,accuracy=accuracy,diet=diet,diet1=diet1,Treatment=Tre,Treatment1=Tre1,
                               ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,
                               ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                               ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                               ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                               ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_name = request.form['adminName']
        password = request.form['password']
        if admin_name == 'admin' and password == 'admin1234':
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Admin name or password is incorrect')
    return render_template('admin_login.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
