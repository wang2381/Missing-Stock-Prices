
##################################################################################

#-------------------Missing Stock Prediction API----------------------------------
#                                 ------by Sirui Wang 11/21/2018

#This is the API for stock data input and predicted data output
# 1. Open Anaconda Prompt (or command window)
# 2. Go to the .py directory
# 3. Type command "Python Missing_Stock.py"
# 4. Type the correct location of the input stock data (e.g., C:/case/input/input03.txt)
# 5. Select the method for inner data interpolation (Type "1" for regularized cubic spline; Type "2" for cubic spline)
#    Note: regularized cubic spline may be time consuming...
# 6. Press Enter, and wait for the predicted stock data output on the Prompt
# 7. Quit

##################################################################################

import numpy as np
from scipy import interpolate
import fileinput
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def Extract_values_new(Location):

    i=-1
    prices = []
    training_feature = []
    training_label = []
    testing_feature = []
    testing_label = []
    total_label = []
    with open(Location,'r') as file:
        for line in file:
            if i==-1:
                n=int(line)
                i=i+1
            else:
                prices.append(line.rstrip().split("\t")[1])

    for i in range(n):
        if not "Missing" in prices[i]:
            training_feature.append(i)
            training_label.append(float(prices[i]))

        else:
            testing_feature.append(i)

    testing_for_regression = []


    for i in range(len(testing_feature)-1,-1,-1):
        if testing_feature[i]==n-1:
            testing_for_regression.append(n-1)

            n-=1
            del testing_feature[i]

        else:
            break

    k=0
    for i in range(len(testing_feature)):
        if testing_feature[i]==k:
            testing_for_regression.append(k)
            k+=1
            del testing_feature[i]

        else:
            break

    print("Data read!")
    print()
    return training_feature, training_label, testing_feature, testing_for_regression

#************************************************************************************************

Location=input('Please enter the path of the input file\n\n')
training_feature, training_label, testing_feature, testing_for_regression = Extract_values_new(Location)

#************************************************************************************************

#This is the function using python interpolate.interp1d (Optional)
def Evaluate_spline_no_extra_interp1d(training_feature, training_label, testing_feature):

    model = interpolate.interp1d(training_feature, training_label, 'cubic',fill_value = 'extrapolate')
    predicting_label = []

    print('Begin interpolation...')

    for each_feature in testing_feature:

        predicting_label.append(model(each_feature))


    dic_inter = {}
    for i in range(len(testing_feature)):
        dic_inter[testing_feature[i]] = predicting_label[i]

    print('Interpolation complete!')
    print()

    return dic_inter

#This is the function using python interpolate.UnivariateSpline with regularization and cross-validation (Optional, time-consuming)
def Evaluate_spline_no_extra_regularized(training_feature, training_label, testing_feature):

    d_opt = 99999
    s_opt = 99999
    predicting_label = []

    for S in np.arange(0.5,6,0.02):

        d_total = 0
        d_avg = 0

        #for i in range(len(training_feature)):
        for i in range(7):
            leave_one_out_feature = []
            leave_one_out_label = []
            training_feature_temp = []
            training_label_temp = []

            np.random.seed(i)
            leave_one_out_feature = np.random.choice(training_feature,len(training_feature)//2,replace=False)
            np.random.seed(i)
            leave_one_out_label = np.random.choice(training_label,len(training_label)//2,replace=False)

            #leave_one_out_feature.append(training_feature[i])
            #leave_one_out_label.append(training_label[i])

            for j in range(len(training_feature)):
                if training_feature[j] not in leave_one_out_feature:
                    training_feature_temp.append(training_feature[j])
                    training_label_temp.append(training_label[j])

            '''
            for j in range(len(training_feature)):

                training_feature_temp.append(training_feature[j])
                training_label_temp.append(training_label[j])

            del training_feature_temp[i]
            del training_label_temp[i]
            '''
            model = interpolate.UnivariateSpline(training_feature_temp, training_label_temp, s = S, k=3)

            '''
            predicting_leave_one_out_label = model(leave_one_out_feature)

            d=(abs(leave_one_out_label-predicting_leave_one_out_label) / leave_one_out_label)*100
            d_total = d_total + d
            '''
            k = 0
            for each_feature in leave_one_out_feature:
                predicting_leave_one_out_label = model(each_feature)

                d = 0
                d=(abs(leave_one_out_label[k]-predicting_leave_one_out_label) / leave_one_out_label[k])*100
                k+=1
                d_total += d



        if d_opt > d_total:
            d_opt = d_total
            s_opt = S


    d_opt = d_opt / (len(training_feature)//2) / 7

    print('The best cross-validation score is: {0}'.format(d_opt))
    print('The best parameter is: {0}'.format(s_opt))
    print('Begin interpolation...')

    model = interpolate.UnivariateSpline(training_feature, training_label, s = s_opt, k=3)

    for each_feature in testing_feature:

        predicting_label.append(model(each_feature))



    dic_inter = {}
    for i in range(len(testing_feature)):
        dic_inter[testing_feature[i]] = predicting_label[i]

    print('Interpolation complete!')
    print()

    return dic_inter

#************************************************************************************************

option = input('Please select the method for inner data prediction \n\n Type "1" for regularized cubic spline; "2" for cubic spline.' )

if option == "1":
    dic_inter = Evaluate_spline_no_extra_regularized(training_feature, training_label, testing_feature)
elif option == "2":
    dic_inter = Evaluate_spline_no_extra_interp1d(training_feature, training_label, testing_feature)
else:
    print("Error!")


#************************************************************************************************
def Evaluate_regression_extra(training_feature, training_label, testing_for_regression):

    print("Begin extrapolation...")
    print()

    d = -9999
    min_mse = 9999
    deg_opt = -2
    predicting_label = []
    dic_extra = {}

    if len(testing_for_regression) == 0:
        return dic_extra


    x_train, x_test, y_train, y_test = train_test_split(training_feature, training_label, test_size=0.2, random_state=3)

    for deg in [7,8,9,10,11,12]:


        model = np.poly1d(np.polyfit(x_train, y_train, deg))

        y_predict = model(x_test)
        mse = mean_squared_error(y_predict, y_test)
        if min_mse > mse:
            min_mse = mse
            deg_opt = deg

    print('The minimum validation MSE is: {0}'.format(min_mse))
    print('The best degree is: {0}'.format(deg_opt))


    model = np.poly1d(np.polyfit(training_feature, training_label, deg))
    for each_feature in testing_feature:


        predicting_label.append(model(each_feature))


    for i in range(len(testing_for_regression)):
        dic_extra[testing_for_regression[i]] = predicting_label[i]

    print('Extrapolation complete!')
    print()




    return dic_extra


#************************************************************************************************

dic_extra = Evaluate_regression_extra(training_feature, training_label, testing_for_regression)

#************************************************************************************************



print("Combining interpolation and extrapolation...")
print()
dic_all = {}
for k,v in dic_inter.items():
    dic_all[k] = v

if len(dic_extra.keys())!=0:
    for k,v in dic_extra.items():
        dic_all[k] = v

for k,v in dict(sorted(dic_all.items())).items():
    print(v)
print()
print("All complete!")
