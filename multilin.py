import numpy as np 
import pandas as pd
import warnings 
from matplotlib import pyplot as plt
import seaborn as sns
import time 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

df = pd.read_csv('avocado.csv')
print df.head()

class OrdinaryLeastSquares(object):

	def __init__(self):
		self.coefficients = [] 

	def fit(self, X, y):
		if len(X.shape) == 1: X = self._reshape_x(X)
		
		X = self._concatenate_ones(X)
		self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
	
	def predict(self, entry):
		b0 = self.coefficients[0]
		other_betas = self.coefficients[1:]
		prediction = b0
	
		for xi, bi in zip(entry, other_betas): prediction += (bi*xi)
		return prediction
	
	#underscore here to make the method itself private.

	def _reshape_x(self, X):
		return X.reshape(-1,1)
	def _concatenate_ones(self,X):
		ones = np.ones(shape = X.shape[0]).reshape(-1,1)
		return np.concatenate((ones,X),1)
 
def regionn(n): 
	regions = {'Albany':1, 'Atlanta' : 2, 'BaltimoreWashington':3, 'Boise':4, 'Boston':5,'BuffaloRochester':6, 'California':7, 'Charlotte':8, 'Chicago':9, 'CincinnatiDayton':10,'Columbus':11,'DallasFtWorth':12, 'Denver':13, 'Detroit':14, 'GrandRapids':15, 'GreatLakes':16,'HarrisburgScranton':17, 'HartfordSpringfield':18, 'Houston':19, 'Indianapolis':20,'Jacksonville':21, 'LasVegas':22, 'LosAngeles':23, 'Louisville':24, 'MiamiFtLauderdale':25,'Midsouth':26, 'Nashville':27, 'NewOrleansMobile':28, 'NewYork':29 ,'Northeast':30,'NorthernNewEngland':31, 'Orlando':32, 'Philadelphia':33, 'PhoenixTucson':34,'Pittsburgh':35, 'Plains':36, 'Portland':37, 'RaleighGreensboro':38, 'RichmondNorfolk':39,'Roanoke':40, 'Sacramento':41, 'SanDiego':42, 'SanFrancisco':43, 'Seattle':44,'SouthCarolina':45, 'SouthCentral':46 ,'Southeast':47, 'Spokane':48, 'StLouis':49, 'Syracuse':50, 'Tampa':51, 'TotalUS':52, 'West':53, 'WestTexNewMexico':54 }
	return regions[n] 

def total_revenue(x, y):
	return x * y


#Read in file 
df = pd.read_csv('avocado.csv')
df.rename(columns={'Total Volume':'TV'}, inplace=True)

#Convert date object to Panadas readable datetime.
df['Date'] = pd.to_datetime(df['Date'])

#Remove unneeded columns etc.
#columns = ['4046', '4225', '4770']
#df.drop(columns, inplace=True, axis=1)
#add in month column
df['month'] = pd.DatetimeIndex(df['Date']).month
#assign a region number to a region. 
 

#df['region_number']  = df.apply(lambda row: regionn(row.region), axis = 1)
df['total revenue']  = df.apply(lambda row: total_revenue(row.TV, row.AveragePrice), axis = 1)
#get the season.
 

#df['season'] = df.apply(lambda dt: (pd.DatetimeIndex.month%12 + 3)//3)

 

#split data set into both conventional and organic 
organic = df[df["type"] == "organic"]
test2 = df[df["type"] == "conventional"]
test1 = test2[test2["region"] == "SanDiego"]
test1 = test1.groupby([df['Date'].dt.date]).mean() 
 
#normalize the dataset TEST 

#test1 = (test1 - test1.mean())/test1.std()


print pd.DataFrame(test1)
#test for one particular region 
#test1 = conventional[conventional["year"] == 2017]

X = test1[[ 'month','year','XLarge Bags', 'Large Bags',  'Total Bags', 'TV']].values
y = test1['AveragePrice'].values

model = OrdinaryLeastSquares()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#X are features here, y is the target column, namely AveragePrice.

#Training is split approx. 70/30, as directed from the module outlines. 
model.fit(X_train,y_train)
print "------------------------------------------------"
print "------------------------------------------------"

print "Training model at a 70% 30% split"

print "------------------------------------------------"
print "------------------------------------------------"
time.sleep(5)
print "------------------------------------------------"
print "------------------------------------------------"

print"Model trained. Predicting Values, please wait..."

time.sleep(5)

# Option to print coefficients
# print(model.coefficients)

y_preds = [] 
for row in X_test: y_preds.append(model.predict(row))
predictions = pd.DataFrame({'Actual' : y_test, 'Predicted' : np.ravel(y_preds)})
print predictions
print "------------------------------------------------"
print "------------------------------------------------"

predictions.plot(kind='line',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


print "------------------------------------------------"
print "------------------------------------------------"
print " Now attempting to check the error of the model using RMSE"
#RMSE Error checkiing 

def rmse(predict, actual):
	return np.sqrt(((predict - actual) ** 2).mean())


#convert the first column of our predictions to an array, followed by the second column of our array.

act = np.array(predictions['Actual'])
pred = np.array(predictions['Predicted'])
 
rmse_val = rmse(pred, act)
r = r2_score(pred, act)

print "The RMS error of this is " + str(rmse_val)
print "The R2 score of this model is " + str(r)



 
