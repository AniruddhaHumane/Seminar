# ----- Identify the columns of the database
# ----- Let us use ORD as an example first and expand to other airports
print("Features in dataset : ", flt2008.columns)
print("No of features in dataset : ", flt2008.columns.shape)
#Features in dataset : Index(['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Diverted', 'DepHr', 'ArrHr'], dtype='object')
#No of features in dataset : (24,)

#----- Building Features set for analysis
#----- Seperatiing classes based on threashold
delay_threshold = 15

#----- Selecting direction
direction = "Origin"
#direction = "Dest"

#----- Selecting features set
Ycol = "DepDelay"
if direction == "Origin":
	# select features for Departure Delays
	Xcols = ["Month", "DayOfWeek", "DepHr", "ArrHr", "UniqueCarrier", "Dest", "CRSElapsedTime", "Distance"] 
else:
    # select Features for Arrival Delays
    Xcols = ['Month', 'DayOfWeek', 'DepHr', 'ArrHr', 'UniqueCarrier', 'Origin', 'CRSElapsedTime', 'Distance']

#----- Extracting important ORD data from original dataset
X_values = flt2008[flt2008[direction] == 'ORD'][Xcols]
Y_values = flt2008[flt2008[direction] == 'ORD'][Ycol]

# Factorise the qualitative variables (Giving unique index to Features and using them instead of actual strings)
X_values['UniqueCarrier'] = pd.factorize(X_values['UniqueCarrier'])[0]

if direction == 'Origin':
    X_values['Dest'] = pd.factorize(X_values['Dest'])[0]
else:
    X_values['Origin'] = pd.factorize(X_values['Origin'])[0]

#----- Selecting random 20k samples and segregating them
rows = np.random.choice(X_values.index.values, 20000)
sampled_X = X_values.ix[rows]
sampled_Y = Y_values.ix[rows]

TrainX, TestX, TrainY, TestY = train_test_split(
    sampled_X, sampled_Y, test_size=0.50, random_state=42)
