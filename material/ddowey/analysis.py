#-----Read data from 2008
flt2008=pd.read_csv('2008.csv')
print("shape of dataset : ", flt2008.shape)
print("Features in dataset : ", flt2008.columns)
print("No of features in dataset : ", flt2008.columns.shape)
# shape of dataset : s(7009728, 29)
# Features in dataset :  Index(['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'], dtype='object')
# No of features in dataset : (29,)
#-----Excluding Cancelled flights as they do not have any delay attribute
flt2008 = flt2008[flt2008['Cancelled']==0]
flt2008.drop(['Cancelled', 'CancellationCode', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'], axis=1, inplace=True)

#-----Generating Departure and Arrival time in hrs
flt2008['DepHr'] = flt2008['CRSDepTime']//100
flt2008['ArrHr'] = flt2008['CRSArrTime']//100

#-----DATA EXPLORATION-----

#----- plt Departure delay distribution
plt.figure(figsize=(12, 6))
plt.hist(flt2008.DepDelay.dropna(),bins = 1000,normed=1, edgecolor='white', color=colorLib[0])
plt.xlim(-25,150)
plt.xlabel('Min')
plt.ylabel('Probability')
plt.title('2008 Departure Delay Distribution')
remove_border()
lt.savefig("1.png", transparent=True)

#----- plt arrival delay distribution
plt.figure(figsize=(12, 6))
plt.hist(flt2008.ArrDelay.dropna(),bins = 1000,normed=1, edgecolor='white', color=colorLib[2])
plt.xlim(-25,150)
plt.xlabel('Min')
plt.ylabel('Probability')
plt.title('2008 Arrival Delay Distribution')
remove_border()
lt.savefig("2.png", transparent=True)
plt.show()

#----- examine the correlation between departure delay by Month
plt.figure(figsize=(12, 6))
flt2008[['Month','DepDelay']].groupby('Month').mean().plot(kind='bar', color=colorLib[0])
plt.xticks(rotation=0)
plt.xlabel('Month of Year')
plt.ylabel('Departure Delay in Min')
plt.title('Average Departure Delay by Month in 2008')
remove_border()
plt.savefig("3.png", transparent=True)
#----- examine the correlation between Arrival delay by Month
flt2008[['Month','ArrDelay']].groupby('Month').mean().plot(kind='bar', color=colorLib[2])
plt.xticks(rotation=0)
plt.xlabel('Month of Year')
plt.ylabel('Arrival Delay in Min')
plt.title('Average Arrival Delay by Month in 2008')
remove_border()
plt.savefig("4.png", transparent=True)
plt.show()

#----- examine the correlation between Departure delay by time of day
plt.figure(figsize=(12,6))   
flt2008[['DepHr','DepDelay']].groupby('DepHr').mean().plot(kind='bar', color=colorLib[0])
plt.xticks(rotation=0)
plt.xlabel('Time of Day')
plt.ylabel('Departure Delay in Min')
plt.title('Average Departure Delay by Time of Day in 2008')
remove_border()
plt.savefig("5.png", transparent=True)
#----- examine the correlation between Arrival delay by time of day
flt2008[['ArrHr','ArrDelay']].groupby('ArrHr').mean().plot(kind='bar', color=colorLib[2])
plt.xticks(rotation=0)
plt.xlabel('Time of Day')
plt.ylabel('Arrival Delay in Min')
plt.title('Average Arrival Delay by Time of Day in 2008')
remove_border()
plt.savefig("6.png", transparent=True)
plt.show()

#----- Comparison of mean delays across 4 different airports 
#----- Chicago O'Hare (ORD), Boston Logan (BOS), San Francisco (SFO) and New York LaGuardia(LGA).

flt2008ORD = flt2008[(flt2008['Origin'] == 'ORD') | (flt2008['Dest']=='ORD')]
flt2008BOS = flt2008[(flt2008['Origin'] == 'BOS') | (flt2008['Dest']=='BOS')]
flt2008SFO = flt2008[(flt2008['Origin'] == 'SFO') | (flt2008['Dest']=='SFO')]
flt2008LGA = flt2008[(flt2008['Origin'] == 'LGA') | (flt2008['Dest']=='LGA')]

flt2008ORD.head().T

#                     94059  102746  102807  103000  103061
# Year                 2008    2008    2008    2008    2008
# Month                   1       1       1       1       1
# DayofMonth              8      25      10       4      22
# DayOfWeek               2       5       4       5       2
# DepTime              1711    1813    1601     658     725
# CRSDepTime           1600    1742    1600     650     725
# ArrTime              2031    2040    1928    1002     834
# CRSArrTime           1945    2004    1945     955     757
# UniqueCarrier          XE      XE      XE      XE      XE
# FlightNum            1226    2229    1226    1220    2408
# TailNum            N14953  N15509  N14974  N14974  N11119
# ActualElapsedTime     140      87     147     124     129
# CRSElapsedTime        165      82     165     125      92
# AirTime               106      49     101      97      90
# ArrDelay               46      36     -17       7      37
# DepDelay               71      31       1       8       0
# Origin                ORD     ORD     ORD     ORD     CLE
# Dest                  EWR     CLE     EWR     EWR     ORD
# Distance              719     316     719     719     316
# TaxiIn                  8       6      19      11       6
# TaxiOut                26      32      27      16      33
# Diverted                0       0       0       0       0
# DepHr                  16      17      16       6       7
# ArrHr                  19      20      19       9       7

#----- Average Departure Delay by Months for 4 Airports
Months = np.array(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
mon = np.arange(len(Months))

sns.set(style="white", font='StixGeneral', context="talk")
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=True, sharey=True)

ax1.set_title('Average Departure Delay by Month for 4 airports')

sns.barplot(mon, flt2008ORD[['Month','DepDelay']].groupby('Month').mean().reset_index().DepDelay, ci=None, palette="coolwarm", ax=ax1)
ax1.set_ylabel('Chicago'); ax1.set_xlabel(''); 
sns.barplot(mon, flt2008BOS[['Month','DepDelay']].groupby('Month').mean().reset_index().DepDelay, ci=None, palette="coolwarm", ax=ax2)
ax2.set_ylabel('Boston'); ax2.set_xlabel('')
sns.barplot(mon, flt2008SFO[['Month','DepDelay']].groupby('Month').mean().reset_index().DepDelay, ci=None, palette="coolwarm", ax=ax3)
ax3.set_ylabel('San Francisco'); ax3.set_xlabel('')
sns.barplot(mon, flt2008LGA[['Month','DepDelay']].groupby('Month').mean().reset_index().DepDelay, ci=None, palette="coolwarm", ax=ax4)
ax4.set_ylabel('New York'); ax4.set_xlabel('')
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[], xticklabels=Months)
plt.tight_layout(h_pad=0)
plt.savefig("7.png", transparent=True)
plt.show()

#----- Average Arrival Delay by Months for 4 Airports
sns.set(style="white", font='StixGeneral', context="talk")
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=True, sharey=True)

ax1.set_title('Average Arrival Delay by Month for 4 airports')

sns.barplot(mon, flt2008ORD[['Month','ArrDelay']].groupby('Month').mean().reset_index().ArrDelay, ci=None, palette="coolwarm", ax=ax1)
ax1.set_ylabel('Chicago'); ax1.set_xlabel(''); 
sns.barplot(mon, flt2008BOS[['Month','ArrDelay']].groupby('Month').mean().reset_index().ArrDelay, ci=None, palette="coolwarm", ax=ax2)
ax2.set_ylabel('Boston'); ax2.set_xlabel('')
sns.barplot(mon, flt2008SFO[['Month','ArrDelay']].groupby('Month').mean().reset_index().ArrDelay, ci=None, palette="coolwarm", ax=ax3)
ax3.set_ylabel('San Francisco'); ax3.set_xlabel('')
sns.barplot(mon, flt2008LGA[['Month','ArrDelay']].groupby('Month').mean().reset_index().ArrDelay, ci=None, palette="coolwarm", ax=ax4)
ax4.set_ylabel('New York'); ax4.set_xlabel('')
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[], xticklabels=Months)
plt.tight_layout(h_pad=0)
plt.savefig("8.png", transparent=True)
plt.show()

#----- Average Departure Delay by hour of day for 4 airports
ORD = flt2008ORD[['DepHr','DepDelay']].groupby('DepHr').mean().reset_index()
BOS = flt2008BOS[['DepHr','DepDelay']].groupby('DepHr').mean().reset_index()
SFO = flt2008SFO[['DepHr','DepDelay']].groupby('DepHr').mean().reset_index()
LGA = flt2008LGA[['DepHr','DepDelay']].groupby('DepHr').mean().reset_index()
sns.set(style="white", font='StixGeneral', context="talk")
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=False, sharey=True)

ax1.set_title('Average Departure Delay by hour of day for 4 airports')

sns.barplot(ORD.DepHr, ORD.DepDelay, ci=None, palette="coolwarm", ax=ax1)
ax1.set_ylabel('Chicago'); ax1.set_xlabel(' '); 
sns.barplot(BOS.DepHr, BOS.DepDelay, ci=None, palette="coolwarm", ax=ax2)
ax2.set_ylabel('Boston'); ax2.set_xlabel('')
sns.barplot(SFO.DepHr, SFO.DepDelay, ci=None, palette="coolwarm", ax=ax3)
ax3.set_ylabel('San Francisco'); ax3.set_xlabel('')
sns.barplot(LGA.DepHr, LGA.DepDelay, ci=None, palette="coolwarm", ax=ax4)
ax4.set_ylabel('New York'); ax4.set_xlabel('')
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])#, xticklabels=Hours)
plt.tight_layout(h_pad=0)
plt.savefig("9.png", transparent=True)
plt.show()

#----- Average Arrival Delay by hour of day for 4 airports
ORD = flt2008ORD[['ArrHr','ArrDelay']].groupby('ArrHr').mean().reset_index()
BOS = flt2008BOS[['ArrHr','ArrDelay']].groupby('ArrHr').mean().reset_index()
SFO = flt2008SFO[['ArrHr','ArrDelay']].groupby('ArrHr').mean().reset_index()
LGA = flt2008LGA[['ArrHr','ArrDelay']].groupby('ArrHr').mean().reset_index()
sns.set(style="white", font='StixGeneral', context="talk")
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=False, sharey=True)

ax1.set_title('Average Arrival Delay by hour of day for 4 airports')

sns.barplot(ORD.ArrHr, ORD.ArrDelay, ci=None, palette="coolwarm", ax=ax1)
ax1.set_ylabel('Chicago'); ax1.set_xlabel(''); 
sns.barplot(BOS.ArrHr, BOS.ArrDelay, ci=None, palette="coolwarm", ax=ax2)
ax2.set_ylabel('Boston'); ax2.set_xlabel('')
sns.barplot(SFO.ArrHr, SFO.ArrDelay, ci=None, palette="coolwarm", ax=ax3)
ax3.set_ylabel('San Francisco'); ax3.set_xlabel('')
sns.barplot(LGA.ArrHr, LGA.ArrDelay, ci=None, palette="coolwarm", ax=ax4)
ax4.set_ylabel('New York'); ax4.set_xlabel('')
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])#, xticklabels=Hours)
plt.tight_layout(h_pad=0)
plt.savefig("10.png", transparent=True)
plt.show()

#----- Average Departure Delay by Carrier in 2008, All  airports
flt2008[['UniqueCarrier','DepDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', color=colorLib[2])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Departure Delay by Carrier in 2008, All  airports')
remove_border()
plt.grid(True)
plt.savefig("11.png", transparent=True)

#----- Average Arrival Delay by Carrier in 2008, All  airports
flt2008[['UniqueCarrier','ArrDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', color=colorLib[0])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Arrival Delay by Carrier in 2008, All  airports')
remove_border()
plt.grid(True)
plt.savefig("12.png", transparent=True)
plt.show()

#-----Average Departure Delay by Carrier in 2008, Chicago
plt.figure(figsize =(12,8))
flt2008ORD[['UniqueCarrier','ArrDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', figsize =(12,8), color=colorLib[0])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Arrival Delay by Carrier in 2008 in Chicago')
remove_border()
plt.grid(True)
plt.savefig("13.png", transparent=True)

#-----Average Arrival Delay by Carrier in 2008, Chicago
flt2008ORD[['UniqueCarrier','DepDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', figsize =(12,8), color=colorLib[2])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Departure Delay by Carrier in 2008 in Chicago')
remove_border()
plt.grid(True)
plt.savefig("14.png", transparent=True)
plt.show()

#-----Average Departure Delay by Carrier in 2008, Boston
plt.figure(figsize =(12,8))
flt2008BOS[['UniqueCarrier','ArrDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', figsize =(12,8), color=colorLib[0])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Arrival Delay by Carrier in 2008 in Boston')
remove_border()
plt.grid(True)
plt.savefig("15.png", transparent=True)

#-----Average Arrival Delay by Carrier in 2008, Boston
flt2008BOS[['UniqueCarrier','DepDelay']].groupby('UniqueCarrier').mean().plot(kind='bar', figsize =(12,8), color=colorLib[2])
plt.xticks(rotation=0)
plt.xlabel('Carrier')
plt.ylabel('Average Delay in Min')
plt.title('Average Departure Delay by Carrier in 2008 in Boston')
remove_border()
plt.grid(True)
plt.savefig("16.png", transparent=True)
plt.show()