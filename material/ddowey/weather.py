# Import weather data
# First half of 2008
wx2008_1 = pd.read_csv('Weather1.csv', low_memory=False, skipinitialspace= True)
# Second half of 2008
wx2008_2 = pd.read_csv('Weather2.csv', low_memory=False, skipinitialspace= True)
# combine data for 2008
wx2008 = wx2008_1.append(wx2008_2)
# delete 2008 files to free memory space
del wx2008_1
del wx2008_2

# create Year and Month columns
wx2008['Year'] = wx2008['YYYYMM']//100
wx2008['Month'] = (wx2008['YYYYMM'] - wx2008['Year']*100)
# examine wx2008 columns
wx2008.columns
wx2008.columns.shape[0]
# Index(['LOCID', 'YYYYMM', 'DAYNUM', 'HR_LOCAL', 'DEP_CT', 'ARR_CT', 'T_O_GATE_DELAY', 'T_O_GATE_DELAY_A', 'T_GATE_DELAY', 'T_GATE_DELAY_A', 'T_OAG_ARPT_DEP', 'T_OAG_ARPT_DEP_A', 'T_PTM_ARPT_DEP', 'T_PTM_ARPT_DEP_A', 'T_O_DELAY_ARR', 'T_O_DELAY_ARR_A', 'T_DELAY_ARR', 'T_DELAY_ARR_A', 'T_DIF_G2G', 'T_DIF_G2G_A', 'O_GATE_DEL_15C', 'O_GATE_DEL_15M', 'O_GATE_DEL_15_A', 'OAG_ARPT_DEP_15C', 'OAG_ARPT_DEP_15M', 'OAG_ARPT_DEP_15_A', 'O_DELAY_ARR_15C', 'O_DELAY_ARR_15M', 'O_DELAY_ARR_15M_A',
#        'O_GATE_DEL_15', 'O_GATE_DEL_15_PCNT', 'DEL_DEP15', 'DEL_DEP15_PCNT', 'OAG_ARPT_DEP_15', 'OAG_ARPT_DEP_15_PCNT', 'PTM_ARPT_DEP_15', 'PTM_ARPT_DEP_15_PCNT', 'O_DELAY_ARR_15', 'O_DELAY_ARR_15_PCNT', 'DEL_ARR15', 'DEL_ARR15_PCNT', 'T_DEP_EDCT_HOLD', 'T_ARR_EDCT_HOLD', 'DEP_EDCT_CNT', 'ARR_EDCT_CNT', 'EDCT_DEP_EARLY', 'EDCT_DEP_LATE', 'EDCT_ARR_EARLY', 'EDCT_ARR_LATE', 'T_TAXI_OUT', 'DEL_TAXI_OUT_CNT', 'T_DELAY_TO', 'T_DELAY_TO_A', 'DELAIR_CT_0', 'T_DELAY_AIR', 'T_DELAY_AIR_A', 'T_TAXI_IN',
#        'DEL_TAXI_IN_CNT', 'T_DELAY_TI', 'T_DELAY_TI_A', 'DEP_CNT', 'ARR_CNT', 'OAG_DEP', 'OAG_ARR', 'ETMS_DEP', 'ETMS_ARR', 'FRO_DEP', 'FRO_ARR', 'MC', 'CEILING', 'VISIBLE', 'TEMP', 'WND_ANGL', 'WND_SPED', 'RUNWAY', 'DEP_DEMAND', 'ARR_DEMAND', 'DEP_RATE', 'ARR_RATE', 'DEP_SCORE', 'ARR_SCORE', 'TOT_UTIL', 'Year', 'Month'],
#       dtype='object')
# 84

# Keep weather variables of our greatest interest
wxcols = ['LOCID', 'Year', 'Month', 'DAYNUM','HR_LOCAL', 'ETMS_DEP', 'ETMS_ARR', 'MC', 'CEILING', 'VISIBLE', 'TEMP', 'WND_ANGL', 'WND_SPED']
wx = wx2008[wxcols]
# create a variable address the number of scheduled operations (both departures and arrivals) 
wx['Demand'] = wx['ETMS_DEP'] + wx['ETMS_ARR']
del wx2008

# We check the resulting dataset to see if we have the right columns and the duplicate the dataset since we need information from both the departure and the arrival airports.
wx.head()
#   LOCID  Year  Month  DAYNUM  HR_LOCAL  ETMS_DEP  ETMS_ARR MC  CEILING VISIBLE TEMP  Demand
# 0   ABQ  2008      1       5         0         0         5  V    200.0   10.00  35        5
# 1   ABQ  2008      1       5         1         1         1  V    130.0   10.00  37        2
# 2   ABQ  2008      1       5         2         0         3  V    130.0   10.00  37        3
# 3   ABQ  2008      1       5         3         2         1  V    130.0   10.00  37        3
# 4   ABQ  2008      1       5         4         2         1  V    120.0   10.00  39        3

wx.drop('ETMS_DEP', axis=1, inplace=True)
wx.drop('ETMS_ARR', axis=1, inplace=True)
wx.rename(columns={'LOCID': 'Origin', 'HR_LOCAL': 'DepHr', 'DAYNUM': 'DayofMonth','Demand':'Origin_Demand'}, inplace=True)
wx.head()

#   Origin  Year  Month  DayofMonth  DepHr MC  CEILING VISIBLE TEMP  Origin_Demand
# 0    ABQ  2008      1           5      0  V    200.0   10.00  35               5
# 1    ABQ  2008      1           5      1  V    130.0   10.00  37               2
# 2    ABQ  2008      1           5      2  V    130.0   10.00  37               3
# 3    ABQ  2008      1           5      3  V    130.0   10.00  37               3
# 4    ABQ  2008      1           5      4  V    120.0   10.00  39               3

wx_arr = wx.copy()
wx_arr.rename(columns={'Origin': 'Dest', 'DepHr': 'ArrHr','Origin_Demand':'Dest_Demand'}, inplace=True)
wx.rename(columns={'MC': 'MC_DEP', 'CEILING': 'CEILING_DEP', 'VISIBLE': 'VISIBLE_DEP', 'TEMP': 'TEMP_DEP'}, inplace=True)
wx_arr.rename(columns={'MC': 'MC_ARR', 'CEILING': 'CEILING_ARR', 'VISIBLE': 'VISIBLE_ARR', 'TEMP': 'TEMP_ARR'}, inplace=True)
wx.head()
wx_arr.head()

#   Origin  Year  Month  DayofMonth  DepHr MC_DEP  CEILING_DEP VISIBLE_DEP TEMP_DEP  Origin_Demand
# 0    ABQ  2008      1           5      0      V        200.0       10.00      35               5
# 1    ABQ  2008      1           5      1      V        130.0       10.00      37               2
# 2    ABQ  2008      1           5      2      V        130.0       10.00      37               3
# 3    ABQ  2008      1           5      3      V        130.0       10.00      37               3
# 4    ABQ  2008      1           5      4      V        120.0       10.00      39               3

#   Dest  Year  Month  DayofMonth  ArrHr MC_ARR  CEILING_ARR VISIBLE_ARR TEMP_ARR  Dest_Demand
# 0  ABQ  2008      1           5      0      V        200.0       10.00      35             5
# 1  ABQ  2008      1           5      1      V        130.0       10.00      37             2
# 2  ABQ  2008      1           5      2      V        130.0       10.00      37             3
# 3  ABQ  2008      1           5      3      V        130.0       10.00      37             3
# 4  ABQ  2008      1           5      4      V        120.0       10.00      39             3

#We check the types of resulting dataset. Some of the variables are numeric values stored as text. We will need to convert them to numeric in order to use in the analysis.
wx_arr.get_dtype_counts()
# float64    1
# int64      5
# object     4
# dtype: int64

# Add Origin Information
temp = pd.merge(flt2008, wx, on =['Origin', 'Year', 'Month','DayofMonth','DepHr'], how ='inner')
# Add Destination Information
flt2008wtr = pd.merge(temp, wx_arr, on =['Dest', 'Year', 'Month','DayofMonth','ArrHr'], how ='inner')
del temp, flt2008
flt2008wtr.head().T

#                         0       1       2       3       4
# Year                 2008    2008    2008    2008    2008
# Month                   1       1       1       1       1
# DayofMonth              3       3       3       3       3
# DayOfWeek               4       4       4       4       4
# DepTime              2003    2010    1901    1916    1934
# CRSDepTime           1955    1925    1855    1915    1930
# ArrTime              2211    2237    2155    2207    2159
# CRSArrTime           2225    2220    2210    2215    2204
# UniqueCarrier          WN      WN      WN      AA      CO
# FlightNum             335     929    3580     543     175
# TailNum            N712SW  N521SW  N206WN  N507AA  N16339
# ActualElapsedTime     128     147     174     171     145
# CRSElapsedTime        150     175     195     180     154
# AirTime               116     129     157     146     123
# ArrDelay              -14      17     -15      -8      -5
# DepDelay                8      45       6       1       4
# Origin                IAD     PHL     PVD     LGA     CLE
# Dest                  TPA     TPA     TPA     TPA     TPA
# Distance              810     920    1137    1011     927
# TaxiIn                  4       5       5       2       7
# TaxiOut                 8      13      12      23      15
# Cancelled               0       0       0       0       0
# CancellationCode      NaN     NaN     NaN     NaN     NaN
# Diverted                0       0       0       0       0
# CarrierDelay          NaN       3     NaN     NaN     NaN
# WeatherDelay          NaN       0     NaN     NaN     NaN
# NASDelay              NaN       0     NaN     NaN     NaN
# SecurityDelay         NaN       0     NaN     NaN     NaN
# LateAircraftDelay     NaN      14     NaN     NaN     NaN
# DepHr                  19      19      18      19      19
# ArrHr                  22      22      22      22      22
# MC_DEP                  V       V       V       V       V
# CEILING_DEP           NaN     NaN     NaN     NaN     NaN
# VISIBLE_DEP         10.00   10.00   10.00   10.00   10.00
# TEMP_DEP              25      23      13      20      18
# Origin_Demand          42      95      20      67      48
# MC_ARR                  V       V       V       V       V
# CEILING_ARR           NaN     NaN     NaN     NaN     NaN
# VISIBLE_ARR         10.00   10.00   10.00   10.00   10.00
# TEMP_ARR              42      42      42      42      42
# Dest_Demand            28      28      28      28      28


flt2008wtr.shape[1]
# 41 --------------------------- 41 Features
flt2008wtr.shape[0]
# 4919764 ---------------------- Almost 5 million Records


#----- Error In Dataset
# flt2008wtr.iloc[23816]
# 
# Year                   2008
# Month                     1
# DayofMonth                4
# DayOfWeek                 5
# DepTime                 842
# CRSDepTime              845
# ArrTime                1025
# CRSArrTime             1018
# UniqueCarrier            UA
# FlightNum               477
# TailNum              N808UA
# ActualElapsedTime       163
# CRSElapsedTime          153
# AirTime                 140
# ArrDelay                  7
# DepDelay                 -3
# Origin                  DEN
# Dest                    SMF
# Distance                910
# TaxiIn                    6
# TaxiOut                  17
# Diverted                  0
# DepHr                     8
# ArrHr                    10
# MC_DEP                    V
# CEILING_DEP             200
# VISIBLE_DEP           10.00
# TEMP_DEP                37
# Origin_Demand           104
# MC_ARR                    I
# CEILING_ARR             NaN
# VISIBLE_ARR               M
# TEMP_ARR                 M
# Dest_Demand               4
# Name: 23816, dtype: object
#
# VISIBLE_ARR = M ann TEMP_ARR are Not Numeric Hence Conversion Is Needed
#
#Converting data to numerical
flt2008wtr['VISIBLE_DEP'] = flt2008wtr['VISIBLE_DEP'].convert_objects(convert_numeric=True)
flt2008wtr['VISIBLE_ARR'] = flt2008wtr['VISIBLE_ARR'].convert_objects(convert_numeric=True)
flt2008wtr['CEILING_DEP'] = flt2008wtr['CEILING_DEP'].convert_objects(convert_numeric=True)
flt2008wtr['CEILING_ARR'] = flt2008wtr['CEILING_ARR'].convert_objects(convert_numeric=True)
flt2008wtr['TEMP_DEP'] = flt2008wtr['TEMP_DEP'].convert_objects(convert_numeric=True)
flt2008wtr['TEMP_ARR'] = flt2008wtr['TEMP_ARR'].convert_objects(convert_numeric=True)
flt2008wtr['WND_ANGL_x'] = flt2008wtr['WND_ANGL_x'].convert_objects(convert_numeric=True)
flt2008wtr['WND_ANGL_y'] = flt2008wtr['WND_ANGL_y'].convert_objects(convert_numeric=True)
flt2008wtr['WND_SPED_x'] = flt2008wtr['WND_SPED_x'].convert_objects(convert_numeric=True)
flt2008wtr['WND_SPED_y'] = flt2008wtr['WND_SPED_y'].convert_objects(convert_numeric=True)
#Index([u'Year', u'Month', u'DayofMonth', u'DayOfWeek', u'DepTime', u'CRSDepTime', u'ArrTime', u'CRSArrTime', u'UniqueCarrier', u'FlightNum', u'TailNum', u'ActualElapsedTime', u'CRSElapsedTime', u'AirTime', u'ArrDelay', u'DepDelay', u'Origin', u'Dest', u'Distance', u'TaxiIn', u'TaxiOut', u'Diverted', u'DepHr', u'ArrHr', u'MC_DEP', u'CEILING_DEP', u'VISIBLE_DEP', u'TEMP_DEP', u'Origin_Demand', u'MC_ARR', u'CEILING_ARR', u'VISIBLE_ARR', u'TEMP_ARR', u'Dest_Demand'], dtype='object')


# Build feature sets for our analysis
# separate classes based on 15 minute delay
delay_threshold = 15.
# choose the direction
direction = 'Origin'
#direction = 'Dest'

Ycol = 'DepDelay'

if direction == 'Origin':
    # Features for Departure Delays
    Xcols = ['Month','DayOfWeek','DepHr','ArrHr','UniqueCarrier','Dest',
                 'CRSElapsedTime','Distance', 'Origin_Demand', 'MC_DEP', 
                  'VISIBLE_DEP', 'TEMP_DEP', 'Dest_Demand', 
                 'MC_ARR',  'VISIBLE_ARR', 'TEMP_ARR', 'WND_ANGL_x', 'WND_ANGL_y', 'WND_SPED_x', 'WND_SPED_y']
else:
    # Features for Arrival Delays
    Xcols = ['Month','DayOfWeek','DepHr','ArrHr','UniqueCarrier','Origin',
                 'CRSElapsedTime','Distance', 'Origin_Demand', 'MC_DEP', 
                  'VISIBLE_DEP', 'TEMP_DEP', 'Dest_Demand', 
                 'MC_ARR', 'VISIBLE_ARR', 'TEMP_ARR', 'WND_ANGL_x', 'WND_ANGL_y', 'WND_SPED_x', 'WND_SPED_y']

# Specify frames for modelling
cols = [direction]+[Ycol]+Xcols

#Dropping NaN
flt2008wtr =  flt2008wtr.dropna()

#Extracting DataSet for future Demonstration
flt2008wtr.to_csv("flt2008wtr.csv", sep="\t", encoding="utf-8")


