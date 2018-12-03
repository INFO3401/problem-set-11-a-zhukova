import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols


# look at what types of dependednt and independent variables you have, choose your test accordingly

# Monday Problem 1: What statistical test would you use for the following scenarios? 
# (a) Does a student's current year (e.g., freshman, sophomore, etc.) effect their GPA?
# ANOVA
#(b) Has the amount of snowfall in the mountains changed over time? 
#  Genrealized regression
#(c) Over the last 10 years, have there been more hikers on average in Estes Park in the spring or summer?
# T-test
#(d) Does a student's home state predict their highest degree level?
# Chi-squared


######################################################################
################ Problem 2 CODE    ###################################
######################################################################

# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

#run a t-test

def runTTest(indepVarA, indepVarB, depVar):
	ttest = scipy.stats.ttest_ind(indepVarA[depVar], indepVarB[depVar])
	print(ttest)

def runANOVA(data, formula):
	model = ols(formula, data).fit()
	aov_table = sm.stats.anova_lm(model, typ = 2)
	print(aov_table)
	

def runLinearRegression(data, formula):
	lm = ols(formula, data).fit()
	#linear_model.LinearRegression()
	#lm.fit(X, target)
	print(lm.summary())

#Run the analsys, extract data
rawData, df = generateDataset('simpsons_paradox.csv')
df['total'] = df['Admitted'] + df['Rejected']
df['AdmittedPercent'] = df['Admitted']/df['total']
df['RejectedPercent'] = df['Rejected']/df['total']

print('Does gender correlate with admissions?')
men = df[(df['Gender'] == 'Male')]
women = df[(df['Gender'] == 'Female')]
print('Admitted column')
runTTest(men, women, 'Admitted')
print('AdmittedPercent column')
runTTest(men, women, 'AdmittedPercent')
print(' ')

#failed to reject the null hyp that both genders have the same acceptance rate

print('Does department correlate with admissions?')
#dep var has more than 2 possible settings -- ANOVA
simpleFormula = 'Admitted ~ C(Department)'
percentFormula = 'AdmittedPercent ~ C(Department)'
#runANOVA(rawData, simpleFormula)
runANOVA(df, percentFormula)
print('')

print('Do gender and department correlate with admissions?')
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
#C denotes categories that you want to run the test on
moreComplexP = 'AdmittedPercent ~ C(Department) + C(Gender)'
#runANOVA(rawData, moreComplex)
runANOVA(df, moreComplexP)
print('')

#C creates on hot encoding of the specified column

print('Linear Regression')
linearFormula = 'AdmittedPercent ~ Mean_Age + C(Department) + C(Gender)'
linearFormula2 = 'AdmittedPercent ~ Mean_Age'
runLinearRegression(df, linearFormula)
runLinearRegression(df, linearFormula2)

# Tells you how much each dep't differs from Dep A (reference category)

print(df[['AdmittedPercent', 'Admitted', 'Mean_Age']].corr())
#negative correlation, departments with older applicants have a lower admitted %

# What factors appear to contribute most heavily to admissions? 



#print(df)
#print(rawData)
#compute % admitted, use that as dep var
#use a pd groupby for admitted/rejected columns
# Gender isn't significant to admissions when using the % Admitted


######################################################################
################ Problem 2 Answers ###################################
######################################################################

# What factors appear to contribute most heavily to admissions?
# It appears that department correlates most heavily with admissions

# Do you think the admissions process is biased based on the available data? Why or why not?
# The admissions process does appear to be biased by gender when you're using the Admitted and Rejected columns 
# but it does not appear to be biased when you're using the AdmittedPercent and RejectedPercent columns.


######################################################################
################ Problem 3 Answers ###################################
######################################################################

# Does gender correlate with admissions?
# 
# Wrong data: 
# T-test Admitted column p value: 0.22859473251723653
# T-test AdmittedPercent column p value: 0.30746484171824817
# Right data:
# T-test Admitted column p value: 0.65721545804125947
# T-test AdmittedPercent column p value: 0.809364334120141

# Clearly, the dirty data made the significance between gender and admissions much more pronounced than it actually was

# Does department correlate with admissions?

# Wrong data ANOVA p-value: 0.000308
# Right data ANOVA p-value: 0.000308
# No change between dirty and clean data in terms of p-value for department and Admitted significance


# Do gender and department correlate with admissions?

# Wrong data:
# Gender ANOVA p-vlaue: 0.408355
# Department ANOVA p-value: 0.004043

# Rigt data:
# Gender ANOVA p-value: 0.362950
# Department ANOVA p-value: 0.000864

# Clearly the wrong data was giving us different values than the right data but they were
# still statistically significant in both cases.


# Linear regression:

# Wrong data:
#                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept              0.3609      0.703      0.514      0.643      -1.875       2.597
# C(Department)[T.B]    -0.0424      0.080     -0.531      0.632      -0.296       0.212
# C(Department)[T.C]    -0.3674      0.067     -5.489      0.012      -0.580      -0.154
# C(Department)[T.D]    -0.4100      0.080     -5.147      0.014      -0.664      -0.156
# C(Department)[T.E]    -0.4836      0.084     -5.732      0.011      -0.752      -0.215
# C(Department)[T.F]    -0.7020      0.078     -9.012      0.003      -0.950      -0.454
# C(Gender)[T.Male]      0.0028      0.055      0.052      0.962      -0.171       0.177
# C(Gender)[T.Male ]    -0.0528      0.067     -0.789      0.488      -0.266       0.160
# Mean_Age               0.0165      0.029      0.569      0.609      -0.076       0.109

# Right data:
# ======================================================================================
#                          coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept              0.1480      0.573      0.258      0.809      -1.443       1.738
# C(Department)[T.B]    -0.0299      0.072     -0.418      0.698      -0.228       0.169
# C(Department)[T.C]    -0.3674      0.062     -5.940      0.004      -0.539      -0.196
# C(Department)[T.D]    -0.3822      0.062     -6.180      0.003      -0.554      -0.210
# C(Department)[T.E]    -0.4516      0.063     -7.168      0.002      -0.626      -0.277
# C(Department)[T.F]    -0.6825      0.066    -10.290      0.001      -0.867      -0.498
# C(Gender)[T.Male]     -0.0194      0.039     -0.497      0.645      -0.128       0.089
# Mean_Age               0.0249      0.024      1.038      0.358      -0.042       0.091

# Clearly, the wrong data was giving us skewed numbers for the p-value because the
# 'Male ' variable was messing everythig up. The right data more thoghroughly showed
# the siginificance of a given department on Acceptance rate and made the significance of 
# gender on acceptance much less significant.


#Correlation: 

#Wrong data:
#                  AdmittedPercent  Admitted  Mean_Age
# AdmittedPercent         1.000000  0.430335 -0.346068
# Admitted                0.430335  1.000000 -0.445492
# Mean_Age               -0.346068 -0.445492  1.000000

# Right data:
#                  AdmittedPercent  Admitted  Mean_Age
# AdmittedPercent         1.000000  0.430335 -0.346068
# Admitted                0.430335  1.000000 -0.445492
# Mean_Age               -0.346068 -0.445492  1.000000

# The correlation showed no difference between the outputs of the wrong data and the correct data.

print(df)